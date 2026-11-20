#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Horse Racing Regressor – Prédiction de valeurs, probabilités & cotes
Version améliorée avec features enrichies et hyperparamètres optimisés
"""

import os, time, pickle, warnings
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import optuna
import matplotlib.pyplot as plt

try:
    from optuna.integration import LightGBMPruningCallback
    USE_PRUNE = True
except ImportError:
    USE_PRUNE = False


# ---------- utilitaire soft‑max amélioré ----------
def softmax_pct(scores: np.ndarray, exp: float = 1.5) -> np.ndarray:
    """Convertit des scores en pourcentages via softmax avec contrôle de température"""
    # Détecter automatiquement si les scores doivent être inversés (pour temps de course)
    invert = np.median(scores) < 0 or (np.min(scores) < 0 and np.max(scores) > 0)
    if invert:
        scores = -scores
        
    shifted = scores - scores.max()
    e = np.exp(shifted * exp)
    return (e / e.sum()) * 100     # pourcent


class HorseRacingRegressor:
    SOFTMAX_EXP = 2.2  # Légèrement augmenté pour mieux différencier les probabilités
    MIN_PCT     = 0.4  # Légèrement réduit

    def __init__(self):
        self.data = None
        self.model = None
        self.feature_names = []
        self.categorical_features = []
        self.feature_importance = None
        
        # Paramètres par défaut légèrement améliorés
        self.lgb_params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.025,
            "num_leaves": 95,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 5,
            "min_data_in_leaf": 20,
            "verbosity": -1,
            "random_state": 42,
            "device_type": "cpu",
            "lambda_l1": 0.2,
            "lambda_l2": 1.0,
            "max_depth": 12,           # Limite la profondeur pour éviter l'overfitting
            "min_gain_to_split": 0.01, # Évite les splits non significatifs
            "max_bin": 255,            # Nombre de bins pour les features numériques
        }

    # 1. Chargement -----------------------------------------------------------------
    def load(self, path: str):
        print("Loading data…")
        t0 = time.time()
        self.data = (
            pd.read_excel(path)
            if path.lower().endswith(".xlsx")
            else pd.read_csv(path, delimiter=";", low_memory=False)
        )
        print(f"{len(self.data):,} rows ({time.time() - t0:.2f}s)")

        if "ID_PV_COURSE" not in self.data.columns and "ID_COURSE" in self.data.columns:
            self.data.rename(columns={"ID_COURSE": "ID_PV_COURSE"}, inplace=True)

        # ne garder que les partants classés
        self.data = self.data[self.data["PLACE"].notna() & (self.data["PLACE"] != 0)]

        if "TEMPS_COURSE" in self.data.columns:
            self.data["target_value"] = pd.to_numeric(self.data["TEMPS_COURSE"], errors="coerce")
        else:
            # Target améliorée: utilisation d'une transformation plus adaptée
            # 1/PLACE donne trop d'importance aux premières places
            # log(1/PLACE) donne une distribution plus équilibrée
            self.data = self.data[self.data["PLACE"].notna() & (self.data["PLACE"] > 0)]
            self.data["target_value"] = np.log1p(1 / self.data["PLACE"])

        for col in ("DATE_COURSE", "DATE_NAISSANCE"):
            if col in self.data.columns:
                self.data[col] = pd.to_datetime(self.data[col], errors="coerce")

        if "DATE_COURSE" in self.data.columns:
            self.data.sort_values("DATE_COURSE", inplace=True)

    # 2. Split temporel --------------------------------------------------------------
    def split_data(self, test_frac: float = 0.2):
        if "DATE_COURSE" in self.data.columns:
            cut_date = self.data["DATE_COURSE"].quantile(1 - test_frac)
            train = self.data[self.data["DATE_COURSE"] < cut_date].copy()
            test  = self.data[self.data["DATE_COURSE"] >= cut_date].copy()
            print(f"Split temporel : train ≤ {cut_date:%d-%m-%Y}")
        else:
            self.data = self.data.sample(frac=1, random_state=42)
            cut = int(len(self.data) * (1 - test_frac))
            train, test = self.data.iloc[:cut].copy(), self.data.iloc[cut:].copy()
        return train, test

    # 3. Feature engineering ---------------------------------------------------------
    def preprocess_data(self, df: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        dfc = df.copy()
        
        # Convertir PLACE en numérique et supprimer les non-partants
        dfc['PLACE'] = pd.to_numeric(dfc['PLACE'], errors='coerce').fillna(0).astype(int)
        
        # Supprimer les doublons (même course / même cheval)
        id_course_col = 'ID_PV_COURSE' if 'ID_PV_COURSE' in dfc.columns else 'ID_COURSE'
        dfc = dfc.drop_duplicates(subset=[id_course_col, 'ID_CHEVAL'])
        
        # Retirer les non-partants
        dfc = dfc[dfc['PLACE'] != 0].copy()
        
        # Nombre de partants par course
        dfc['num_starters'] = dfc.groupby(id_course_col)['ID_CHEVAL'].transform('size')
        
        # Probabilité « rang inversé »
        dfc['proba'] = (dfc['num_starters'] - dfc['PLACE'] + 1) / dfc['num_starters']
        
        # Statistiques par cheval
        horse_perf = (
            dfc.groupby('ID_CHEVAL')
              .agg(
                  horse_participations = ('PLACE', 'size'),
                  horse_wins           = ('PLACE', lambda s: (s == 1).sum()),
                  horse_podiums        = ('PLACE', lambda s: (s <= 3).sum()),
                  horse_avg_rank       = ('PLACE', 'mean')
              )
        )
        
        horse_perf['horse_win_rate'] = horse_perf['horse_wins'] / horse_perf['horse_participations']
        horse_perf['horse_podium_rate'] = horse_perf['horse_podiums'] / horse_perf['horse_participations']
        
        dfc = dfc.merge(horse_perf.drop(columns=['horse_wins', 'horse_podiums']),
                      left_on='ID_CHEVAL', right_index=True, how='left')
        
        # Statistiques par paire cheval-jockey
        pair_perf = (
            dfc.groupby(['ID_CHEVAL', 'ID_JOCKEY'])
              .agg(
                  pair_participations = ('PLACE', 'size'),
                  pair_wins           = ('PLACE', lambda s: (s == 1).sum()),
                  pair_podiums        = ('PLACE', lambda s: (s <= 3).sum()),
                  pair_avg_rank       = ('PLACE', 'mean')
              )
        )
        
        pair_perf['pair_win_rate'] = pair_perf['pair_wins'] / pair_perf['pair_participations']
        pair_perf['pair_podium_rate'] = pair_perf['pair_podiums'] / pair_perf['pair_participations']
        
        dfc = dfc.merge(pair_perf.drop(columns=['pair_wins', 'pair_podiums']),
                       on=['ID_CHEVAL', 'ID_JOCKEY'], how='left')
        
        # Feature "course_size" qui est potentiellement redondante avec num_starters
        dfc["course_size"] = dfc.groupby(id_course_col)["ID_CHEVAL"].transform("size")

        if "DATE_COURSE" in dfc.columns:
            dfc["jour_semaine"] = dfc["DATE_COURSE"].dt.weekday
            dfc["mois"] = dfc["DATE_COURSE"].dt.month
            dfc["annee"] = dfc["DATE_COURSE"].dt.year
            # Nouvelles features temporelles
            dfc["trimestre"] = (dfc["mois"] - 1) // 3 + 1
            dfc["jour_annee"] = dfc["DATE_COURSE"].dt.dayofyear
            dfc["jour_mois"] = dfc["DATE_COURSE"].dt.day

        if "POIDS_REEL" in dfc.columns:
            dfc["POIDS_REEL"] = pd.to_numeric(dfc["POIDS_REEL"], errors="coerce")
            dfc["log_weight"] = np.log1p(dfc["POIDS_REEL"].clip(lower=0))
            dfc["delta_weight"] = dfc["POIDS_REEL"] - dfc.groupby(id_course_col)["POIDS_REEL"].transform("mean")
            # Nouvelles features de poids relatif
            dfc["weight_rank"] = dfc.groupby(id_course_col)["POIDS_REEL"].rank(pct=True)
            dfc["weight_zscore"] = dfc.groupby(id_course_col)["POIDS_REEL"].transform(
                lambda x: (x - x.mean()) / x.std() if len(x) > 1 and x.std() > 0 else 0
            )

        if "SEXE" in dfc.columns:
            dfc["sexe_encoded"] = dfc["SEXE"].map({"M": 1, "F": 0}).fillna(0).astype("int8")

        if {"DATE_COURSE", "DATE_NAISSANCE"}.issubset(dfc.columns):
            dfc["horse_age_at_race"] = ((dfc["DATE_COURSE"] - dfc["DATE_NAISSANCE"]).dt.days / 365.25).fillna(0)
            # Nouvelle feature: catégorie d'âge
            dfc["age_category"] = pd.cut(
                dfc["horse_age_at_race"], 
                bins=[0, 3, 5, 8, 20], 
                labels=[0, 1, 2, 3]
            ).fillna(0).astype(int)

        if "DATE_COURSE" in dfc.columns:
            dfc.sort_values(["ID_CHEVAL", "DATE_COURSE"], inplace=True)
            dfc["days_from_last_race"] = dfc.groupby("ID_CHEVAL")["DATE_COURSE"].diff().dt.days.fillna(365)
            # Feature de fraîcheur - exponentielle décroissante
            dfc["freshness"] = np.clip(np.exp(-dfc["days_from_last_race"] / 60), 0, 1)
            # Inactivité trop longue (binaire)
            dfc["long_inactive"] = (dfc["days_from_last_race"] > 180).astype(int)

        # Stats historiques améliorées
        dfc["performance_value"] = dfc["PLACE"]
        
        # --- Stats par entité (cheval, jockey, entraîneur) ---
        for key, entity in {
            "horse": "ID_CHEVAL", 
            "jockey": "ID_JOCKEY",
            "trainer": "ID_ENTRAINEUR"
        }.items():
            if entity in dfc.columns:
                dfc.sort_values([entity, "DATE_COURSE"], inplace=True)
                
                # Moyenne pondérée exponentielle (donne plus d'importance aux courses récentes)
                dfc[f"{key}_ewm_perf"] = dfc.groupby(entity)["performance_value"].shift(1).ewm(span=5).mean()
                
                # Stats sur différentes fenêtres temporelles
                for N in (3, 5, 10):
                    dfc[f"{key}_last{N}_mean"] = (
                        dfc.groupby(entity)["performance_value"].shift(1).rolling(N, min_periods=1).mean()
                    )
                    # Écart-type pour capturer la stabilité des performances
                    dfc[f"{key}_last{N}_std"] = (
                        dfc.groupby(entity)["performance_value"].shift(1).rolling(N, min_periods=3).std().fillna(0)
                    )
                    # Tendance (pente) des performances récentes
                    if N >= 5:
                        window = dfc.groupby(entity)["performance_value"].shift(1).rolling(N, min_periods=3)
                        ranks = np.arange(1, N+1)
                        def trend_slope(x):
                            if len(x) >= 3:
                                valid_x = x.dropna()
                                if len(valid_x) >= 3:
                                    # Calcul de la pente de régression linéaire simplifiée
                                    y = valid_x.values
                                    x = np.arange(len(y))
                                    return np.polyfit(x, y, 1)[0] if len(x) > 1 else 0
                            return 0
                        dfc[f"{key}_last{N}_trend"] = window.apply(trend_slope).fillna(0)
                
                # Les features de ratio de victoires et podiums sont déjà calculées plus haut
                # pour le cheval, et peuvent être calculées de la même façon pour jockey et entraîneur
                if key != "horse":  # Éviter la duplication pour les chevaux
                    wins = dfc.groupby(entity)["PLACE"].transform(
                        lambda x: x.rolling(10, min_periods=1).apply(lambda y: (y == 1).mean())
                    ).shift(1).fillna(0)
                    dfc[f"{key}_win_rate"] = wins
                    
                    podiums = dfc.groupby(entity)["PLACE"].transform(
                        lambda x: x.rolling(10, min_periods=1).apply(lambda y: (y <= 3).mean())
                    ).shift(1).fillna(0)
                    dfc[f"{key}_podium_rate"] = podiums
                
                # Nombre de courses total (experience) - pour jockey et entraîneur
                if key != "horse":  # Déjà calculé pour les chevaux
                    dfc[f"{key}_race_count"] = dfc.groupby(entity).cumcount().shift(1).fillna(0)
        
        # --- Interactions entre entités ---
        if {"ID_CHEVAL", "ID_JOCKEY"}.issubset(dfc.columns):
            # Combinaison cheval-jockey (familiarité) - complète les statistiques de pair déjà calculées
            dfc["horse_jockey"] = dfc["ID_CHEVAL"].astype(str) + "_" + dfc["ID_JOCKEY"].astype(str)
            dfc["horse_jockey_familiarity"] = np.minimum(dfc["pair_participations"] / 5, 1.0)
        
        # --- Stats par course ---
        # Compétitivité de la course
        if "COTE_DIRECT" in dfc.columns:
            dfc["COTE_DIRECT"] = pd.to_numeric(dfc["COTE_DIRECT"], errors="coerce")
            dfc["odds_rank"] = dfc.groupby(id_course_col)["COTE_DIRECT"].rank(pct=True)
            dfc["odds_vs_median"] = dfc["COTE_DIRECT"] / dfc.groupby(id_course_col)["COTE_DIRECT"].transform("median")
            
        # --- Features contextuelles ---
        if "CODE_HIPPODROME" in dfc.columns:
            # Stats par hippodrome pour chaque entité
            for entity, col in [("horse", "ID_CHEVAL"), ("jockey", "ID_JOCKEY")]:
                if col in dfc.columns:
                    # Clé combinée entité + hippodrome
                    dfc[f"{entity}_track"] = dfc[col].astype(str) + "_" + dfc["CODE_HIPPODROME"].astype(str)
                    # Performance moyenne sur cet hippodrome
                    track_perf = dfc.groupby(f"{entity}_track")["PLACE"].transform(
                        lambda x: x.shift(1).rolling(min_periods=1, window=10).mean()
                    ).fillna(dfc.groupby(col)["PLACE"].transform(
                        lambda x: x.shift(1).rolling(min_periods=1, window=5).mean()
                    ))
                    dfc[f"{entity}_track_perf"] = track_perf
                    
        # --- Features de distance ---
        if "DISTANCE_COURSE" in dfc.columns:
            dfc["DISTANCE_COURSE"] = pd.to_numeric(dfc["DISTANCE_COURSE"], errors="coerce")
            # Distance normalisée (pour comparer entre disciplines)
            if "CODE_DISCIPLINE_COURSE" in dfc.columns:
                dfc["distance_norm"] = dfc["DISTANCE_COURSE"] / dfc.groupby("CODE_DISCIPLINE_COURSE")["DISTANCE_COURSE"].transform("median")
                
                # Performance par distance pour chaque cheval
                dfc["horse_distance"] = dfc["ID_CHEVAL"].astype(str) + "_" + (
                    pd.cut(dfc["distance_norm"], bins=[0, 0.8, 1.2, 2.0], labels=["court", "moyen", "long"])
                    .fillna("moyen").astype(str)
                )
                dfc["horse_distance_perf"] = dfc.groupby("horse_distance")["PLACE"].transform(
                    lambda x: x.shift(1).rolling(min_periods=1, window=5).mean()
                ).fillna(dfc.groupby("ID_CHEVAL")["PLACE"].transform(
                    lambda x: x.shift(1).rolling(min_periods=1, window=3).mean()
                ))

        # S'assurer qu'il n'y a pas de valeurs manquantes
        dfc.fillna(0, inplace=True)
        return dfc
    
    # 4. Sélection des features ------------------------------------------------------
    def build_features(self, df: pd.DataFrame):
        num = df.select_dtypes("number").columns.tolist()
        blacklist = {
            "target_value", "course_size", "PLACE", "performance_value",
            "ID_COURSE", "ID_PV_COURSE", "ID_SOCIETE_COURSE", "ID_CHEVAL",
            "ID_JOCKEY", "ID_PROPRIETAIRE", "ID_ENTRAINEUR", "ID_NAISSEUR",
            "NUMERO_PMU_SEC", "VH", "NUMERO_PMU_DC",
            "ID_PV_COURSE_1", "ID_JOCKEY_DC", "POIDS_REEL", "num_starters",
            "jour_semaine", "mois", "annee", "log_weight", "proba",
            "probability_pct", "odds", "norm_prediction", "horse_jockey","pair_participations"
        }
        num = [c for c in num if c not in blacklist]

        self.categorical_features = [c for c in (
            "CODE_DISTANCE_COURSE", "CODE_DISCIPLINE_COURSE",
            "CODE_RACE", "CODE_CATEGORIE_COURSE", "CODE_HIPPODROME", 
            "age_category"  # Nouvelle catégorie ajoutée
        ) if c in df.columns]

        self.feature_names = list(dict.fromkeys(num + self.categorical_features))
        print(f"Features finales : {len(self.feature_names)} features")
        return self.feature_names

    # 5. Préparation LightGBM --------------------------------------------------------
    def prepare_for_training(self, df: pd.DataFrame):
        dfc = df.copy()
        for cat in self.categorical_features:
            dfc[cat] = dfc[cat].astype("category").cat.codes
        X = dfc[self.feature_names]
        y = dfc["target_value"]
        return X, y, dfc

    # 6. Recherche Optuna -----------------------------------------------------------
    def optuna_search(self, X_train, y_train):
        def objective(trial):
            p = {
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
                "num_leaves": trial.suggest_int("num_leaves", 31, 255),
                "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.15, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
                "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
                "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
                "max_depth": trial.suggest_int("max_depth", 5, 20),
                "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.001, 0.05),
                "max_bin": trial.suggest_int("max_bin", 100, 500),
                "verbosity": -1,
                "random_state": 42,
                "device_type": "cpu",
            }

            # Validation temporelle avec plus de folds pour une meilleure estimation
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            for tr_idx, val_idx in tscv.split(X_train):
                X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
                X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
                dtr = lgb.Dataset(X_tr, y_tr, feature_name=self.feature_names,
                                  categorical_feature=self.categorical_features)
                dval = lgb.Dataset(X_val, y_val, reference=dtr,
                                   feature_name=self.feature_names,
                                   categorical_feature=self.categorical_features)
                callbacks = [lgb.early_stopping(50, verbose=False)]
                if USE_PRUNE:
                    callbacks.insert(0, LightGBMPruningCallback(trial, "rmse"))
                model = lgb.train(p, dtr, 500, valid_sets=[dval], valid_names=["val"],
                                  callbacks=callbacks)
                preds = model.predict(X_val, num_iteration=model.best_iteration)
                scores.append(np.sqrt(mean_squared_error(y_val, preds)))
            return np.mean(scores)

        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        # Augmentation des trials pour une meilleure exploration
        study.optimize(objective, n_trials=200, show_progress_bar=True)
        self.lgb_params.update(study.best_trial.params)
        print("🎯 Paramètres optimaux :", study.best_trial.params)

    # 7. Entraînement final ----------------------------------------------------------
    def train_final(self, X_train, y_train, X_test, y_test):
        dtr = lgb.Dataset(X_train, y_train, feature_name=self.feature_names,
                          categorical_feature=self.categorical_features)
        dte = lgb.Dataset(X_test, y_test, reference=dtr, feature_name=self.feature_names,
                          categorical_feature=self.categorical_features)

        self.model = lgb.train(
            self.lgb_params, dtr, num_boost_round=1000,  # Plus d'iterations potentielles
            valid_sets=[dtr, dte], valid_names=["train", "val"],
            callbacks=[lgb.early_stopping(50, verbose=True),  # Patient early stopping
                       lgb.log_evaluation(200)]
        )

        # Importance des features
        imp = pd.DataFrame({
            "Feature": self.feature_names,
            "Gain": self.model.feature_importance(importance_type="gain")
        }).sort_values("Gain", ascending=False)
        self.feature_importance = imp  # Sauvegarde pour analyse ultérieure
        
        # Visualisation plus élaborée
        plt.figure(figsize=(12, 8))
        plt.barh(imp["Feature"][:20], imp["Gain"][:20])
        plt.gca().invert_yaxis()
        plt.title("Top 20 des features les plus importantes (gain)")
        plt.tight_layout()
        plt.savefig("feature_importance_regressor.png", dpi=300)
        
        # Afficher les features les plus importantes
        print("\nTop 10 features les plus importantes:")
        for i, (feat, gain) in enumerate(zip(imp["Feature"][:10], imp["Gain"][:10]), 1):
            print(f"{i}. {feat}: {gain:.2f}")

    # 8. Évaluation -----------------------------------------------------------------
    def evaluate_model(self, X_test, y_test):
        preds = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae  = mean_absolute_error(y_test, preds)
        r2   = r2_score(y_test, preds)
        print(f"\nÉvaluation RMSE {rmse:.4f} | MAE {mae:.4f} | R² {r2:.4f}")
        
        # Analyse des erreurs par quantile
        error = np.abs(y_test - preds)
        error_df = pd.DataFrame({'true': y_test, 'pred': preds, 'error': error})
        error_quantiles = error_df.groupby(pd.qcut(error_df['true'], 5))['error'].agg(['mean', 'count'])
        print("\nAnalyse des erreurs par quantile de la valeur cible:")
        print(error_quantiles)
        
        return {'rmse': rmse, 'mae': mae, 'r2': r2}

    # 9. Prédictions + analyse -------------------------------------------------------
    def predict_and_analyze(self, test_df: pd.DataFrame, actual_values=True) -> pd.DataFrame:
        dfc = test_df.copy()
        for cat in self.categorical_features:
            dfc[cat] = dfc[cat].astype("category").cat.codes
        dfc["predicted_value"] = self.model.predict(
            dfc[self.feature_names], num_iteration=self.model.best_iteration)

        # proba + cote avec softmax amélioré
        dfc["probability_pct"] = dfc.groupby("ID_PV_COURSE")["predicted_value"]\
            .transform(lambda s: softmax_pct(s.values, self.SOFTMAX_EXP).round(4))\
            .clip(lower=self.MIN_PCT)
            
        # Renormalisation pour que la somme fasse 100%
        dfc["probability_pct"] = dfc.groupby("ID_PV_COURSE")["probability_pct"]\
            .transform(lambda x: x / x.sum() * 100)
            
        dfc["odds"] = (100 / dfc["probability_pct"]).round(2)
        
        # Analyse des performances de prédiction par course
        if actual_values and "PLACE" in dfc.columns:
            winners = dfc[dfc["PLACE"] == 1]["ID_PV_COURSE"].unique()
            winner_probs = dfc[dfc["PLACE"] == 1]["probability_pct"].values
            avg_winner_prob = np.mean(winner_probs) if len(winner_probs) > 0 else 0
            
            top_probs = dfc.loc[dfc.groupby("ID_PV_COURSE")["probability_pct"].idxmax()]
            correct_top = top_probs[top_probs["PLACE"] == 1]
            success_rate = len(correct_top) / len(top_probs) * 100 if len(top_probs) > 0 else 0
            
            print(f"\nAnalyse des prédictions:")
            print(f"- Probabilité moyenne attribuée aux vainqueurs: {avg_winner_prob:.2f}%")
            print(f"- Taux de succès du favori: {success_rate:.2f}%")
            
            # Top N accuracy
            for n in [2, 3, 5]:
                topn_success = 0
                for course_id, course_df in dfc.groupby("ID_PV_COURSE"):
                    top_horses = course_df.nlargest(n, "probability_pct")["ID_CHEVAL"].tolist()
                    winner = course_df[course_df["PLACE"] == 1]["ID_CHEVAL"].values
                    if len(winner) > 0 and winner[0] in top_horses:
                        topn_success += 1
                topn_rate = topn_success / len(dfc["ID_PV_COURSE"].unique()) * 100
                print(f"- Top-{n} accuracy: {topn_rate:.2f}%")
            
        return dfc

    # 10. Export Excel --------------------------------------------------------------
    def export_results(self, test_df: pd.DataFrame):
        df = self.predict_and_analyze(test_df)
        recent = (df.groupby("ID_PV_COURSE")["DATE_COURSE"].first().nlargest(20).index
                  if "DATE_COURSE" in df.columns else df["ID_PV_COURSE"].unique()[-20:])

        with pd.ExcelWriter("regression_results.xlsx", engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Tous_resultats", index=False)
            
            # Feuille d'analyse globale
            summary = pd.DataFrame({
                "ID_PV_COURSE": [],
                "Taille_Course": [],
                "Winner_ID": [],
                "Winner_Prob": [],
                "Predicted_Winner": [],
                "Predicted_Winner_Prob": []
            })
            
            for cid, grp in df[df["ID_PV_COURSE"].isin(recent)].groupby("ID_PV_COURSE"):
                sorted_grp = grp.sort_values("probability_pct", ascending=False)
                sorted_grp.to_excel(writer, sheet_name=str(cid)[:31], index=False)
                
                # Ajouter à l'analyse
                if "PLACE" in sorted_grp.columns:
                    winner = sorted_grp[sorted_grp["PLACE"] == 1]
                    if not winner.empty:
                        win_id = winner["ID_CHEVAL"].iloc[0]
                        win_prob = winner["probability_pct"].iloc[0]
                    else:
                        win_id = "Unknown"
                        win_prob = 0
                        
                    pred_win_id = sorted_grp["ID_CHEVAL"].iloc[0]
                    pred_win_prob = sorted_grp["probability_pct"].iloc[0]
                    
                    summary = pd.concat([summary, pd.DataFrame({
                        "ID_PV_COURSE": [cid],
                        "Taille_Course": [len(grp)],
                        "Winner_ID": [win_id],
                        "Winner_Prob": [win_prob],
                        "Predicted_Winner": [pred_win_id],
                        "Predicted_Winner_Prob": [pred_win_prob]
                    })], ignore_index=True)
            
            if not summary.empty:
                summary["Correct_Prediction"] = summary["Winner_ID"] == summary["Predicted_Winner"]
                summary.to_excel(writer, sheet_name="Analyse_Globale", index=False)
                
        print("✅ regression_results.xlsx généré")

    # 11. Sauvegarde modèle ---------------------------------------------------------
    def save_model(self, path="models/horse_racing_regressor.lgb"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)
        
        # Sauvegarde plus complète des métadonnées
        metadata = {
            "features": self.feature_names,
            "categorical": self.categorical_features,
            "params": self.lgb_params,
            "timestamp": datetime.now().isoformat(),
            "feature_importance": self.feature_importance.to_dict() if self.feature_importance is not None else None,
            "version": "1.2.0",
            "softmax_exp": self.SOFTMAX_EXP,
            "min_pct": self.MIN_PCT
        }
        
        with open(path + ".meta.pkl", "wb") as f:
            pickle.dump(metadata, f)
        print(f"✅ Modèle et métadonnées sauvegardés dans {path}")

    # 12. Pipeline complet ----------------------------------------------------------
    def run(self, dataset_path: str, optimize=True):
        self.load(dataset_path)
        train, test = self.split_data(0.20)
        
        print("\nPrétraitement des données d'entraînement...")
        train_p = self.preprocess_data(train, is_train=True)
        
        print("Prétraitement des données de test...")
        test_p  = self.preprocess_data(test,  is_train=False)
        
        print("\nConstruction des features...")
        self.build_features(train_p)
        
        print("Préparation des données pour l'entraînement...")
        X_train, y_train, _ = self.prepare_for_training(train_p)
        X_test,  y_test,  _ = self.prepare_for_training(test_p)
        
        if optimize:
            print("\nOptimisation des hyperparamètres avec Optuna...")
            self.optuna_search(X_train, y_train)
        
        print("\nEntraînement du modèle final...")
        self.train_final(X_train, y_train, X_test, y_test)
        
        print("\nÉvaluation du modèle...")
        metrics = self.evaluate_model(X_test, y_test)
        
        print("\nGénération des prédictions et analyses...")
        self.export_results(test_p)
        
        print("\nSauvegarde du modèle...")
        self.save_model()
        
        return metrics


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    print("======= HorseRacingRegressor amélioré =======")
    print(f"Date d'exécution: {datetime.now():%Y-%m-%d %H:%M:%S}")
    metrics = HorseRacingRegressor().run("data/dataset.xlsx")
    
    print("\n========== Résultats finaux ==========")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"R²: {metrics['r2']:.4f}")
    print("======================================")


if __name__ == "__main__":
    main()

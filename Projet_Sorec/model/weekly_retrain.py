
import os
import pickle
import shutil
import logging
import argparse
import warnings
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib 
matplotlib.use("Agg")   # pas de GUI — compatible cron/serveur
import matplotlib.pyplot as plt  
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

try:
    import optuna  # type: ignore
    OPTUNA_OK  = True
except ImportError:
    OPTUNA_OK  = False

# ==============================================================================
# CONFIGURATION
# ==============================================================================

SCRIPT_DIR   = Path(os.path.dirname(os.path.abspath(__file__))) 
BASE_DIR     = SCRIPT_DIR.parent 
DATASET_PATH     = BASE_DIR / "data" / "dataset.xlsx"   # fallback Excel
DATASET_CSV_PATH = BASE_DIR / "data" / "dataset.csv"    # priorité CSV (lecture rapide)
MODEL_DIR    = SCRIPT_DIR / "models"
MODEL_PATH   = MODEL_DIR / "horse_racing_regressor.lgb"
META_PATH    = Path(str(MODEL_PATH) + ".meta.pkl")
BACKUP_DIR   = MODEL_DIR / "backups"
LOG_DIR      = SCRIPT_DIR / "logs"
REPORT_DIR   = SCRIPT_DIR / "reports"

# Seuil de dégradation acceptable avant rollback (ex: 0.05 = 5% de dégradation RMSE tolérée)
MAX_RMSE_DEGRADATION = 0.05

# Hyperparamètres par défaut (utilisés si pas d'Optuna ou pas de meta existante)
DEFAULT_PARAMS = {
    "objective":        "regression",
    "metric":           "rmse",
    "learning_rate":    0.025,
    "num_leaves":       95,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq":     5,
    "min_data_in_leaf": 20,
    "verbosity":        -1,
    "random_state":     42,
    "device_type":      "cpu",
    "lambda_l1":        0.2,
    "lambda_l2":        1.0,
    "max_depth":        12,
    "min_gain_to_split":0.01,
    "max_bin":          255,
}

# Colonnes à exclure des features (identifiants, target, fuites de données)
FEATURE_BLACKLIST = {
    "target_value", "course_size", "PLACE", "performance_value",
    "ID_COURSE", "ID_PV_COURSE", "ID_PV_COURSE_1", "ID_SOCIETE_COURSE",
    "ID_CHEVAL", "ID_JOCKEY", "ID_PROPRIETAIRE", "ID_ENTRAINEUR", "ID_NAISSEUR",
    "NUMERO_PMU_SEC", "VH", "NUMERO_PMU_DC", "ID_JOCKEY_DC", "POIDS_REEL",
    "num_starters", "jour_semaine", "mois", "annee", "log_weight", "proba",
    "probability_pct", "odds", "norm_prediction", "horse_jockey", "pair_participations",
    "NOM", "NOM_MERE", "NOM_PERE", "NOM_GRAND_PERE",
    "DATE_COURSE", "DATE_NAISSANCE",
}

# ==============================================================================
# LOGGING
# ==============================================================================

def setup_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("WEEKLY_RETRAIN")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# ==============================================================================
# 1. CHARGEMENT DES DONNÉES
# ==============================================================================

def load_dataset(logger: logging.Logger) -> pd.DataFrame:
    """
    Charge le dataset maître et applique les filtres de base.
    Seuls les partants avec un PLACE valide sont gardés (même logique que lgbmancdtst.py).
    """
    # Priorité au CSV (lecture ~10x plus rapide que Excel)
    if DATASET_CSV_PATH.exists():
        logger.info(f"Chargement du dataset : {DATASET_CSV_PATH}")
        t0 = time.time()
        df = pd.read_csv(DATASET_CSV_PATH, encoding='cp1252', low_memory=False)
        logger.info(f"Brut : {len(df):,} lignes en {time.time()-t0:.1f}s")
    elif DATASET_PATH.exists():
        logger.info(f"CSV introuvable — fallback Excel : {DATASET_PATH}")
        t0 = time.time()
        df = pd.read_excel(DATASET_PATH, engine='openpyxl')
        logger.info(f"Brut : {len(df):,} lignes en {time.time()-t0:.1f}s")
    else:
        raise FileNotFoundError(
            f"Dataset introuvable :\n  CSV : {DATASET_CSV_PATH}\n  Excel : {DATASET_PATH}"
        )

    # Normalisation colonnes
    df.columns = df.columns.str.upper().str.strip()

    # Supprimer colonnes parasites BOM 
    bom_cols = [c for c in df.columns if c != 'ID_COURSE' and c.endswith('ID_COURSE')]
    if bom_cols:
        df = df.drop(columns=bom_cols)

    # Conversion numérique des colonnes clés (nécessaire quand lu depuis CSV)
    numeric_cols = [
        "PLACE", "POIDS_CALCULE", "POIDS_REEL", "CORDE_SEC", "CORDE_DC",
        "NUMERO_PMU_SEC", "NUMERO_PMU_DC", "CODE_DISTANCE_COURSE",
        "ID_CHEVAL", "ID_JOCKEY", "ID_PROPRIETAIRE", "ID_ENTRAINEUR",
        "ID_NAISSEUR", "ID_COURSE", "ID_PV_COURSE", "ID_SOCIETE_COURSE",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Renommage SOREC si nécessaire
    if "ID_PV_COURSE" not in df.columns and "ID_COURSE" in df.columns:
        df.rename(columns={"ID_COURSE": "ID_PV_COURSE"}, inplace=True)

    # Parsing dates
    for col in ("DATE_COURSE", "DATE_NAISSANCE"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

    # Filtrage : garder seulement les partants classés
    df = df[df["PLACE"].notna() & (df["PLACE"] != 0)].copy()
    df = df[df["PLACE"] > 0].copy()
    logger.info(f"Après filtrage PLACE : {len(df):,} lignes")

    # Construction de la target (même formule que lgbmancdtst.py)
    df["target_value"] = np.log1p(1 / df["PLACE"])

    # Tri chronologique
    if "DATE_COURSE" in df.columns:
        df.sort_values("DATE_COURSE", inplace=True)
        df.reset_index(drop=True, inplace=True)
        date_min = df["DATE_COURSE"].min().date()
        date_max = df["DATE_COURSE"].max().date()
        logger.info(f"Période : {date_min} → {date_max}")

    return df


# ==============================================================================
# 2. SPLIT TEMPOREL
# ==============================================================================

def split_temporal(df: pd.DataFrame, test_frac: float = 0.20,
                   logger: logging.Logger = None):
    """
    Split temporel strict : les données de test sont TOUJOURS postérieures au train.
    Évite toute fuite de données temporelle.
    """
    cut_date = df["DATE_COURSE"].quantile(1 - test_frac)
    train = df[df["DATE_COURSE"] <  cut_date].copy()
    test  = df[df["DATE_COURSE"] >= cut_date].copy()

    if logger:
        logger.info(f"Split : train={len(train):,} lignes (avant {cut_date.date()}) "
                    f"| test={len(test):,} lignes")
    return train, test


# ==============================================================================
# 3. FEATURE ENGINEERING
# ==============================================================================

def build_features(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Reproduit exactement le feature engineering de lgbmancdtst.py (preprocess_data).
    Toutes les stats sont calculées avec .shift(1) pour éviter la fuite de données.
    """
    dfc = df.copy()
    id_col = "ID_PV_COURSE" if "ID_PV_COURSE" in dfc.columns else "ID_COURSE"

    logger.info("Feature engineering en cours...")

    # --- Taille de la course ---
    dfc["course_size"] = dfc.groupby(id_col)["ID_CHEVAL"].transform("size")
    dfc["num_starters"] = dfc["course_size"]

    # --- Probabilité rang inversé ---
    dfc["proba"] = (dfc["num_starters"] - dfc["PLACE"] + 1) / dfc["num_starters"]

    # --- Features temporelles ---
    if "DATE_COURSE" in dfc.columns:
        dfc["jour_semaine"] = dfc["DATE_COURSE"].dt.weekday
        dfc["mois"]         = dfc["DATE_COURSE"].dt.month
        dfc["annee"]        = dfc["DATE_COURSE"].dt.year
        dfc["trimestre"]    = (dfc["mois"] - 1) // 3 + 1
        dfc["jour_annee"]   = dfc["DATE_COURSE"].dt.dayofyear
        dfc["jour_mois"]    = dfc["DATE_COURSE"].dt.day

    # --- Features de poids ---
    if "POIDS_REEL" in dfc.columns:
        dfc["POIDS_REEL"]   = pd.to_numeric(dfc["POIDS_REEL"], errors="coerce")
        dfc["log_weight"]   = np.log1p(dfc["POIDS_REEL"].clip(lower=0))
        dfc["delta_weight"] = dfc["POIDS_REEL"] - dfc.groupby(id_col)["POIDS_REEL"].transform("mean")
        dfc["weight_rank"]  = dfc.groupby(id_col)["POIDS_REEL"].rank(pct=True)
        dfc["weight_zscore"]= dfc.groupby(id_col)["POIDS_REEL"].transform(
            lambda x: (x - x.mean()) / x.std() if len(x) > 1 and x.std() > 0 else 0
        )
    elif "POIDS_CALCULE" in dfc.columns:
        dfc["POIDS_REEL"]   = pd.to_numeric(dfc["POIDS_CALCULE"], errors="coerce")
        dfc["log_weight"]   = np.log1p(dfc["POIDS_REEL"].clip(lower=0))
        dfc["delta_weight"] = dfc["POIDS_REEL"] - dfc.groupby(id_col)["POIDS_REEL"].transform("mean")
        dfc["weight_rank"]  = dfc.groupby(id_col)["POIDS_REEL"].rank(pct=True)

    # --- Sexe ---
    sex_col = "CODE_CRITERE_SEXE" if "CODE_CRITERE_SEXE" in dfc.columns else "SEXE"
    if sex_col in dfc.columns:
        dfc["sexe_encoded"] = dfc[sex_col].map({"M": 1, "F": 0}).fillna(0).astype("int8")

    # --- Âge du cheval ---
    if {"DATE_COURSE", "DATE_NAISSANCE"}.issubset(dfc.columns):
        dfc["horse_age_at_race"] = (
            (dfc["DATE_COURSE"] - dfc["DATE_NAISSANCE"]).dt.days / 365.25
        ).fillna(0)
        dfc["age_category"] = pd.cut(
            dfc["horse_age_at_race"],
            bins=[0, 3, 5, 8, 20], labels=[0, 1, 2, 3]
        ).fillna(0).astype(int)

    # --- Jours depuis dernière course ---
    if "DATE_COURSE" in dfc.columns:
        dfc.sort_values(["ID_CHEVAL", "DATE_COURSE"], inplace=True)
        dfc["days_from_last_race"] = (
            dfc.groupby("ID_CHEVAL")["DATE_COURSE"].diff().dt.days.fillna(365)
        )
        dfc["freshness"]    = np.clip(np.exp(-dfc["days_from_last_race"] / 60), 0, 1)
        dfc["long_inactive"]= (dfc["days_from_last_race"] > 180).astype(int)

    # --- Stats historiques par entité ---
    dfc["performance_value"] = dfc["PLACE"]

    for key, entity in {"horse": "ID_CHEVAL",
                        "jockey": "ID_JOCKEY",
                        "trainer": "ID_ENTRAINEUR"}.items():
        if entity not in dfc.columns:
            continue
        dfc.sort_values([entity, "DATE_COURSE"], inplace=True)

        # EWM (Exponentially Weighted Mean) — courses récentes pondérées plus fort
        dfc[f"{key}_ewm_perf"] = (
            dfc.groupby(entity)["performance_value"].shift(1).ewm(span=5).mean()
        )

        # Fenêtres glissantes 3 / 5 / 10 courses
        for N in (3, 5, 10):
            dfc[f"{key}_last{N}_mean"] = (
                dfc.groupby(entity)["performance_value"]
                .shift(1).rolling(N, min_periods=1).mean()
            )
            dfc[f"{key}_last{N}_std"] = (
                dfc.groupby(entity)["performance_value"]
                .shift(1).rolling(N, min_periods=3).std().fillna(0)
            )
            if N >= 5:
                def trend_slope(x):
                    v = x.dropna().values
                    if len(v) >= 3:
                        return np.polyfit(np.arange(len(v)), v, 1)[0]
                    return 0.0
                dfc[f"{key}_last{N}_trend"] = (
                    dfc.groupby(entity)["performance_value"]
                    .shift(1).rolling(N, min_periods=3).apply(trend_slope).fillna(0)
                )

        # Taux victoires / podiums (jockey et entraîneur seulement)
        if key != "horse":
            dfc[f"{key}_win_rate"] = (
                dfc.groupby(entity)["PLACE"]
                .transform(lambda x: x.rolling(10, min_periods=1)
                           .apply(lambda y: (y == 1).mean())).shift(1).fillna(0)
            )
            dfc[f"{key}_podium_rate"] = (
                dfc.groupby(entity)["PLACE"]
                .transform(lambda x: x.rolling(10, min_periods=1)
                           .apply(lambda y: (y <= 3).mean())).shift(1).fillna(0)
            )
            dfc[f"{key}_race_count"] = (
                dfc.groupby(entity).cumcount().shift(1).fillna(0)
            )

    # --- Stats cheval (win_rate, podium_rate, avg_rank) ---
    if "ID_CHEVAL" in dfc.columns:
        dfc.sort_values(["ID_CHEVAL", "DATE_COURSE"], inplace=True)
        dfc["horse_participations"] = dfc.groupby("ID_CHEVAL").cumcount().shift(1).fillna(0)
        dfc["horse_win_rate"]  = (
            dfc.groupby("ID_CHEVAL")["PLACE"]
            .transform(lambda x: (x.shift(1) == 1).expanding().mean()).fillna(0)
        )
        dfc["horse_podium_rate"] = (
            dfc.groupby("ID_CHEVAL")["PLACE"]
            .transform(lambda x: (x.shift(1) <= 3).expanding().mean()).fillna(0)
        )
        dfc["horse_avg_rank"] = (
            dfc.groupby("ID_CHEVAL")["PLACE"]
            .transform(lambda x: x.shift(1).expanding().mean()).fillna(0)
        )

    # --- Stats paire cheval-jockey ---
    if {"ID_CHEVAL", "ID_JOCKEY"}.issubset(dfc.columns):
        dfc["horse_jockey"] = (
            dfc["ID_CHEVAL"].astype(str) + "_" + dfc["ID_JOCKEY"].astype(str)
        )
        dfc.sort_values(["horse_jockey", "DATE_COURSE"], inplace=True)
        dfc["pair_participations"] = dfc.groupby("horse_jockey").cumcount().shift(1).fillna(0)
        dfc["pair_avg_rank"] = (
            dfc.groupby("horse_jockey")["PLACE"]
            .transform(lambda x: x.shift(1).expanding().mean()).fillna(0)
        )
        dfc["pair_win_rate"] = (
            dfc.groupby("horse_jockey")["PLACE"]
            .transform(lambda x: (x.shift(1) == 1).expanding().mean()).fillna(0)
        )
        dfc["pair_podium_rate"] = (
            dfc.groupby("horse_jockey")["PLACE"]
            .transform(lambda x: (x.shift(1) <= 3).expanding().mean()).fillna(0)
        )
        dfc["horse_jockey_familiarity"] = np.minimum(dfc["pair_participations"] / 5, 1.0)

    # --- Distance ---
    if "CODE_DISTANCE_COURSE" in dfc.columns:
        dfc["CODE_DISTANCE_COURSE"] = pd.to_numeric(dfc["CODE_DISTANCE_COURSE"], errors="coerce")
        if "CODE_CATEGORIE_COURSE" in dfc.columns:
            dfc["distance_norm"] = (
                dfc["CODE_DISTANCE_COURSE"] /
                dfc.groupby("CODE_CATEGORIE_COURSE")["CODE_DISTANCE_COURSE"].transform("median")
            )
            dfc["horse_distance"] = (
                dfc["ID_CHEVAL"].astype(str) + "_" +
                pd.cut(dfc["distance_norm"], bins=[0, 0.8, 1.2, 2.0],
                       labels=["court", "moyen", "long"]).fillna("moyen").astype(str)
            )
            dfc["horse_distance_perf"] = (
                dfc.groupby("horse_distance")["PLACE"]
                .transform(lambda x: x.shift(1).rolling(min_periods=1, window=5).mean())
                .fillna(dfc.groupby("ID_CHEVAL")["PLACE"]
                        .transform(lambda x: x.shift(1).rolling(min_periods=1, window=3).mean()))
            )

    # Remplir les NaN : string avec "", numériques avec 0
    string_cols = dfc.select_dtypes(include=['object', 'string']).columns
    for col in string_cols:
        dfc[col] = dfc[col].fillna("")
    dfc = dfc.fillna(0)
    
    logger.info(f"Feature engineering terminé — {len(dfc.columns)} colonnes disponibles")
    return dfc


# ==============================================================================
# 4. SÉLECTION DES FEATURES
# ==============================================================================

def select_features(df: pd.DataFrame, logger: logging.Logger):
    """
    Sélectionne les features numériques et catégorielles.
    Identique à build_features() + prepare_for_training() de lgbmancdtst.py.
    """
    num_cols = df.select_dtypes("number").columns.tolist()
    feature_names = [c for c in num_cols if c not in FEATURE_BLACKLIST]

    cat_features = [c for c in (
        "CODE_DISTANCE_COURSE", "CODE_DISCIPLINE_COURSE",
        "CODE_RACE_CHEVAL", "CODE_CATEGORIE_COURSE",
        "CODE_HIPPODROME", "age_category"
    ) if c in df.columns]

    # Dédoublonnage (num peut contenir des cat)
    feature_names = list(dict.fromkeys(feature_names + cat_features))

    logger.info(f"Features sélectionnées : {len(feature_names)} "
                f"({len(cat_features)} catégorielles)")
    return feature_names, cat_features


def encode_categoricals(df: pd.DataFrame, cat_features: list) -> pd.DataFrame:
    dfc = df.copy()
    for c in cat_features:
        if c in dfc.columns:
            dfc[c] = dfc[c].astype("category").cat.codes
    return dfc


# ==============================================================================
# 5. OPTUNA — OPTIMISATION DES HYPERPARAMÈTRES
# ==============================================================================

def run_optuna(X_train, y_train, feature_names: list, cat_features: list,
               n_trials: int = 50, logger: logging.Logger = None) -> dict:
    """
    Recherche les meilleurs hyperparamètres avec Optuna.
    n_trials=50 pour le retraining hebdomadaire (compromis vitesse/qualité).
    Utilise 200 trials pour l'entraînement initial (comme lgbmancdtst.py).
    """
    if not OPTUNA_OK:
        logger.warning("Optuna non installé — hyperparamètres par défaut utilisés.")
        return DEFAULT_PARAMS.copy()

    logger.info(f"Recherche Optuna ({n_trials} trials)...")

    def objective(trial):
        p = {
            "objective":        "regression",
            "metric":           "rmse",
            "boosting_type":    trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
            "num_leaves":       trial.suggest_int("num_leaves", 31, 255),
            "learning_rate":    trial.suggest_float("learning_rate", 5e-3, 0.15, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq":     trial.suggest_int("bagging_freq", 1, 10),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
            "lambda_l1":        trial.suggest_float("lambda_l1", 0.0, 5.0),
            "lambda_l2":        trial.suggest_float("lambda_l2", 0.0, 10.0),
            "max_depth":        trial.suggest_int("max_depth", 5, 20),
            "min_gain_to_split":trial.suggest_float("min_gain_to_split", 0.001, 0.05),
            "max_bin":          trial.suggest_int("max_bin", 100, 500),
            "verbosity":        -1,
            "random_state":     42,
            "device_type":      "cpu",
        }
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        for tr_idx, val_idx in tscv.split(X_train):
            X_tr, y_tr   = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
            dtr  = lgb.Dataset(X_tr, y_tr, feature_name=feature_names,
                               categorical_feature=cat_features)
            dval = lgb.Dataset(X_val, y_val, reference=dtr,
                               feature_name=feature_names,
                               categorical_feature=cat_features)
            callbacks = [lgb.early_stopping(50, verbose=False)]
            try:
                from optuna.integration import LightGBMPruningCallback  # type: ignore
                callbacks.insert(0, LightGBMPruningCallback(trial, "rmse"))
            except ImportError:
                pass  # optuna.integration non disponible
            m = lgb.train(p, dtr, 800, valid_sets=[dval], valid_names=["val"],
                          callbacks=callbacks)
            preds_val   = m.predict(X_val, num_iteration=m.best_iteration)
            preds_train = m.predict(X_tr,  num_iteration=m.best_iteration)
            val_rmse   = np.sqrt(mean_squared_error(y_val,  preds_val))
            train_rmse = np.sqrt(mean_squared_error(y_tr,   preds_train))
            # Pénaliser l'overfitting : score = val_rmse + 0.5 × gap train/val
            gap = max(0.0, val_rmse - train_rmse)
            scores.append(val_rmse + 0.5 * gap)
        return np.mean(scores)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = DEFAULT_PARAMS.copy()
    best.update(study.best_trial.params)
    if logger:
        logger.info(f"Meilleurs paramètres Optuna : RMSE={study.best_value:.4f}")
        logger.info(str(study.best_trial.params))
    return best


# ==============================================================================
# 6. ENTRAÎNEMENT FINAL
# ==============================================================================

def train_model(X_train, y_train, X_val, y_val,
                params: dict, feature_names: list, cat_features: list,
                logger: logging.Logger) -> lgb.Booster:
    logger.info("Entraînement du modèle LightGBM...")
    dtr = lgb.Dataset(X_train, y_train, feature_name=feature_names,
                      categorical_feature=cat_features)
    dval = lgb.Dataset(X_val, y_val, reference=dtr, feature_name=feature_names,
                       categorical_feature=cat_features)

    model = lgb.train(
        params, dtr,
        num_boost_round=1500,
        valid_sets=[dtr, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(50, verbose=True),
            lgb.log_evaluation(200)
        ]
    )
    logger.info(f"Entraînement terminé — {model.best_iteration} arbres")
    return model


# ==============================================================================
# 7. ÉVALUATION
# ==============================================================================

def evaluate(model: lgb.Booster, X_test, y_test,
             logger: logging.Logger) -> dict:
    preds = model.predict(X_test, num_iteration=model.best_iteration)
    rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae   = float(mean_absolute_error(y_test, preds))
    r2    = float(r2_score(y_test, preds))

    # Brier Score : calibration des probabilités (0 = parfait, 0.25 = aléatoire)
    preds_norm = (preds - preds.min()) / (preds.max() - preds.min() + 1e-9)
    y_binary   = (y_test > y_test.median()).astype(int).values
    brier      = float(np.mean((preds_norm - y_binary) ** 2))

    logger.info(f"Métriques → RMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2:.4f} | Brier={brier:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2, "brier_score": brier}


def evaluate_racing_accuracy(model: lgb.Booster, test_df: pd.DataFrame,
                              feature_names: list, cat_features: list,
                              logger: logging.Logger) -> dict:
    """
    Métriques spécifiques aux courses hippiques (plus pertinentes que RMSE seul) :
    - Probabilité moyenne assignée au vainqueur réel
    - Taux de succès du favori prédit (Top-1)
    - Top-3 et Top-5 accuracy
    """
    dfc = encode_categoricals(test_df, cat_features)
    missing = [f for f in feature_names if f not in dfc.columns]
    for f in missing:
        dfc[f] = 0

    dfc["predicted_value"] = model.predict(
        dfc[feature_names], num_iteration=model.best_iteration
    )

    # Softmax par course
    def softmax_pct(scores, exp=8.0):
        shifted = scores - scores.max()
        e = np.exp(shifted * exp)
        return (e / e.sum()) * 100

    dfc["win_prob"] = (
        dfc.groupby("ID_PV_COURSE")["predicted_value"]
        .transform(lambda s: softmax_pct(s.values))
    )

    results = {"top1": 0, "top3": 0, "top5": 0,
               "avg_winner_prob": 0.0, "n_courses": 0}
    winner_probs = []

    for course_id, grp in dfc.groupby("ID_PV_COURSE"):
        winners = grp[grp["PLACE"] == 1]
        if winners.empty:
            continue
        results["n_courses"] += 1
        winner_id   = winners["ID_CHEVAL"].iloc[0]
        winner_prob = winners["win_prob"].iloc[0]
        winner_probs.append(winner_prob)

        sorted_grp = grp.sort_values("win_prob", ascending=False)
        top1 = sorted_grp["ID_CHEVAL"].iloc[0]
        top3 = sorted_grp["ID_CHEVAL"].iloc[:3].tolist()
        top5 = sorted_grp["ID_CHEVAL"].iloc[:5].tolist()

        if top1 == winner_id:
            results["top1"] += 1
        if winner_id in top3:
            results["top3"] += 1
        if winner_id in top5:
            results["top5"] += 1

    n = results["n_courses"]
    if n > 0:
        results["top1_pct"]          = round(results["top1"] / n * 100, 2)
        results["top3_pct"]          = round(results["top3"] / n * 100, 2)
        results["top5_pct"]          = round(results["top5"] / n * 100, 2)
        results["avg_winner_prob"]   = round(float(np.mean(winner_probs)), 2)

    logger.info(f"  Courses évaluées       : {n}")
    logger.info(f"  Top-1 accuracy (favori): {results.get('top1_pct', 0):.1f}%")
    logger.info(f"  Top-3 accuracy         : {results.get('top3_pct', 0):.1f}%")
    logger.info(f"  Top-5 accuracy         : {results.get('top5_pct', 0):.1f}%")
    logger.info(f"  Prob. moy. vainqueur   : {results.get('avg_winner_prob', 0):.2f}%")
    return results


# ==============================================================================
# 8. SAUVEGARDE ET COMPARAISON
# ==============================================================================

def load_old_metrics() -> dict | None:
    """Charge les métriques du modèle actuellement en production."""
    if not META_PATH.exists():
        return None
    try:
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
        return meta.get("metrics")
    except Exception:
        return None


def backup_current_model(logger: logging.Logger):
    """Sauvegarde le modèle actuel avant écrasement."""
    if not MODEL_PATH.exists():
        return
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for ext in ["", ".meta.pkl"]:
        src = Path(str(MODEL_PATH) + ext)
        if src.exists():
            dst = BACKUP_DIR / f"horse_racing_regressor_{ts}.lgb{ext}"
            shutil.copy2(src, dst)
    # Garder seulement les 5 derniers backups
    for old in sorted(BACKUP_DIR.glob("*.lgb"))[:-5]:
        old.unlink()
        meta_old = Path(str(old) + ".meta.pkl")
        if meta_old.exists():
            meta_old.unlink()
    logger.info(f"Backup modèle créé : horse_racing_regressor_{ts}.lgb")


def save_model(model: lgb.Booster, feature_names: list, cat_features: list,
               params: dict, metrics: dict, racing_metrics: dict,
               logger: logging.Logger):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    metadata = {
        "features":         feature_names,
        "categorical":      cat_features,
        "params":           params,
        "metrics":          metrics,
        "racing_metrics":   racing_metrics,
        "timestamp":        datetime.now().isoformat(),
        "version":          "weekly",
        "softmax_exp":      8.0,
        "min_pct":          0.4,
        "dataset_path":     str(DATASET_PATH),
    }
    with open(str(META_PATH), "wb") as f:
        pickle.dump(metadata, f)
    logger.info(f"Modèle sauvegardé : {MODEL_PATH}")


# ==============================================================================
# 9. RAPPORT HEBDOMADAIRE
# ==============================================================================

def save_report(metrics: dict, racing_metrics: dict, old_metrics: dict | None,
                feature_names: list, model: lgb.Booster,
                logger: logging.Logger):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts          = datetime.now().strftime("%Y%m%d")
    report_path = REPORT_DIR / f"retrain_report_{ts}.xlsx"

    # Importance des features
    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Gain":    model.feature_importance(importance_type="gain"),
        "Split":   model.feature_importance(importance_type="split"),
    }).sort_values("Gain", ascending=False)

    # Comparaison avec l'ancien modèle
    comparison = {"Métrique": ["RMSE", "MAE", "R²", "Brier Score", "Top-1%", "Top-3%", "Top-5%", "Prob.Vainqueur%"]}
    comparison["Nouveau modèle"] = [
        metrics.get("rmse", "-"), metrics.get("mae", "-"), metrics.get("r2", "-"),
        metrics.get("brier_score", "-"),
        racing_metrics.get("top1_pct", "-"), racing_metrics.get("top3_pct", "-"),
        racing_metrics.get("top5_pct", "-"), racing_metrics.get("avg_winner_prob", "-"),
    ]
    if old_metrics:
        comparison["Ancien modèle"] = [
            old_metrics.get("rmse", "-"), old_metrics.get("mae", "-"),
            old_metrics.get("r2", "-"), old_metrics.get("brier_score", "-"),
            old_metrics.get("top1_pct", "-"), old_metrics.get("top3_pct", "-"),
            old_metrics.get("top5_pct", "-"), old_metrics.get("avg_winner_prob", "-"),
        ]

    comp_df = pd.DataFrame(comparison)

    # Graphique feature importance
    fig_path = REPORT_DIR / f"feature_importance_{ts}.png"
    plt.figure(figsize=(12, 8))
    plt.barh(imp_df["Feature"][:20], imp_df["Gain"][:20], color="steelblue")
    plt.gca().invert_yaxis()
    plt.title(f"Top 20 Features - {datetime.now().strftime('%d/%m/%Y')}")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    # Sauvegarde Excel
    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        comp_df.to_excel(writer, sheet_name="Comparaison_Métriques", index=False)
        imp_df.to_excel(writer, sheet_name="Feature_Importance", index=False)

    logger.info(f"Rapport sauvegardé : {report_path}")
    logger.info(f"Graphique sauvegardé : {fig_path}")


# ==============================================================================
# 10. PIPELINE PRINCIPAL
# ==============================================================================

def run_weekly_retrain(force: bool = False, use_optuna: bool = True,
                       dry_run: bool = False, report_only: bool = False,
                       n_optuna_trials: int = 50):
    warnings.filterwarnings("ignore")
    logger = setup_logger()

    logger.info("=" * 60)
    logger.info(f" RETRAINING HEBDOMADAIRE - {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    if dry_run:
        logger.info("  ⚠  MODE DRY-RUN : modèle non sauvegardé")
    logger.info("=" * 60)

    # --- Mode rapport uniquement ---
    if report_only:
        old = load_old_metrics()
        if old:
            logger.info("Métriques du modèle actuel :")
            for k, v in old.items():
                logger.info(f"  {k}: {v}")
        else:
            logger.info("Aucune métrique sauvegardée pour le modèle actuel.")
        return

    # --- Vérification dataset ---
    if not DATASET_PATH.exists():
        logger.error(f"Dataset introuvable : {DATASET_PATH}")
        logger.error("Lancez d'abord maintenance_data.py pour mettre à jour le dataset.")
        return

    t_start = time.time()

    # --- 1. Chargement ---
    df = load_dataset(logger)

    # --- 2. Split temporel ---
    train_raw, test_raw = split_temporal(df, test_frac=0.20, logger=logger)

    # --- 3. Feature engineering ---
    logger.info("Prétraitement train...")
    train_fe = build_features(train_raw, logger)
    logger.info("Prétraitement test...")
    test_fe  = build_features(test_raw, logger)

    # --- 4. Sélection features ---
    feature_names, cat_features = select_features(train_fe, logger)

    # --- 5. Préparation matrices ---
    train_enc = encode_categoricals(train_fe, cat_features)
    test_enc  = encode_categoricals(test_fe,  cat_features)

    # Assurer que toutes les features existent
    for f in feature_names:
        if f not in train_enc.columns: train_enc[f] = 0
        if f not in test_enc.columns:  test_enc[f]  = 0

    X_train = train_enc[feature_names]
    y_train = train_enc["target_value"]
    X_test  = test_enc[feature_names]
    y_test  = test_enc["target_value"]

    # --- 6. Hyperparamètres ---
    # On réutilise les meilleurs params de la semaine précédente comme point de départ
    old_meta = None
    if META_PATH.exists():
        try:
            with open(str(META_PATH), "rb") as f:
                old_meta = pickle.load(f)
            params = old_meta.get("params", DEFAULT_PARAMS.copy())
            logger.info("Hyperparamètres de la semaine précédente chargés comme point de départ.")
        except Exception:
            params = DEFAULT_PARAMS.copy()
    else:
        params = DEFAULT_PARAMS.copy()

    if use_optuna:
        params = run_optuna(X_train, y_train, feature_names, cat_features,
                            n_trials=n_optuna_trials, logger=logger)
    else:
        logger.info("Optuna désactivé — hyperparamètres réutilisés.")

    # --- 7. Entraînement ---
    model = train_model(X_train, y_train, X_test, y_test,
                        params, feature_names, cat_features, logger)

    # --- 8. Évaluation ---
    logger.info("\n--- Métriques de régression ---")
    metrics = evaluate(model, X_test, y_test, logger)

    logger.info("\n--- Métriques hippiques ---")
    racing_metrics = evaluate_racing_accuracy(
        model, test_enc, feature_names, cat_features, logger
    )
    # Fusionner les métriques pour la sauvegarde
    metrics.update(racing_metrics)

    # --- 9. Comparaison avec l'ancien modèle (rollback si dégradation) ---
    old_metrics = load_old_metrics()
    should_save = True

    if old_metrics and "rmse" in old_metrics:
        old_rmse = old_metrics["rmse"]
        new_rmse = metrics["rmse"]
        degradation = (new_rmse - old_rmse) / old_rmse

        if degradation > MAX_RMSE_DEGRADATION:
            logger.warning(f"DÉGRADATION DÉTECTÉE : RMSE {old_rmse:.4f} → {new_rmse:.4f} "
                           f"({degradation*100:.1f}% > seuil {MAX_RMSE_DEGRADATION*100:.0f}%)")
            logger.warning("Le nouveau modèle N'EST PAS sauvegardé. Ancien modèle conservé.")
            should_save = False
        else:
            delta = new_rmse - old_rmse
            sign  = "+" if delta > 0 else ""
            logger.info(f"RMSE : {old_rmse:.4f} → {new_rmse:.4f} ({sign}{delta:.4f})")
    else:
        logger.info("Aucun modèle existant — première sauvegarde.")

    # --- 10. Sauvegarde ---
    if dry_run:
        logger.info("[DRY-RUN] Simulation terminée. Aucun fichier modifié.")
    elif should_save:
        backup_current_model(logger)
        save_model(model, feature_names, cat_features, params,
                   metrics, racing_metrics, logger)

    # --- 11. Rapport ---
    save_report(metrics, racing_metrics, old_metrics,
                feature_names, model, logger)

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info(f" RETRAINING TERMINÉ en {elapsed/60:.1f} minutes")
    logger.info(f"  RMSE     : {metrics['rmse']:.4f}")
    logger.info(f"  Top-1%%  : {metrics.get('top1_pct', 'N/A')}%%")
    logger.info(f"  Top-3%%  : {metrics.get('top3_pct', 'N/A')}%%")
    logger.info("=" * 60)


# ==============================================================================
# POINT D'ENTRÉE
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retraining hebdomadaire du modèle LightGBM - SOREC"
    )
    parser.add_argument("--force", action="store_true",
                        help="Forcer l'exécution (ignore le filtre dimanche)")
    parser.add_argument("--no-optuna", action="store_true",
                        help="Désactiver Optuna (rapide, ~5min)")
    parser.add_argument("--optuna-trials", type=int, default=50,
                        help="Nombre de trials Optuna (défaut: 50)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Évaluer sans sauvegarder le modèle")
    parser.add_argument("--report-only", action="store_true",
                        help="Afficher les métriques du modèle actuel")
    args = parser.parse_args()

    run_weekly_retrain(
        force        = args.force,
        use_optuna   = not args.no_optuna,
        dry_run      = args.dry_run,
        report_only  = args.report_only,
        n_optuna_trials = args.optuna_trials,
    )
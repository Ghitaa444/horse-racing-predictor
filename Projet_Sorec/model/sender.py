#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import warnings
import pickle
import smtplib
import pandas as pd
import numpy as np
import lightgbm as lgb
import glob
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# ============================================================================
# NOUVELLES FONCTIONS DE DÉTECTION AUTOMATIQUE
# ============================================================================

def get_files_by_date_pattern(base_dir=".", target_date=None):
    """
    Détecte automatiquement les fichiers basés sur les patterns de dates
    - test files: courses de demain (target_date + 1 jour)
    - hist files: résultats d'aujourd'hui (target_date)
    """
    if target_date is None:
        target_date = datetime.now()
    

    # Date de demain pour les fichiers de test (courses à prédire)
    tomorrow = target_date + timedelta(days=1)
    
    found_files = {'test': [], 'hist': []}
    
    # === FICHIERS DE TEST (COURSES DE DEMAIN) ===
    tomorrow_formats = [
        tomorrow.strftime("%d%m%Y"),     # 16052025 (si aujourd'hui = 15/05/2025)
        tomorrow.strftime("%d_%m_%Y"),   # 16_05_2025
        tomorrow.strftime("%Y%m%d"),     # 20250516
        tomorrow.strftime("%Y_%m_%d"),   # 2025_05_16
        tomorrow.strftime("%d-%m-%Y"),   # 16-05-2025
        tomorrow.strftime("%Y-%m-%d"),   # 2025-05-16
    ]
    
   
    for date_fmt in tomorrow_formats:
        test_patterns = [
            f"cours{date_fmt}.*",
            f"course{date_fmt}.*", 
            f"courses_{date_fmt}.*",
            f"*{date_fmt}*cours*.*",
            f"*{date_fmt}*course*.*"
        ]
        
        for pattern in test_patterns:
            matches = glob.glob(os.path.join(base_dir+"new-races/", pattern), recursive=True)
            found_files['test'].extend(matches)
            print(os.path.join(base_dir+"new_races/", pattern))
 
    # === FICHIERS HISTORIQUES (RÉSULTATS D'AUJOURD'HUI) ===
    yesterday = target_date - timedelta(days=1)
    yesterday_formats = [
        yesterday.strftime("%d%m%Y"),       # 14052025 (si aujourd'hui = 15/05/2025, veille = 14/05/2025)
        yesterday.strftime("%d_%m_%Y"),     # 14_05_2025
        yesterday.strftime("%Y%m%d"),       # 20250514
        yesterday.strftime("%Y_%m_%d"),     # 2025_05_14
        yesterday.strftime("%d-%m-%Y"),     # 14-05-2025
        yesterday.strftime("%Y-%m-%d"),     # 2025-05-14
    ]
    
    for t_fmt in yesterday_formats:
        hist_patterns = [
            f"resulta*{t_fmt}.*",
            f"result*{t_fmt}.*",
            f"*{t_fmt}*result*.*",
            f"*{t_fmt}*resulta*.*"
        ]
        
        for pattern in hist_patterns:
            matches = glob.glob(os.path.join(base_dir+"old-races-results/", pattern), recursive=True)
            found_files['hist'].extend(matches)
    
    # Supprimer les doublons et trier par date de modification (plus récent en premier)
    found_files['test'] = sorted(list(set(found_files['test'])), 
                                key=lambda x: os.path.getmtime(x), reverse=True)
    found_files['hist'] = sorted(list(set(found_files['hist'])), 
                                key=lambda x: os.path.getmtime(x), reverse=True)
    
    return found_files

def find_latest_files_by_pattern(base_dir=".", patterns=None):
    """
    Trouve les fichiers les plus récents correspondant aux patterns donnés
    """
    if patterns is None:
        patterns = {
            'test': ['cours*.csv', 'cours*.CSV', 'course*.csv', 'course*.CSV'],
            'hist': ['result*.xlsx', 'result*.xls', 'resulta*.xlsx'],
            'hist_base': ['resulta_combined*.xlsx']
        }
    
    found_files = {}
    
    for file_type, pattern_list in patterns.items():
        candidates = []
        for pattern in pattern_list:
            matches = glob.glob(os.path.join(base_dir, pattern))
            candidates.extend(matches)
        
        if candidates:
            # Trier par date de modification (plus récent en premier)
            latest_file = max(candidates, key=lambda x: os.path.getmtime(x))
            found_files[file_type] = latest_file
            print(f"Fichier {file_type} détecté: {latest_file}")
        else:
            found_files[file_type] = None
            print(f"Aucun fichier {file_type} trouvé")
    
    return found_files

def auto_detect_files(base_dir=".", target_date=None):
    """
    Fonction principale pour détecter les fichiers
    """
    if target_date is None:
        target_date = datetime.now()
    
    tomorrow = target_date + timedelta(days=1)
    
    print(f"\nRecherche des fichiers...")
    print(f"Date cible (courses): {tomorrow.strftime('%d/%m/%Y')} (demain)")
    yesterday = target_date - timedelta(days=1)
    print(f"Date historique (résultats): {yesterday.strftime('%d/%m/%Y')} (veille)")
    
    # Essayer d'abord la détection par date
    found_files = get_files_by_date_pattern(base_dir, target_date)
    
    # Afficher les résultats de la détection
    if found_files['test']:
        print(f"Fichiers courses trouvés: {len(found_files['test'])}")
        for f in found_files['test'][:3]:  # Afficher les 3 premiers
            print(f"  - {os.path.basename(f)}")
    else:
        print("Aucun fichier de courses trouvé")
        
    if found_files['hist']:
        print(f"Fichiers résultats trouvés: {len(found_files['hist'])}")
        for f in found_files['hist'][:3]:  # Afficher les 3 premiers
            print(f"  - {os.path.basename(f)}")
    else:
               print("Aucun fichier de résultats de la veille trouvé")
    
    # Si aucun fichier trouvé par date, utiliser la recherche par pattern
    if not found_files['test'] or not found_files['hist']:
        print("\nRecherche par pattern de fichiers récents...")
        latest_files = find_latest_files_by_pattern(base_dir)
        
        if not found_files['test'] and latest_files.get('test'):
            found_files['test'] = [latest_files['test']]
            print(f"Fichier course trouvé: {os.path.basename(latest_files['test'])}")
        if not found_files['hist'] and latest_files.get('hist'):
            found_files['hist'] = [latest_files['hist']]
            print(f"Fichier historique trouvé: {os.path.basename(latest_files['hist'])}")
        if not found_files.get('hist_base') and latest_files.get('hist_base'):
            found_files['hist_base'] = latest_files['hist_base']
            print(f"Fichier base trouvé: {os.path.basename(latest_files['hist_base'])}")
    
    return found_files

def setup_command_line():
    """
    Configuration des arguments de ligne de commande
    """
    parser = argparse.ArgumentParser(description='Horse Racing Prediction with File Detection')
    
    parser.add_argument('--detect', action='store_true', 
                       help='Détection des fichiers')
    parser.add_argument('--hist', type=str, 
                       help='Forcer le fichier historique de base')
    parser.add_argument('--hist2', type=str, 
                       help='Forcer le fichier des résultats de la veille')
    parser.add_argument('--test', type=str, 
                       help='Forcer le fichier des courses du jour')
    parser.add_argument('--date', type=str, 
                       help='Date cible (format: YYYY-MM-DD)')
    parser.add_argument('--no-email', action='store_true',
                       help='Ne pas envoyer d\'email')
    parser.add_argument('--list-files', action='store_true',
                       help='Lister les fichiers détectés seulement')
    
    return parser.parse_args()

# ---------- utilitaire soft‑max amélioré ----------
def softmax_pct(scores: np.ndarray, exp: float = 1.5) -> np.ndarray:
    """Convertit des scores en pourcentages via softmax avec contrôle de température"""
    # Détecter automatiquement si les scores doivent être inversés (pour temps de course)
    invert = np.median(scores) < 0 or (np.min(scores) < 0 and np.max(scores) > 0)
    if invert:
        scores = -scores
        
    shifted = scores - scores.max()
    e = np.exp(shifted * exp)
    return (e / e.sum()) * 100

class HorseRacingSender:
    def __init__(self, model_file: str):
        self.model_file = model_file
        self.model = None
        self.meta = None
        self.data = None
        self.df_test = None
        self.feature_names = []
        self.categorical_features = []
        # Valeurs par défaut qui seront mises à jour depuis les métadonnées
        self.SOFTMAX_EXP = 8 # Modifié de 10.0 à 2.2 comme demandé
        self.MIN_PCT = 0.4

    def load_model_and_meta(self):
        # Chargement du modèle
        try:
            self.model = lgb.Booster(model_file=self.model_file)
        except Exception as e:
            print("⚠ Impossible de charger directement le modèle LightGBM :", e)
            txt = open(self.model_file, "r").read()
            self.model = lgb.Booster(model_str=txt)

        # Chargement des métadonnées
        meta_path = f"{self.model_file}.meta.pkl"
        with open(meta_path, "rb") as f:
            self.meta = pickle.load(f)

        self.feature_names = self.meta["features"]
        self.categorical_features = self.meta["categorical"]
        
        # Récupération des paramètres de softmax depuis les métadonnées
        # Nous ignorons la valeur des métadonnées pour conserver notre valeur de 2.2
        # if "softmax_exp" in self.meta:
        #     self.SOFTMAX_EXP = self.meta["softmax_exp"]
        if "min_pct" in self.meta:
            self.MIN_PCT = self.meta["min_pct"]
            
        print(f"Modèle et méta chargés — features attendues : {len(self.feature_names)}")
        print(f"Paramètres: SOFTMAX_EXP={self.SOFTMAX_EXP}, MIN_PCT={self.MIN_PCT}")

    def load_data(self, hist_path: str, test_path: str):
        # Charger les données historiques pour les features dépendantes du temps
        df_hist = (
            pd.read_excel(hist_path)
            if hist_path.lower().endswith(".xlsx")
            else pd.read_csv(hist_path, delimiter=";", low_memory=False)
        )
        
        if "ID_COURSE" in df_hist.columns and "ID_PV_COURSE" not in df_hist.columns:
            df_hist.rename(columns={"ID_COURSE": "ID_PV_COURSE"}, inplace=True)

        # Charger les données de test (nouvelles courses)
        df_test = (
            pd.read_excel(test_path)
            if test_path.lower().endswith(".xlsx")
            else pd.read_csv(test_path, delimiter=";", low_memory=False)
        )
        
        # DEBUG: Afficher les colonnes disponibles dans df_test
        print("Colonnes disponibles dans le fichier de test:")
        for i, col in enumerate(df_test.columns):
            print(f"  {i}: {col}")
        
        # Standardisation des colonnes: majuscules/minuscules
        # Pour gérer les fichiers CSV avec des noms de colonnes en minuscules
        df_test.columns = [col.upper() if not col.isupper() else col for col in df_test.columns]
        
      # Standardisation des colonnes
        df_test.columns = [col.upper() if not col.isupper() else col for col in df_test.columns]
        
        rename_map = {
            "ID_COURSE": "ID_PV_COURSE",
            "POIDS_CALCULE": "POIDS_REEL"
        }
        
        if "ID_COURSE" not in df_test.columns and "ID_PV_COURSE" not in df_test.columns:
            potential_id_columns = [col for col in df_test.columns if "ID" in col and "COURSE" in col]
            if potential_id_columns:
                rename_map[potential_id_columns[0]] = "ID_PV_COURSE"
                print(f"Colonne alternative trouvée pour ID_PV_COURSE: {potential_id_columns[0]}")
            else:
                print("Aucune colonne ID_COURSE ou similaire trouvée, création d'une colonne artificielle")
                if "NUMERO_COURSE" in df_test.columns:
                    df_test["ID_PV_COURSE"] = df_test["NUMERO_COURSE"].astype(str)
                elif "NUM_COURSE" in df_test.columns:
                    df_test["ID_PV_COURSE"] = df_test["NUM_COURSE"].astype(str)
                else:
                    df_test["ID_PV_COURSE"] = df_test.index.astype(str)
                print("Colonne ID_PV_COURSE créée artificiellement")
                
        df_test.rename(columns={k: v for k, v in rename_map.items() 
                            if k in df_test.columns}, inplace=True)
        
        # Vérifier si ID_PV_COURSE existe maintenant
        if "ID_PV_COURSE" not in df_test.columns:
            raise KeyError(f"La colonne ID_PV_COURSE est requise mais introuvable. Colonnes disponibles: {', '.join(df_test.columns)}")
            
        # Conversion des dates
        for col in ("DATE_COURSE", "DATE_NAISSANCE"):
            if col in df_test.columns:
                try:
                    df_test[col] = pd.to_datetime(df_test[col], dayfirst=True, errors="coerce")
                    print(f"Colonne {col} convertie en datetime")
                except Exception as e:
                    print(f"Impossible de convertir la colonne {col} en datetime: {e}")
            if col in df_hist.columns:
                df_hist[col] = pd.to_datetime(df_hist[col], dayfirst=True, errors="coerce")
                
        # Ne garder dans l'historique que les partants classés
        df_hist = df_hist[df_hist["PLACE"].notna() & (df_hist["PLACE"] != 0)]
        
        self.df_test = df_test.copy()
        self.data = pd.concat([df_hist, df_test], ignore_index=True)
        print(f"Données chargées : {len(df_hist):,} historiques + {len(df_test):,} test → {len(self.data):,}")

    def prepare_features(self):
        df = self.data.copy()
        try:
            test_ids = self.df_test["ID_PV_COURSE"].unique()
            print(f"Identifiants de courses identifiés: {len(test_ids)} courses uniques")
        except KeyError as e:
            print(f"Erreur: {e}")
            print(f"Colonnes disponibles dans df_test: {', '.join(self.df_test.columns)}")
            raise

        # ---- Feature engineering adapté du HorseRacingRegressor ----
        
        # Taille de la course
        df["course_size"] = df.groupby("ID_PV_COURSE")["ID_CHEVAL"].transform("size")
        
        # Ajout des features manquantes identifiées
        # POIDS_CALCULE peut être nécessaire si différent de POIDS_REEL
        if "POIDS_REEL" in df.columns and "POIDS_CALCULE" not in df.columns:
            df["POIDS_CALCULE"] = df["POIDS_REEL"]
            
        # CORDE_DC (position à la corde)
        if "CORDE_DC" not in df.columns and "CORDE" in df.columns:
            df["CORDE_DC"] = df["CORDE"]
        else:
            df["CORDE_DC"] = 0

        # Features temporelles
        if "DATE_COURSE" in df.columns:
            df["jour_semaine"] = df["DATE_COURSE"].dt.weekday
            df["mois"] = df["DATE_COURSE"].dt.month
            df["annee"] = df["DATE_COURSE"].dt.year
            df["trimestre"] = (df["mois"] - 1) // 3 + 1
            df["jour_annee"] = df["DATE_COURSE"].dt.dayofyear
            df["jour_mois"] = df["DATE_COURSE"].dt.day

        # Features de poids
        if "POIDS_REEL" in df.columns:
            df["POIDS_REEL"] = pd.to_numeric(df["POIDS_REEL"], errors="coerce")
            df["log_weight"] = np.log1p(df["POIDS_REEL"].clip(lower=0))
            df["delta_weight"] = df["POIDS_REEL"] - df.groupby("ID_PV_COURSE")["POIDS_REEL"].transform("mean")
            df["weight_rank"] = df.groupby("ID_PV_COURSE")["POIDS_REEL"].rank(pct=True)
            df["weight_zscore"] = df.groupby("ID_PV_COURSE")["POIDS_REEL"].transform(
                lambda x: (x - x.mean()) / x.std() if len(x) > 1 and x.std() > 0 else 0
            )

        # Sexe
        if "SEXE" in df.columns:
            df["sexe_encoded"] = df["SEXE"].map({"M": 1, "F": 0}).fillna(0).astype("int8")

        # Âge du cheval
        if {"DATE_COURSE", "DATE_NAISSANCE"}.issubset(df.columns):
            df["horse_age_at_race"] = ((df["DATE_COURSE"] - df["DATE_NAISSANCE"]).dt.days / 365.25).fillna(0)
            df["age_category"] = pd.cut(
                df["horse_age_at_race"], 
                bins=[0, 3, 5, 8, 20], 
                labels=[0, 1, 2, 3]
            ).fillna(0).astype(int)

        # Jours depuis dernière course et fraîcheur
        if "DATE_COURSE" in df.columns:
            df.sort_values(["ID_CHEVAL", "DATE_COURSE"], inplace=True)
            df["days_from_last_race"] = df.groupby("ID_CHEVAL")["DATE_COURSE"].diff().dt.days.fillna(365)
            df["freshness"] = np.clip(np.exp(-df["days_from_last_race"] / 60), 0, 1)
            df["long_inactive"] = (df["days_from_last_race"] > 180).astype(int)

        # Performance historique
        # Pour TEMPS_COURSE ou PLACE selon celui qui est présent
        if "TEMPS_COURSE" in df.columns:
            df["performance_value"] = pd.to_numeric(df["TEMPS_COURSE"], errors="coerce")
        else:
            df["performance_value"] = df["PLACE"].fillna(0)
            
        # --- Features de participation et de performance moyenne ---
        # Stats du cheval
        if "ID_CHEVAL" in df.columns:
            df.sort_values(["ID_CHEVAL", "DATE_COURSE"], inplace=True)
            # Nombre de courses du cheval
            df["horse_participations"] = df.groupby("ID_CHEVAL").cumcount().shift(1).fillna(0)
            # Classement moyen
            df["horse_avg_rank"] = df.groupby("ID_CHEVAL")["PLACE"].transform(
                lambda x: x.shift(1).expanding().mean()
            ).fillna(0)
            
        # --- Statistiques sur la paire cheval-jockey ---
        if {"ID_CHEVAL", "ID_JOCKEY"}.issubset(df.columns):
            # Clé d'association cheval-jockey
            pair_key = "horse_jockey"
            if pair_key not in df.columns:
                df[pair_key] = df["ID_CHEVAL"].astype(str) + "_" + df["ID_JOCKEY"].astype(str)
                
            df.sort_values([pair_key, "DATE_COURSE"], inplace=True)
            
            # Nombre de courses ensemble
            df["pair_participations"] = df.groupby(pair_key).cumcount().shift(1).fillna(0)
            
            # Classement moyen ensemble
            df["pair_avg_rank"] = df.groupby(pair_key)["PLACE"].transform(
                lambda x: x.shift(1).expanding().mean()
            ).fillna(0)
            
            # Taux de victoire ensemble
            df["pair_win_rate"] = df.groupby(pair_key)["PLACE"].transform(
                lambda x: (x.shift(1) == 1).expanding().mean()
            ).fillna(0)
            
            # Taux de podium ensemble
            df["pair_podium_rate"] = df.groupby(pair_key)["PLACE"].transform(
                lambda x: (x.shift(1) <= 3).expanding().mean()
            ).fillna(0)
            
        # --- Stats par entité (cheval, jockey, entraîneur) ---
        for key, entity in {
            "horse": "ID_CHEVAL", 
            "jockey": "ID_JOCKEY",
            "trainer": "ID_ENTRAINEUR"
        }.items():
            if entity in df.columns:
                df.sort_values([entity, "DATE_COURSE"], inplace=True)
                
                # Moyenne pondérée exponentielle
                df[f"{key}_ewm_perf"] = df.groupby(entity)["performance_value"].shift(1).ewm(span=5).mean()
                
                # Stats sur différentes fenêtres temporelles
                for N in (3, 5, 10):
                    df[f"{key}_last{N}_mean"] = (
                        df.groupby(entity)["performance_value"].shift(1).rolling(N, min_periods=1).mean()
                    )
                    # Écart-type pour capturer la stabilité des performances
                    df[f"{key}_last{N}_std"] = (
                        df.groupby(entity)["performance_value"].shift(1).rolling(N, min_periods=3).std().fillna(0)
                    )
                    # Tendance des performances récentes
                    if N >= 5:
                        window = df.groupby(entity)["performance_value"].shift(1).rolling(N, min_periods=3)
                        def trend_slope(x):
                            if len(x) >= 3:
                                valid_x = x.dropna()
                                if len(valid_x) >= 3:
                                    y = valid_x.values
                                    x = np.arange(len(y))
                                    return np.polyfit(x, y, 1)[0] if len(x) > 1 else 0
                            return 0
                        df[f"{key}_last{N}_trend"] = window.apply(trend_slope).fillna(0)
                
                # Ratio de victoires et podiums
                wins = df.groupby(entity)["PLACE"].transform(
                    lambda x: x.rolling(10, min_periods=1).apply(lambda y: (y == 1).mean())
                ).shift(1).fillna(0)
                df[f"{key}_win_rate"] = wins
                
                podiums = df.groupby(entity)["PLACE"].transform(
                    lambda x: x.rolling(10, min_periods=1).apply(lambda y: (y <= 3).mean())
                ).shift(1).fillna(0)
                df[f"{key}_podium_rate"] = podiums
                
                # Nombre de courses total (experience)
                df[f"{key}_race_count"] = df.groupby(entity).cumcount().shift(1).fillna(0)

        # --- Interactions entre entités ---
        if {"ID_CHEVAL", "ID_JOCKEY"}.issubset(df.columns):
            # Combinaison cheval-jockey (familiarité)
            df["horse_jockey"] = df["ID_CHEVAL"].astype(str) + "_" + df["ID_JOCKEY"].astype(str)
            df["horse_jockey_count"] = df.groupby("horse_jockey").cumcount().shift(1).fillna(0)
            df["horse_jockey_familiarity"] = np.minimum(df["horse_jockey_count"] / 5, 1.0)
        
        # --- Stats par course ---
        # Compétitivité de la course
        if "COTE_DIRECT" in df.columns:
            df["COTE_DIRECT"] = pd.to_numeric(df["COTE_DIRECT"], errors="coerce")
            df["odds_rank"] = df.groupby("ID_PV_COURSE")["COTE_DIRECT"].rank(pct=True)
            df["odds_vs_median"] = df["COTE_DIRECT"] / df.groupby("ID_PV_COURSE")["COTE_DIRECT"].transform("median")
            
        # --- Features contextuelles ---
        if "CODE_HIPPODROME" in df.columns:
            # Stats par hippodrome pour chaque entité
            for entity, col in [("horse", "ID_CHEVAL"), ("jockey", "ID_JOCKEY")]:
                if col in df.columns:
                    # Clé combinée entité + hippodrome
                    df[f"{entity}_track"] = df[col].astype(str) + "_" + df["CODE_HIPPODROME"].astype(str)
                    # Performance moyenne sur cet hippodrome
                    track_perf = df.groupby(f"{entity}_track")["PLACE"].transform(
                        lambda x: x.shift(1).rolling(min_periods=1, window=10).mean()
                    ).fillna(df.groupby(col)["PLACE"].transform(
                        lambda x: x.shift(1).rolling(min_periods=1, window=5).mean()
                    ))
                    df[f"{entity}_track_perf"] = track_perf
                    
        # --- Features de distance ---
        if "DISTANCE_COURSE" in df.columns:
            df["DISTANCE_COURSE"] = pd.to_numeric(df["DISTANCE_COURSE"], errors="coerce")
            # Distance normalisée (pour comparer entre disciplines)
            df["distance_norm"] = df["DISTANCE_COURSE"] / df.groupby("CODE_DISCIPLINE_COURSE")["DISTANCE_COURSE"].transform("median")
            
            # Performance par distance pour chaque cheval
            df["horse_distance"] = df["ID_CHEVAL"].astype(str) + "_" + (
                pd.cut(df["distance_norm"], bins=[0, 0.8, 1.2, 2.0], labels=["court", "moyen", "long"])
                .fillna("moyen").astype(str)
            )
            df["horse_distance_perf"] = df.groupby("horse_distance")["PLACE"].transform(
                lambda x: x.shift(1).rolling(min_periods=1, window=5).mean()
            ).fillna(df.groupby("ID_CHEVAL")["PLACE"].transform(
                lambda x: x.shift(1).rolling(min_periods=1, window=3).mean()
            ))

        df.fillna(0, inplace=True)
        self.data = df
        print(f"Feature engineering terminé - {len(self.data.columns)} colonnes disponibles")

    def predict_and_export(self, output_file_fr: str, output_file_en: str = None, output_details: str = None):
        df = self.data.copy()

        # Encodage catégoriel
        for c in self.categorical_features:
            if c in df.columns:
                df[c] = df[c].astype("category").cat.codes

        # Features manquantes
        missing_features = []
        for f in self.feature_names:
            if f not in df.columns:
                missing_features.append(f)
                df[f] = 0
        
        if missing_features:
            print(f"⚠ {len(missing_features)} features manquantes, ajoutées avec valeur 0:")
            for i, f in enumerate(missing_features[:5]):  # Afficher les 5 premières features manquantes
                print(f"  {i+1}. {f}")
            if len(missing_features) > 5:
                print(f"  ... et {len(missing_features)-5} autres")

        # Prédiction
        X = df[self.feature_names]
        df["predicted_value"] = self.model.predict(X, num_iteration=self.model.best_iteration)
        print(f"Prédictions générées pour {len(df)} lignes")

        # Application directe de softmax pour les probabilités
        # Assurons-nous que le calcul des probabilités est bien effectué
        print(f"Calcul des probabilités avec SOFTMAX_EXP={self.SOFTMAX_EXP}")
        
        def apply_softmax(group):
            values = group["predicted_value"].values
            probas = softmax_pct(values, exp=self.SOFTMAX_EXP)
            return pd.Series(probas, index=group.index)
            
        # Calculer les probabilités par course
        df["win_probability_pct"] = df.groupby("ID_PV_COURSE").apply(apply_softmax).reset_index(level=0, drop=True)
        
        # Vérification des résultats de probabilité
        for course_id in df["ID_PV_COURSE"].unique()[:3]:  # Afficher quelques exemples
            course_df = df[df["ID_PV_COURSE"] == course_id]
            print(f"\nDistribution des probabilités pour course {course_id}:")
            for i, (idx, row) in enumerate(course_df.sort_values("win_probability_pct", ascending=False).head(5).iterrows()):
                print(f"  {i+1}. Prob: {row['win_probability_pct']:.2f}%")
        
        # Renormalisation pour que la somme fasse 100%
        df["win_probability_pct"] = df.groupby("ID_PV_COURSE")["win_probability_pct"].transform(
            lambda x: x / x.sum() * 100
        ).clip(lower=self.MIN_PCT)
        
        # Calcul original des cotes à partir des probabilités
        df["cote_original"] = (100 / df["win_probability_pct"]).round(2)

        print("\nAjustement des cotes selon la nouvelle règle...")
        # Pour chaque course, ajuster les cotes
        adjusted_courses = 0
        for course_id in df["ID_PV_COURSE"].unique():
            # Filtrer les données pour cette course
            course_mask = df["ID_PV_COURSE"] == course_id
            course_df = df[course_mask]
            
            # Trouver le favori (probabilité la plus élevée)
            favorite_idx = course_df["win_probability_pct"].idxmax()
            favorite_original_odds = df.loc[favorite_idx, "cote_original"]
            
            # Calculer le delta à soustraire
            delta = favorite_original_odds - 1.1
            
            # Définir la cote du favori à 1.1
            df.loc[favorite_idx, "cote"] = 1.1
            
            # Soustraire le delta des cotes pour tous les autres chevaux de la course
            non_favorite_mask = course_mask & (df.index != favorite_idx)
            df.loc[non_favorite_mask, "cote"] = (df.loc[non_favorite_mask, "cote_original"] - delta).clip(lower=1.2)
            
            # Afficher des informations pour les premières courses (pour vérification)
            if adjusted_courses < 3:
                print(f"\nCourse {course_id} - Ajustement de cotes:")
                print(f"  Favori: cote originale {favorite_original_odds:.2f} → nouvelle cote 1.10")
                print(f"  Delta appliqué aux autres partants: {delta:.2f}")
                # Afficher quelques exemples d'ajustement pour les autres chevaux
                for i, (idx, row) in enumerate(df.loc[non_favorite_mask].sort_values("win_probability_pct", ascending=False).head(3).iterrows()):
                    print(f"  Cheval {i+2}: cote originale {row['cote_original']:.2f} → nouvelle cote {row['cote']:.2f}")
            
            adjusted_courses += 1

        print(f"Cotes ajustées pour {adjusted_courses} courses")

        # Vérification des cotes ajustées
        for course_id in df["ID_PV_COURSE"].unique()[:3]:  # Afficher quelques exemples
            course_df = df[df["ID_PV_COURSE"] == course_id]
            print(f"\nDistribution des cotes ajustées pour course {course_id}:")
            for i, (idx, row) in enumerate(course_df.sort_values("win_probability_pct", ascending=False).head(5).iterrows()):
                print(f"  {i+1}. Prob: {row['win_probability_pct']:.2f}% -> Cote ajustée: {row['cote']:.2f}")

        # Filtre test
        test_ids = self.df_test["ID_PV_COURSE"].unique()
        df_res = df[df["ID_PV_COURSE"].isin(test_ids)].copy()

        # Ajouts
        if "NUM_COURSE" in df_res.columns:
            df_res["num_course"] = df_res["NUM_COURSE"]
        elif "NUMERO_COURSE" in df_res.columns:
            df_res["num_course"] = df_res["NUMERO_COURSE"]
        else:
            df_res["num_course"] = df_res.groupby("ID_PV_COURSE").ngroup() + 1
            
        df_res["reunion"] = "R9" if "reunion" not in df_res.columns else df_res["reunion"]
        
        # Extraction de la date des courses
        if "DATE_COURSE" in df_res.columns:
            try:
                # Vérifier si la colonne est bien en format datetime
                if pd.api.types.is_datetime64_any_dtype(df_res["DATE_COURSE"]):
                    df_res["date_course_formattee"] = df_res["DATE_COURSE"].dt.strftime("%d/%m/%Y")
                    print("Format de date correctement formaté")
                else:
                    # Si la colonne existe mais n'est pas en datetime, essayer de la convertir
                    df_res["date_course_formattee"] = pd.to_datetime(df_res["DATE_COURSE"], 
                                                                  dayfirst=True, 
                                                                  errors="coerce").dt.strftime("%d/%m/%Y")
                    print("Colonne DATE_COURSE convertie manuellement")
            except Exception as e:
                print(f"Erreur lors du formatage de date: {e}")
                print("Utilisation de la date actuelle")
                df_res["date_course_formattee"] = datetime.now().strftime("%d/%m/%Y")
        else:
            df_res["date_course_formattee"] = datetime.now().strftime("%d/%m/%Y")

        # Colonnes à exporter
        export_cols = [
            "reunion", "num_course", "date_course_formattee",
            "ID_PV_COURSE", "NUMERO_PMU_SEC",
            "cote"
        ]

        # Ajout du nom du cheval pour identifier les prédictions
        if "NOM_CHEVAL" in df_res.columns:
            export_cols.insert(5, "NOM_CHEVAL")
        elif "LIBELLE_CHEVAL" in df_res.columns:
            df_res["NOM_CHEVAL"] = df_res["LIBELLE_CHEVAL"]
            export_cols.insert(5, "NOM_CHEVAL")

        # Vérification de l'existence des colonnes obligatoires
        missing_cols = [col for col in export_cols if col not in df_res.columns]
        if missing_cols:
            print(f"Colonnes manquantes dans le résultat: {missing_cols}")
            for col in missing_cols:
                if col == "NUMERO_PMU_SEC":
                    # Chercher des alternatives pour NUMERO_PMU_SEC
                    potential_num_cols = [c for c in df_res.columns if "NUMERO" in c and "PMU" in c]
                    if potential_num_cols:
                        df_res["NUMERO_PMU_SEC"] = df_res[potential_num_cols[0]]
                        print(f"Utilisation de {potential_num_cols[0]} pour NUMERO_PMU_SEC")
                    else:
                        df_res["NUMERO_PMU_SEC"] = df_res.groupby("ID_PV_COURSE").cumcount() + 1
                        print("Création d'un NUMERO_PMU_SEC séquentiel")

        # Corriger les colonnes manquantes finales
        export_cols = [col for col in export_cols if col in df_res.columns or col in ("cote")]

        # Version française
        df_fr = (
            df_res[export_cols]
            .sort_values(["num_course", "cote"], ascending=[True, True])  # Trier par numéro de course et cote croissante
            .rename(columns={
                "reunion":              "Réunion",
                "num_course":           "N° Course",
                "date_course_formattee": "Date",
                "ID_PV_COURSE":         "ID Course",
                "NUMERO_PMU_SEC":       "Numéro PMU",
                "NOM_CHEVAL":           "Cheval",
                "cote":                 "Cote",
            })
        )
        df_fr.to_excel(output_file_fr, index=False)
        print(f"Version française exportée: {output_file_fr}")

        # Version anglaise
        if output_file_en:
            df_en = (
                df_res[export_cols]
                .sort_values(["num_course", "cote"], ascending=[True, True])  # Trier par numéro de course et cote croissante
                .rename(columns={
                    "reunion":              "Meeting",
                    "num_course":           "Race No",
                    "date_course_formattee": "Date",
                    "ID_PV_COURSE":         "Race ID",
                    "NUMERO_PMU_SEC":       "PMU Number",
                    "NOM_CHEVAL":           "Horse",
                    "cote":                 "Odds",
                })
            )
            df_en.to_excel(output_file_en, index=False)
            print(f"English version exported: {output_file_en}")

        # Fichier détaillé avec toutes les features et statistiques
        if output_details:
            # Sélectionner les colonnes clés et features principales pour analyse détaillée
            cols = ['ID_PV_COURSE', 'NUMERO_PMU_SEC', 'date_course_formattee', 'num_course']  # Ajout de la date et num_course
            if 'NOM_CHEVAL' in df_res.columns:
                cols.append('NOM_CHEVAL')
            
            # Ajout d'informations détaillées sur le cheval et caractéristiques de la course
            detail_cols = cols + [
                'predicted_value', 'win_probability_pct', 'cote',
                'ID_CHEVAL', 'ID_JOCKEY', 'ID_ENTRAINEUR',
                'SEXE', 'POIDS_REEL', 'horse_age_at_race'
            ]
            
            # Ajouter des informations de performances
            perf_cols = [c for c in df_res.columns if any(
                x in c for x in ['_perf', 'win_rate', 'podium_rate', '_familiarity', '_mean', '_std', '_trend']
            )]
            
            # Fusion des listes de colonnes
            detail_cols.extend([c for c in perf_cols if c in df_res.columns])
            
            # Exporter les détails
            df_detail = df_res[
                [c for c in detail_cols if c in df_res.columns]
            ].sort_values(['num_course', 'cote'], ascending=[True, True])  # Trier par numéro de course et cote croissante
            
            df_detail.to_excel(output_details, index=False)
            print(f"Détails des prédictions exportés: {output_details}")

        # Analyse des prédictions
        print("\nAnalyse des prédictions:")
        for course_id, course_df in df_res.groupby("ID_PV_COURSE"):
            top3 = course_df.nlargest(3, "win_probability_pct")
            
            # Afficher avec nom du cheval si disponible
            if 'NOM_CHEVAL' in top3.columns:
                print(f"Course {course_id} - Top 3:")
                for i in range(min(3, len(top3))):
                    print(f"  {i+1}. {top3['NOM_CHEVAL'].iloc[i]} - "
                          f"{top3['win_probability_pct'].iloc[i]:.2f}% - "
                          f"Cote: {top3['cote'].iloc[i]:.2f}")
            else:
                print(f"Course {course_id} - Top 3 probabilités: "
                      f"{top3['win_probability_pct'].values[0]:.2f}%, "
                      f"{top3['win_probability_pct'].values[1]:.2f}%, "
                      f"{top3['win_probability_pct'].values[2]:.2f}%")

        return [f for f in (output_file_fr, output_file_en, output_details) if f]

    def send_mail(self, smtp_user, smtp_pass, to_addr, subject, body, attachment_paths):
        if isinstance(to_addr, list):
            to_list = to_addr
            to_str  = ", ".join(to_addr)
        else:
            to_list = [to_addr]
            to_str  = to_addr

        msg = MIMEMultipart()
        msg["From"]    = smtp_user
        msg["To"]      = to_str
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        if not isinstance(attachment_paths, list):
            attachment_paths = [attachment_paths]

        for path in attachment_paths:
            part = MIMEBase("application", "octet-stream")
            with open(path, "rb") as f:
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(path)}")
            msg.attach(part)

        server = smtplib.SMTP("smg.sorec.ma", 25)
        #server.login(smtp_user, smtp_pass)
        server.send_message(msg)
        server.quit()
        print(f"Email envoyé à {len(to_list)} destinataire(s): {to_str} avec {len(attachment_paths)} pièce(s) jointe(s)")

def run_prediction(
    hist_path: str,
    test_path: str,
    model_file: str,
    output_file_fr: str,
    output_file_en: str = None,
    output_details: str = None,
    smtp_user: str = None,
    smtp_pass: str = None,
    receiver: str = None,
    subject: str = None,
    body: str = None
):
    warnings.filterwarnings("ignore")
    
    print("\n====== HorseRacingSender adapté ======")
    print(f"Date d'exécution: {datetime.now():%Y-%m-%d %H:%M:%S}")
    
    try:
        sender = HorseRacingSender(model_file)
        sender.load_model_and_meta()
        sender.load_data(hist_path, test_path)
        sender.prepare_features()
        outputs = sender.predict_and_export(output_file_fr, output_file_en, output_details)
        
        if smtp_user and smtp_pass and receiver and subject and body:
            sender.send_mail(smtp_user, smtp_pass, receiver, subject, body, outputs)
            
        print("\n=======================================")
        return outputs
    except Exception as e:
        print(f"\n❌ Une erreur s'est produite: {str(e)}")
        import traceback
        print(traceback.format_exc())
        print("\n=======================================")
        return None

def combine_excel_files(file1, file2, output_file="data/dataset.xlsx"):
    """
    Combines two Excel files into one.
    """
    try:
        # Check if input files exist
        if not os.path.exists(file1):
            print(f"Error: File {file1} not found.")
            return
        if not os.path.exists(file2):
            print(f"Error: File {file2} not found.")
            return
            
        # Read Excel files
        print(f"Reading {file1}...")
        df1 = pd.read_excel(file1)
        print(f"Reading {file2}...")
        df2 = pd.read_excel(file2)
        
        # Convertir les noms de colonnes en majuscules pour les deux DataFrames
        df1.columns = df1.columns.str.upper()
        df2.columns = df2.columns.str.upper()
        
        # Print info about each file
        print(f"\nFile 1 ({file1}) has {len(df1)} rows and {len(df1.columns)} columns")
        print(f"File 2 ({file2}) has {len(df2)} rows and {len(df2.columns)} columns")
        
        # Combine DataFrames (rows from both files)
        print("\nCombining files...")
        combined_df = pd.concat([df1, df2], ignore_index=True)
        
        # Save combined DataFrame to a new Excel file
        print(f"Saving combined data to {output_file}...")
        combined_df.to_excel(output_file, index=False)
        
        print(f"Combination complete! Combined file has {len(combined_df)} rows.")
        print(f"Output saved to: {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# ============================================================================
# SECTION PRINCIPALE MODIFIÉE
# ============================================================================

if __name__ == "__main__":
    # Configuration des arguments de ligne de commande
    try:
        args = setup_command_line()
    except SystemExit:
        # Si pas d'arguments, utiliser les valeurs par défaut
        args = type('Args', (), {
            'detect': True,
            'hist': None,
            'hist2': None, 
            'test': None,
            'date': None,
            'no_email': False,
            'list_files': False
        })()
    
    # Gestion de la date
    target_date = None
    if hasattr(args, 'date') and args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            print("Format de date invalide. Utilisation de la date actuelle.")
            target_date = datetime.now()
    else:
        target_date = datetime.now()
    
    # Détection ou utilisation des fichiers spécifiés
    if args.detect or not (args.hist and args.test):
        print("\nMode détection activé")
        detected_files = auto_detect_files(base_dir="data/", target_date=target_date)
        
        # Utiliser les fichiers détectés ou ceux spécifiés
        HIST = args.hist if args.hist else "data/dataset.xlsx"
        
        # HIST2 = fichier des résultats d'aujourd'hui (optionnel)
        HIST2 = args.hist2 if args.hist2 else (detected_files['hist'][0] if detected_files['hist'] else "")
        
        # Éviter que HIST2 soit le même fichier que HIST
        if HIST2 and os.path.abspath(HIST2) == os.path.abspath(HIST):
            print(f"HIST2 est identique à HIST, ignorer la combinaison")
            HIST2 = ""
        
        # TEST = fichier des courses de demain (obligatoire)
        TEST = args.test if args.test else (detected_files['test'][0] if detected_files['test'] else None)
        
        if not TEST:
            print("\nAucun fichier de test détecté. Recherche manuelle...")
            # Essayer de trouver le fichier le plus récent qui contient "cours"
            all_csv_files = glob.glob("cours*.csv") + glob.glob("cours*.CSV") + glob.glob("course*.csv")
            if all_csv_files:
                TEST = max(all_csv_files, key=lambda x: os.path.getmtime(x))
                print(f"Fichier trouvé: {TEST}")
            else:
                print("ERREUR: Aucun fichier de courses trouvé. Veuillez placer le fichier dans le répertoire.")
                print("Format attendu: cours[DATE_DEMAIN].csv (ex: cours16052025.csv)")
                exit(1)
            
    else:
        print("\nUtilisation des fichiers spécifiés")
        HIST = args.hist
        HIST2 = args.hist2 if args.hist2 else ""
        TEST = args.test
    
    # Vérification du fichier de base obligatoire
    if not os.path.exists(HIST):
        print(f"ERREUR: Fichier de base historique non trouvé: {HIST}")
        exit(1)
    
    # Vérification du fichier de test obligatoire
    if not TEST or not os.path.exists(TEST):
        print(f"ERREUR: Fichier de courses non trouvé: {TEST}")
        exit(1)
    
    # Affichage des fichiers utilisés
    print(f"\nFichiers utilisés:")
    print(f"  Base historique: {HIST}")
    if HIST2 and os.path.exists(HIST2):
        print(f"  Résultats aujourd'hui: {HIST2} (sera combiné avec la base)")
    elif HIST2:
        print(f"  Résultats aujourd'hui: {HIST2} (FICHIER NON TROUVÉ - ignoré)")
        HIST2 = ""  # Réinitialiser si le fichier n'existe pas
    else:
        print(f"  Résultats aujourd'hui: (aucun fichier - utilisation base seule)")
    print(f"  Courses de demain: {TEST}")
    
    # Si seulement lister les fichiers
    if hasattr(args, 'list_files') and args.list_files:
        print("\nListe des fichiers terminée")
        exit(0)
    
    # Génération des noms de fichiers de sortie basés sur la date de demain
    tomorrow = target_date + timedelta(days=1)
    date_str = tomorrow.strftime("%d_%m_%Y")
    MODEL_FILE = "models/horse_racing_regressor.lgb"
    OUTPUT_FR = f"predictions_{date_str}_fr.xlsx"
    OUTPUT_EN = f"predictions_{date_str}_en.xlsx"
    OUTPUT_DETAILS = f"predictions_{date_str}_details.xlsx"
    OUTPUT_COMBINED = "../data/dataset.xlsx"
    
    # Combinaison des fichiers historiques (si HIST2 disponible)
    if HIST2 and os.path.exists(HIST2):
        print(f"\nCombinaison des fichiers historiques...")
        try:
            combine_excel_files(HIST, HIST2, OUTPUT_COMBINED)
            HIST = OUTPUT_COMBINED
            print("Combinaison réussie")
        except Exception as e:
            print(f"AVERTISSEMENT: Echec de la combinaison des fichiers: {e}")
            print("Continuation avec le fichier de base seul")
    else:
        print(f"\nAucun fichier de résultats récents - utilisation de la base historique seule")
    
    # Configuration email
    if not (hasattr(args, 'no_email') and args.no_email):
        SMTP_USER = "ai@sorec.ma"
        SMTP_PASS = "Sorec@i2025"
        RECEIVER = ["supgam@sorec.ma", "exploitation@sorec.ma", "programmes@sorec.ma", "crc@sorec.ma",
                    "HBENABDESSELAM@sorec.ma", "hasrar@sorec.ma", "jfoudsi@sorec.ma"]
                    
        SUBJECT = f"Côtes Fixes Automatiques R9 {tomorrow.strftime('%d/%m/%Y')}"
        BODY = (
            f"Bonjour,\n\n"
            f"Vous trouverez en pièce jointe les prédictions pour les courses du {tomorrow.strftime('%d/%m/%Y')}.\n\n"
            f"Hello,\n\n"
            f"Please find attached the predictions for races of {tomorrow.strftime('%d/%m/%Y')}.\n\n"
            f"Cordialement / Best regards."
        )
    else:
        SMTP_USER = SMTP_PASS = RECEIVER = SUBJECT = BODY = None
        print("Envoi d'email désactivé")
    
    # Exécution de la prédiction
    try:
        outputs = run_prediction(
            HIST, TEST, MODEL_FILE,
            OUTPUT_FR, OUTPUT_EN, OUTPUT_DETAILS,
            SMTP_USER, SMTP_PASS,
            RECEIVER, SUBJECT, BODY
        )
        
        if outputs:
            print(f"\nPrédictions générées avec succès:")
            for output in outputs:
                print(f"   {output}")
        else:
            print("\nEchec de la génération des prédictions")
            
    except Exception as e:
        print(f"\nErreur lors de l'exécution: {str(e)}")
        import traceback
        traceback.print_exc()

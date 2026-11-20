import os
import glob
import shutil
import logging
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_DIR       = Path(__file__).resolve().parent.parent
MASTER_PATH    = BASE_DIR / "data" / "dataset.xlsx"
MASTER_CSV_PATH = BASE_DIR / "data" / "dataset.csv"
INPUT_DIR      = BASE_DIR / "data" / "old-races-results"
ARCHIVE_DIR    = BASE_DIR / "data" / "results-treated"
BACKUP_DIR     = BASE_DIR / "data" / "backups"
LOG_DIR        = BASE_DIR / "logs"

# Colonnes de référence attendues dans le Master Dataset
MASTER_COLUMNS = [
    'ID_COURSE', 'ID_PV_COURSE', 'HIP', 'ID_SOCIETE_COURSE', 'DATE_COURSE',
    'CODE_CATEGORISATION_COURSE', 'CODE_CATEGORIE_COURSE', 'CODE_AGE_CHEVAL',
    'CODE_CRITERE_SEXE', 'CODE_ORIGINE_CHEVAL', 'CODE_RACE_CHEVAL',
    'CODE_CLASSIFICATION_COURSE', 'CODE_DISTANCE_COURSE', 'ID_CHEVAL',
    'ID_JOCKEY', 'ID_PROPRIETAIRE', 'ID_ENTRAINEUR', 'ID_NAISSEUR',
    'NUMERO_PMU_SEC', 'CORDE_SEC', 'VH', 'POIDS_CALCULE',
    'CODE_TYPE_OEILLERE', 'CORDE_DC', 'NUMERO_PMU_DC', 'PLACE',
    'PV_COURSE', 'ID_JOCKEY_DC', 'POIDS_REEL', 'NOM',
    'NOM_MERE', 'NOM_PERE', 'NOM_GRAND_PERE'
]

# Clés uniques pour le dédoublonnage
DEDUP_KEYS = ['ID_COURSE', 'ID_CHEVAL']

# Encodages à essayer pour les CSV (ordre de priorité)
CSV_ENCODINGS = ['cp1252', 'utf-8', 'latin-1', 'utf-8-sig']

# Colonnes souvent absentes selon le type d'export SOREC.
# Elles sont utiles quand elles existent, mais leur absence ne bloque pas la sync.
OPTIONAL_MISSING_COLUMNS = {
    'ID_PV_COURSE', 'CODE_CATEGORISATION_COURSE', 'CODE_ORIGINE_CHEVAL',
    'VH', 'CODE_TYPE_OEILLERE', 'CORDE_DC', 'NUMERO_PMU_DC', 'PLACE',
    'PV_COURSE', 'ID_JOCKEY_DC', 'POIDS_REEL', 'NOM_GRAND_PERE'
}

# ==============================================================================
# LOGGING
# ==============================================================================

def setup_logger() -> logging.Logger:
    """Configure un logger avec sortie console + fichier horodaté."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"sync_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("SYNC_MASTER")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")

    # Handler console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    # Handler fichier
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


# ==============================================================================
# FONCTIONS UTILITAIRES
# ==============================================================================

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise les noms de colonnes et fusionne les doublons crees par BOM."""
    df = df.copy()
    cleaned_columns = []
    for col in df.columns:
        name = str(col)
        # Supprimer toutes les variantes connues du BOM UTF-8
        name = name.replace('\ufeff', '')           # BOM unicode natif
        name = name.replace('\u00cf\u00bb\u00bf', '') # BOM mal décodé en latin-1
        name = name.replace('Ï»¿', '')              # BOM cp1252 majuscules
        name = name.replace('ï»¿', '')              # BOM cp1252 minuscules (CSV SOREC: 'ï»¿id_course')
        name = name.upper().strip()
        if name.endswith('ID_COURSE') and name != 'ID_COURSE':
            name = 'ID_COURSE'
        cleaned_columns.append(name)
    df.columns = cleaned_columns

    if df.columns.duplicated().any():
        merged_cols = {}
        for col in dict.fromkeys(df.columns):
            positions = [i for i, current in enumerate(df.columns) if current == col]
            if len(positions) == 1:
                merged_cols[col] = df.iloc[:, positions[0]]
            else:
                merged_cols[col] = df.iloc[:, positions].bfill(axis=1).iloc[:, 0]
        df = pd.DataFrame(merged_cols, index=df.index)

    # Supprimer les colonnes parasites BOM résiduelles (ex: 'Ï»¿ID_COURSE')
    bom_parasites = [c for c in df.columns if c != 'ID_COURSE' and c.endswith('ID_COURSE')]
    if bom_parasites:
        df = df.drop(columns=bom_parasites)

    return df


def read_file_safe(filepath: str, logger: logging.Logger) -> pd.DataFrame | None:
    """
    Lit un fichier Excel ou CSV avec gestion robuste des encodages.
    Retourne None si la lecture échoue.
    """
    ext = os.path.splitext(filepath)[1].lower()

    # --- Excel ---
    if ext == '.xlsx':
        try:
            df = pd.read_excel(filepath, engine='openpyxl')
            return df
        except Exception as e:
            logger.error(f"Impossible de lire {os.path.basename(filepath)} (Excel) : {e}")
            return None

    # --- CSV : essayer plusieurs encodages ---
    if ext == '.csv':
        for enc in CSV_ENCODINGS:
            try:
                # Force la lecture initiale en texte pour éviter les altérations automatiques de Pandas
                df = pd.read_csv(filepath, encoding=enc, sep=None, engine='python', dtype=str)
                logger.debug(f"CSV lu avec encodage '{enc}' : {os.path.basename(filepath)}")
                return df
            except (UnicodeDecodeError, Exception):
                continue
        logger.error(f"Aucun encodage ne fonctionne pour {os.path.basename(filepath)}")
        return None

    logger.warning(f"Extension non supportée : {ext} ({os.path.basename(filepath)})")
    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise les noms de colonnes et applique les renommages SOREC connus.
    Sécurise également le type de ID_COURSE.
    """
    # 1. id_course devient ID_COURSE ici automatiquement grâce au passage en majuscules
    df = normalize_column_names(df)

    # 2. Renommages spécifiques aux exports SOREC
    rename_map = {
        'ID_PV_COURSE_1': 'ID_PV_COURSE', 
        'DATE':           'DATE_COURSE',    
        'NUM_COURSE':     'ID_COURSE',      
        'LIBELLE_CHEVAL': 'NOM',            
        'NOM_CHEVAL':     'NOM',            
        'SEXE':           'CODE_CRITERE_SEXE',  
    }
    
    applicable = {k: v for k, v in rename_map.items()
                  if k in df.columns and v not in df.columns}
    df = df.rename(columns=applicable)

    extra_columns = [col for col in df.columns if col not in MASTER_COLUMNS]
    if extra_columns:
        df = df.drop(columns=extra_columns)

    # 3. Nettoyage et forçage en String de ID_COURSE et ID_CHEVAL pour éviter les bugs de doublons
    if 'ID_COURSE' in df.columns:
        df['ID_COURSE'] = (df['ID_COURSE'].astype(object).astype(str).str.strip()
                           .str.replace(r'\.0$', '', regex=True)
                           .replace({'': None, 'nan': None, 'None': None, 'NaT': None}))# type: ignore

    if 'ID_CHEVAL' in df.columns:
        df['ID_CHEVAL'] = (df['ID_CHEVAL'].astype(object).astype(str).str.strip()
                           .str.replace(r'\.0$', '', regex=True)
                           .replace({'': None, 'nan': None, 'None': None, 'NaT': None}))# type: ignore

    return df


def validate_columns(df: pd.DataFrame, filename: str, logger: logging.Logger) -> None:
    """
    Vérifie les colonnes manquantes/supplémentaires vs le Master.
    Log uniquement, ne bloque pas.
    """
    df_cols = set(df.columns)
    master_cols = set(MASTER_COLUMNS)

    missing  = master_cols - df_cols
    extra    = df_cols - master_cols

    critical_missing = missing - OPTIONAL_MISSING_COLUMNS
    optional_missing = missing & OPTIONAL_MISSING_COLUMNS

    if critical_missing:
        logger.warning(f"[{filename}] Colonnes critiques absentes ({len(critical_missing)}) : {sorted(critical_missing)}")
    if optional_missing:
        logger.debug(f"[{filename}] Colonnes optionnelles absentes ({len(optional_missing)}) : {sorted(optional_missing)}")
    if extra:
        logger.debug(f"[{filename}] Colonnes supplémentaires ({len(extra)}) : {sorted(extra)}")


def normalize_date(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Normalise DATE_COURSE en datetime64[ns] (compatible avec sender.py).
    Conserve le type datetime pour que les calculs .dt.days, .dt.weekday fonctionnent.
    """
    if 'DATE_COURSE' not in df.columns:
        logger.warning("Colonne DATE_COURSE absente du fichier — ignorée.")
        return df

    before = df['DATE_COURSE'].isna().sum()
    if pd.api.types.is_datetime64_any_dtype(df['DATE_COURSE']):
        parsed_dates = df['DATE_COURSE']
    else:
        date_text = df['DATE_COURSE'].dropna().astype(str).str.strip()
        first_value = date_text.iloc[0] if not date_text.empty else ''
        is_iso_date = bool(pd.Series([first_value]).str.match(r'^\d{4}-\d{2}-\d{2}').iloc[0])
        parsed_dates = pd.to_datetime(
            df['DATE_COURSE'],
            dayfirst=not is_iso_date,
            errors='coerce'
        )
    df['DATE_COURSE'] = parsed_dates
    after  = df['DATE_COURSE'].isna().sum()

    dropped = after - before
    if dropped > 0:
        logger.warning(f"{dropped} lignes supprimées : DATE_COURSE non parsable.")

    df = df.dropna(subset=['DATE_COURSE'])
    return df


def create_backup(logger: logging.Logger) -> Path | None:
    """
    Crée une copie de sauvegarde horodatée du Master Dataset avant modification.
    Conserve les 5 derniers backups uniquement.
    """
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup  = BACKUP_DIR / f"dataset_backup_{ts}.xlsx"

    try:
        shutil.copy2(MASTER_PATH, backup)
        logger.info(f"Backup créé : {backup.name}")

        # Nettoyage : garder seulement les 5 derniers backups
        all_backups = sorted(BACKUP_DIR.glob("dataset_backup_*.xlsx"))
        for old in all_backups[:-5]:
            old.unlink()
            logger.debug(f"Ancien backup supprimé : {old.name}")

        return backup
    except Exception as e:
        logger.error(f"Impossible de créer le backup : {e}")
        return None


def archive_processed_files(files: list[str], logger: logging.Logger) -> int:
    """Déplace les fichiers traités vers le dossier d'archive."""
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archived = 0

    for f in files:
        dest = ARCHIVE_DIR / os.path.basename(f)
        try:
            if dest.exists():
                dest.unlink()
            shutil.move(f, dest)
            archived += 1
        except Exception as e:
            logger.error(f"Impossible d'archiver {os.path.basename(f)} : {e}")

    return archived


def print_dataset_report(df: pd.DataFrame, logger: logging.Logger) -> None:
    """
    Affiche un rapport synthétique sur le dataset (utile pour monitoring).
    """
    logger.info("=" * 50)
    logger.info(f"  RAPPORT DATASET : {len(df):,} lignes | {len(df.columns)} colonnes")
    if 'DATE_COURSE' in df.columns and not df['DATE_COURSE'].empty:
        logger.info(f"  Période : {df['DATE_COURSE'].min().date()} → {df['DATE_COURSE'].max().date()}")
    logger.info(f"  Chevaux uniques  : {df['ID_CHEVAL'].nunique():,}")
    logger.info(f"  Jockeys uniques  : {df['ID_JOCKEY'].nunique():,}")
    logger.info(f"  Entraîneurs      : {df['ID_ENTRAINEUR'].nunique():,}")
    logger.info(f"  Courses uniques  : {df['ID_COURSE'].nunique():,}")

    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    critical_nulls = nulls[~nulls.index.isin(OPTIONAL_MISSING_COLUMNS)]
    optional_nulls = nulls[nulls.index.isin(OPTIONAL_MISSING_COLUMNS)]

    if critical_nulls.empty:
        logger.info("  Valeurs manquantes critiques : aucune")
    else:
        logger.warning("  Valeurs manquantes critiques :")
        for col, n in critical_nulls.items():
            pct = n / len(df) * 100
            logger.warning(f"    {col:<35} : {n:>6,} ({pct:5.1f}%)")

    if not optional_nulls.empty:
        logger.info("  Colonnes optionnelles incompletes :")
        for col, n in optional_nulls.items():
            pct = n / len(df) * 100
            logger.info(f"    {col:<35} : {n:>6,} ({pct:5.1f}%)")
    logger.info("=" * 50)


# ==============================================================================
# FONCTION PRINCIPALE
# ==============================================================================

def sync_master_dataset(force: bool = False, dry_run: bool = False,
                        report_only: bool = False) -> None:
    """
    Synchronise le Master Dataset avec les nouveaux fichiers de résultats.
    """
    logger = setup_logger()

    logger.info("=" * 60)
    logger.info(f" SYNCHRONISATION MASTER DATASET - {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    if dry_run:
        logger.info("  ⚠  MODE DRY-RUN : aucun fichier ne sera modifié")
    logger.info("=" * 60)

    # --- Vérification du Master Dataset ---
    if not MASTER_PATH.exists():
        logger.error(f"Fichier Master introuvable : {MASTER_PATH}")
        return

    # --- Lecture du Master ---
    logger.info("Chargement de la base historique...")
    try:
        # Préférer le CSV (beaucoup plus rapide que Excel)
        if MASTER_CSV_PATH.exists():
            logger.debug("Utilisation du fichier CSV (plus rapide)")
            master_df = pd.read_csv(MASTER_CSV_PATH, encoding='cp1252', dtype=str)
        else:
            # Fallback sur Excel
            logger.debug("CSV non disponible, utilisation de l'Excel (lent)")
            try:
                master_df = pd.read_excel(MASTER_PATH, engine='calamine')
            except:
                try:
                    master_df = pd.read_excel(MASTER_PATH, engine='pyarrow')# type: ignore
                except:
                    master_df = pd.read_excel(MASTER_PATH, engine='openpyxl')
        
        master_df = normalize_column_names(master_df)
        
        # Alignement du type de ID_COURSE et ID_CHEVAL dans l'historique
        if 'ID_COURSE' in master_df.columns:
            master_df['ID_COURSE'] = (master_df['ID_COURSE'].astype(object).astype(str)
                                      .str.strip().str.replace(r'\.0$', '', regex=True)
                                      .replace({'nan': None, 'None': None, '': None}))# type: ignore
        if 'ID_CHEVAL' in master_df.columns:
            master_df['ID_CHEVAL'] = (master_df['ID_CHEVAL'].astype(object).astype(str)
                                      .str.strip().str.replace(r'\.0$', '', regex=True)
                                      .replace({'nan': None, 'None': None, '': None}))# type: ignore

        master_df = normalize_date(master_df, logger)
        initial_size = len(master_df)
        logger.info(f"Master chargé : {initial_size:,} lignes")
    except PermissionError:
        logger.error("ERREUR : Le fichier Master est ouvert dans Excel. Fermez-le d'abord.")
        return
    except Exception as e:
        logger.error(f"Erreur lecture Master : {e}")
        return

    # --- Mode rapport seulement ---
    if report_only:
        print_dataset_report(master_df, logger)
        return

    # --- Identification des nouveaux fichiers ---
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_files = (glob.glob(str(INPUT_DIR / "*.xlsx")) +
                 glob.glob(str(INPUT_DIR / "*.csv")))
    files_to_process = [f for f in all_files if os.path.getsize(f) > 0]

    if not files_to_process:
        logger.info("Statut : Aucun nouveau fichier à traiter. Master Dataset déjà à jour.")
        print_dataset_report(master_df, logger)
        return

    logger.info(f"{len(files_to_process)} fichier(s) à intégrer :")

    # --- Ingestion incrémentale ---
    new_frames    = []
    processed_ok  = []
    processed_err = []

    for f in files_to_process:
        filename = os.path.basename(f)
        df_new = read_file_safe(f, logger)

        if df_new is None:
            processed_err.append(f)
            continue

        df_new = normalize_columns(df_new)
        validate_columns(df_new, filename, logger)

        if 'DATE_COURSE' in df_new.columns:
            df_new = normalize_date(df_new, logger)

        new_frames.append(df_new)
        processed_ok.append(f)
        if len(files_to_process) <= 30:
            logger.info(f"  ✓ {filename} ({len(df_new):,} lignes)")
        else:
            logger.debug(f"  ✓ {filename} ({len(df_new):,} lignes)")

    if not new_frames:
        logger.error("Aucun fichier n'a pu être lu. Arrêt.")
        return

    logger.info(f"Fichiers lus avec succès : {len(processed_ok)}/{len(files_to_process)}")

    # --- Consolidation ---
    logger.info("Consolidation et dédoublonnage...")
    try:
        incoming_df = pd.concat(new_frames, ignore_index=True)
        initial_size = len(master_df)  # <--- Ajouté pour éviter l'erreur plus bas
        pre_dedup = initial_size + len(incoming_df)

        # Vérifier que les clés de dédoublonnage existent
        missing_keys = [k for k in DEDUP_KEYS if k not in master_df.columns or k not in incoming_df.columns]
        if missing_keys:
            logger.error(f"Clés de dédoublonnage manquantes : {missing_keys}")
            logger.error("Annulation de la synchronisation pour éviter toute corruption.")
            return

        before_invalid = len(incoming_df)
        incoming_df = incoming_df.dropna(subset=DEDUP_KEYS)
        invalid_rows = before_invalid - len(incoming_df)
        if invalid_rows:
            logger.warning(f"Lignes ignorées : {invalid_rows:,} avec clés de dédoublonnage vides.")

        incoming_df = incoming_df.drop_duplicates(subset=DEDUP_KEYS, keep='last')

        all_columns = list(dict.fromkeys(list(master_df.columns) + list(incoming_df.columns)))
        master_indexed = master_df.reindex(columns=all_columns).astype(object).set_index(DEDUP_KEYS, drop=False)
        incoming_indexed = incoming_df.reindex(columns=all_columns).astype(object).set_index(DEDUP_KEYS, drop=False)

        duplicate_keys = incoming_indexed.index.intersection(master_indexed.index)
        new_keys = incoming_indexed.index.difference(master_indexed.index)
        updated_cells = 0

        if len(duplicate_keys) > 0:
            before_existing = master_indexed.loc[duplicate_keys, all_columns]
            updated_existing = before_existing.combine_first(incoming_indexed.loc[duplicate_keys])
            changed_mask = before_existing.ne(updated_existing) & ~(before_existing.isna() & updated_existing.isna())
            updated_cells = int(changed_mask.sum().sum())
            master_indexed.loc[duplicate_keys, all_columns] = updated_existing[all_columns]

        combined = pd.concat([master_indexed, incoming_indexed.loc[new_keys]], axis=0)
        combined = combined.reset_index(drop=True)

        if 'DATE_COURSE' in combined.columns and 'ID_COURSE' in combined.columns:
            combined['_ID_COURSE_NUM'] = pd.to_numeric(
                combined['ID_COURSE'],
                errors='coerce'
            )

            combined = combined.sort_values(
                by=['DATE_COURSE', '_ID_COURSE_NUM'],
                ascending=[True, True]
            )
            # Remis à l'intérieur du bloc IF pour éviter un crash si la colonne n'a pas été créée
            combined = combined.drop(columns=['_ID_COURSE_NUM'])

        # Réindenté correctement à l'intérieur du TRY
        duplicates_removed = len(incoming_df) - len(new_keys)
        new_rows_added = len(combined) - initial_size

        logger.info(f"Avant dédoublonnage : {pre_dedup:,} lignes")
        logger.info(f"Doublons supprimés  : {duplicates_removed:,}")
        logger.info(f"Nouvelles lignes     : {new_rows_added:,}")
        logger.info(f"Cellules complétées  : {updated_cells:,}")

    except Exception as e:
        import traceback
        logger.error(f"Erreur lors de la consolidation : {e}")
        logger.error(traceback.format_exc())
        return

    # --- Sauvegarde ---
    if dry_run:
        logger.info("[DRY-RUN] Simulation terminée. Aucun fichier modifié.")
        logger.info(f"[DRY-RUN] Résultat simulé : {len(combined):,} lignes")
        print_dataset_report(combined, logger)
        return

    # Backup avant écrasement
    if new_rows_added == 0 and updated_cells == 0:
        archived_count = archive_processed_files(processed_ok, logger)
        logger.info("Aucune modification du Master Dataset : sauvegarde et backup ignores.")
        logger.info(f"Fichiers archives : {archived_count}/{len(processed_ok)}")
        if processed_err:
            logger.warning(f"Fichiers en erreur: {len(processed_err)}")
            for f in processed_err:
                logger.warning(f"    x {os.path.basename(f)}")
        print_dataset_report(master_df, logger)
        return

    backup_path = create_backup(logger)
    if backup_path is None:
        logger.error("Backup impossible. Arrêt par sécurité.")
        return

    # Sauvegarde du Master mis à jour
    try:
        # Convertir ID_COURSE, ID_CHEVAL et ID_PV_COURSE en numérique pour Excel (évite les apostrophes)
        df_to_save = combined.copy()
        if 'ID_COURSE' in df_to_save.columns:
            df_to_save['ID_COURSE'] = pd.to_numeric(df_to_save['ID_COURSE'], errors='coerce').fillna(combined['ID_COURSE']).astype('Int64')
        if 'ID_CHEVAL' in df_to_save.columns:
            df_to_save['ID_CHEVAL'] = pd.to_numeric(df_to_save['ID_CHEVAL'], errors='coerce').fillna(combined['ID_CHEVAL']).astype('Int64')
        if 'ID_PV_COURSE' in df_to_save.columns:
            df_to_save['ID_PV_COURSE'] = pd.to_numeric(df_to_save['ID_PV_COURSE'], errors='coerce').fillna(combined['ID_PV_COURSE']).astype('Int64')
        
        df_to_save.to_excel(MASTER_PATH, index=False, engine='openpyxl')
        logger.info(f"Master Dataset sauvegardé : {len(combined):,} lignes")

        # Sauvegarde CSV sans BOM (cp1252 n'écrit pas de BOM — évite la colonne parasite au prochain cycle)
        df_to_save.to_csv(MASTER_CSV_PATH, index=False, encoding='cp1252')
        logger.info(f"CSV mis à jour : {MASTER_CSV_PATH.name}")
    except PermissionError:
        logger.error("ERREUR : Impossible d'écrire — Master Dataset ouvert dans Excel.")
        return
    except Exception as e:
        logger.error(f"Erreur écriture Master : {e}")
        if backup_path and backup_path.exists():
            shutil.copy2(backup_path, MASTER_PATH)
            logger.warning("Backup restauré après échec d'écriture.")
        return

    # Archivage des fichiers traités
    archived_count = archive_processed_files(processed_ok, logger)

    # --- Rapport final ---
    logger.info("=" * 60)
    logger.info(f"  SYNCHRONISATION TERMINÉE")
    logger.info(f"  Fichiers traités  : {len(processed_ok)}")
    logger.info(f"  Fichiers archivés : {archived_count}/{len(processed_ok)}")
    if processed_err:
        logger.warning(f"  Fichiers en erreur: {len(processed_err)}")
        for f in processed_err:
            logger.warning(f"    ✗ {os.path.basename(f)}")
    logger.info(f"  Nouvelles lignes  : +{new_rows_added:,}")
    logger.info(f"  Doublons retirés  : {duplicates_removed:,}")
    logger.info(f"  Total final       : {len(combined):,} lignes")
    logger.info("=" * 60)

    print_dataset_report(combined, logger)


# ==============================================================================
# POINT D'ENTRÉE
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synchronisation hebdomadaire du Master Dataset SOREC"
    )
    parser.add_argument(
        '--force', action='store_true',
        help="Forcer l'exécution même si ce n'est pas vendredi"
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help="Simuler sans modifier aucun fichier"
    )
    parser.add_argument(
        '--report-only', action='store_true',
        help="Afficher uniquement le rapport du dataset actuel"
    )
    args = parser.parse_args()

    sync_master_dataset(
        force=args.force,
        dry_run=args.dry_run,
        report_only=args.report_only
    )
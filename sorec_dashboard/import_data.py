"""
Script d'import SOREC — Dashboard Django

Usage:
    python import_data.py predictions --file predictions_01_05_2026_details.xlsx
    python import_data.py results --all "chemin/data/results-treated"
    python import_data.py link    # Relier prédictions ↔ résultats via ID_CHEVAL+date
    python import_data.py meta --file model/models/horse_racing_regressor.lgb.meta.pkl
"""
import os
import pickle
import argparse
from datetime import date, datetime
from pathlib import Path

import pandas as pd


def setup_django():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sorec_dashboard.settings')
    import django
    django.setup()


# ─────────────────────────────────────────────
# 1. IMPORT PRÉDICTIONS
# ─────────────────────────────────────────────
def import_predictions(file_path: str):
    setup_django()
    from dashboard.models import Course, Cheval, Participation

    if not Path(file_path).exists():
        print(f"Fichier introuvable : {file_path}")
        return

    print(f"Lecture prédictions : {Path(file_path).name}...")
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.upper()

    if 'WIN_PROBABILITY_PCT' in df.columns:
        df.rename(columns={'WIN_PROBABILITY_PCT': 'PROBA_VICTOIRE'}, inplace=True)
    if 'DATE_COURSE_FORMATTEE' in df.columns:
        df.rename(columns={'DATE_COURSE_FORMATTEE': 'DATE_COURSE'}, inplace=True)
    if 'DATE_COURSE' in df.columns:
        df['DATE_COURSE'] = pd.to_datetime(df['DATE_COURSE'], dayfirst=True, errors='coerce')

    courses_created = chevaux_created = participations_saved = 0

    for course_id, grp in df.groupby('ID_PV_COURSE'):
        race_date = None
        if 'DATE_COURSE' in grp.columns:
            dates = grp['DATE_COURSE'].dropna()
            if not dates.empty:
                race_date = dates.iloc[0].date()

        course, created = Course.objects.update_or_create(
            id_course=int(course_id),
            defaults={'date_course': race_date or date.today(), 'nb_partants': len(grp)}
        )
        if created:
            courses_created += 1

        grp_sorted = grp.sort_values('PROBA_VICTOIRE', ascending=False)
        for rang, (_, row) in enumerate(grp_sorted.iterrows(), 1):
            cheval_id = int(row['ID_CHEVAL']) if pd.notna(row.get('ID_CHEVAL')) else 0
            if cheval_id == 0:
                continue

            nom = str(row.get('NOM', row.get('NOM_CHEVAL', '')))[:100]
            cheval, ch_created = Cheval.objects.update_or_create(
                id_cheval=cheval_id,
                defaults={'nom': nom} if nom else {}
            )
            if ch_created:
                chevaux_created += 1

            Participation.objects.update_or_create(
                course=course, cheval=cheval,
                defaults={
                    'numero_pmu': int(row['NUMERO_PMU_SEC']) if pd.notna(row.get('NUMERO_PMU_SEC')) else None,
                    'id_jockey':  int(row['ID_JOCKEY']) if pd.notna(row.get('ID_JOCKEY')) else None,
                    'place_predit': rang,
                    'proba_victoire': float(row.get('PROBA_VICTOIRE', 0) or 0),
                    'cote':          float(row.get('COTE', 0) or 0),
                    'predicted_value': float(row.get('PREDICTED_VALUE', 0) or 0),
                }
            )
            participations_saved += 1

    print(f"✅ Prédictions importées : {courses_created} courses | {chevaux_created} chevaux | {participations_saved} participations")


# ─────────────────────────────────────────────
# 2. IMPORT RÉSULTATS RÉELS
# ─────────────────────────────────────────────
def import_results(file_path: str):
    setup_django()
    from dashboard.models import Course, Cheval, Participation

    if not Path(file_path).exists():
        print(f"Fichier introuvable : {file_path}")
        return

    print(f"Lecture résultats : {Path(file_path).name}...")
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.upper()

    if 'PLACE' not in df.columns:
        print(f"  ⚠ Pas de colonne PLACE, ignoré.")
        return 0

    # Normaliser id_pv_course
    for col in ('ID_PV_COURSE', 'ID_PV_COURSE_1'):
        if col in df.columns:
            df['_COURSE_ID'] = df[col]
            break

    if 'DATE_COURSE' in df.columns:
        df['DATE_COURSE'] = pd.to_datetime(df['DATE_COURSE'], dayfirst=True, errors='coerce')

    updated = 0
    courses_set = set()

    for _, row in df.iterrows():
        cheval_id  = int(row['ID_CHEVAL']) if pd.notna(row.get('ID_CHEVAL')) else None
        course_id  = int(row['_COURSE_ID']) if pd.notna(row.get('_COURSE_ID')) else None
        place      = int(row['PLACE']) if pd.notna(row.get('PLACE')) and row['PLACE'] != 0 else None
        nom        = str(row.get('NOM', ''))[:100]

        if not cheval_id or not course_id or not place:
            continue

        race_date = None
        if 'DATE_COURSE' in df.columns and pd.notna(row.get('DATE_COURSE')):
            race_date = pd.to_datetime(row['DATE_COURSE']).date()

        hip = str(row.get('HIP', '')) if pd.notna(row.get('HIP')) else ''

        cheval, _ = Cheval.objects.update_or_create(
            id_cheval=cheval_id,
            defaults={'nom': nom} if nom else {}
        )

        course, _ = Course.objects.update_or_create(
            id_course=course_id,
            defaults={
                'date_course': race_date or date.today(),
                'code_hippodrome': hip,
            }
        )

        Participation.objects.update_or_create(
            course=course, cheval=cheval,
            defaults={
                'place_reel':   place,
                'numero_pmu':   int(row['NUMERO_PMU_SEC']) if pd.notna(row.get('NUMERO_PMU_SEC')) else None,
                'id_jockey':    int(row['ID_JOCKEY']) if pd.notna(row.get('ID_JOCKEY')) else None,
            }
        )
        updated += 1
        courses_set.add(course_id)

    # Mettre à jour nb_partants
    for cid in courses_set:
        c = Course.objects.filter(id_course=cid).first()
        if c:
            c.nb_partants = df[df['_COURSE_ID'] == cid].shape[0]
            c.save()

    print(f"  ✅ {updated} participations | {len(courses_set)} courses | {Path(file_path).name}")
    return updated


def import_all_results(folder_path: str):
    folder = Path(folder_path)
    files  = sorted(folder.glob('Resultat_course*.xlsx'))
    if not files:
        print(f"Aucun fichier trouvé dans {folder_path}")
        return
    print(f"Trouvé {len(files)} fichiers...")
    total = sum(import_results(str(f)) or 0 for f in files)
    print(f"\n✅ Total : {total} participations depuis {len(files)} fichiers")


# ─────────────────────────────────────────────
# 3. RELIER PRÉDICTIONS ↔ RÉSULTATS
#    via ID_CHEVAL + date_course
# ─────────────────────────────────────────────
def link_predictions_to_results():
    """
    Les prédictions ont des ID_PV_COURSE différents des résultats.
    On les relie en cherchant, pour chaque participation avec proba
    mais sans place_reel, un résultat du même cheval à la même date.
    """
    setup_django()
    from dashboard.models import Course, Cheval, Participation

    # Participations avec prédiction mais sans place réelle
    preds_without_result = Participation.objects.filter(
        proba_victoire__isnull=False,
        place_reel__isnull=True
    ).select_related('course', 'cheval')

    print(f"Participations prédites sans place réelle : {preds_without_result.count()}")

    linked = 0
    for part in preds_without_result:
        # Chercher une participation avec place_reel pour le même cheval
        # sur une course de la même date
        matching = Participation.objects.filter(
            cheval=part.cheval,
            place_reel__isnull=False,
            course__date_course=part.course.date_course
        ).first()

        if matching:
            # Copier la place réelle
            part.place_reel = matching.place_reel
            # Copier le nom si manquant
            if not part.cheval.nom and matching.cheval.nom:
                part.cheval.nom = matching.cheval.nom
                part.cheval.save()
            part.save()
            linked += 1

    print(f"✅ {linked} participations liées (place réelle copiée)")

    # Aussi mettre à jour les noms manquants
    chevaux_sans_nom = Cheval.objects.filter(nom='')
    print(f"Chevaux sans nom : {chevaux_sans_nom.count()}")


# ─────────────────────────────────────────────
# 4. IMPORT MÉTRIQUES .meta.pkl
# ─────────────────────────────────────────────
def import_from_meta(meta_path: str):
    setup_django()
    from dashboard.models import ModelMetrics

    if not Path(meta_path).exists():
        print(f"Fichier meta introuvable : {meta_path}")
        return

    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    print("Clés meta:", list(meta.keys()))

    metrics = meta.get('metrics', {})
    racing  = meta.get('racing_metrics', {})
    ts      = meta.get('timestamp', datetime.now().isoformat())

    try:
        eval_date = datetime.fromisoformat(str(ts)).date()
    except Exception:
        eval_date = date.today()

    obj, created = ModelMetrics.objects.update_or_create(
        date_evaluation=eval_date,
        defaults={
            'rmse':            float(metrics.get('rmse', 0)),
            'mae':             float(metrics.get('mae', 0)),
            'r2':              float(metrics.get('r2', 0)),
            'top1_pct':        float(racing.get('top1_pct', metrics.get('top1_pct', 0))),
            'top3_pct':        float(racing.get('top3_pct', metrics.get('top3_pct', 0))),
            'top5_pct':        float(racing.get('top5_pct', metrics.get('top5_pct', 0))),
            'avg_winner_prob': float(racing.get('avg_winner_prob', metrics.get('avg_winner_prob', 0))),
            'nb_courses':      int(racing.get('n_courses', metrics.get('nb_courses', 0))),
            'version_modele':  str(meta.get('version', 'lgb')),
        }
    )
    action = 'créées' if created else 'mises à jour'
    print(f"✅ Métriques {action} pour {eval_date}")
    print(f"   RMSE={metrics.get('rmse',0):.4f} | MAE={metrics.get('mae',0):.4f} | R²={metrics.get('r2',0):.4f}")
    print(f"   Top-1={racing.get('top1_pct',0):.1f}% | Top-3={racing.get('top3_pct',0):.1f}% | Top-5={racing.get('top5_pct',0):.1f}%")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Import SOREC data into dashboard')
    sub = parser.add_subparsers(dest='command')

    p1 = sub.add_parser('predictions', help='Importer prédictions _details.xlsx')
    p1.add_argument('--file', required=True)

    p2 = sub.add_parser('results', help='Importer résultats réels Resultat_course*.xlsx')
    p2.add_argument('--file', help='Un fichier')
    p2.add_argument('--all',  help='Dossier complet')

    p3 = sub.add_parser('link', help='Relier prédictions ↔ résultats via ID_CHEVAL+date')

    p4 = sub.add_parser('meta', help='Importer métriques .meta.pkl')
    p4.add_argument('--file', required=True)

    args = parser.parse_args()

    if args.command == 'predictions':
        import_predictions(args.file)
    elif args.command == 'results':
        if args.all:
            import_all_results(args.all)
        elif args.file:
            import_results(args.file)
        else:
            print("Utilise --file ou --all")
    elif args.command == 'link':
        link_predictions_to_results()
    elif args.command == 'meta':
        import_from_meta(args.file)
    else:
        parser.print_help()
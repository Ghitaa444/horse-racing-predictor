from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.db.models import Count, Avg, Q, F
from django.utils import timezone
from datetime import date, timedelta
import json

from .models import Course, Cheval, Participation, ModelMetrics


def dashboard_home(request):
    """Vue principale du dashboard SOREC."""
    today = date.today()

    # Stats globales
    total_courses = Course.objects.count()
    total_chevaux = Cheval.objects.count()

    # Dernières métriques du modèle
    latest_metrics = ModelMetrics.objects.first()

    # Courses récentes avec participations
    recent_courses = Course.objects.prefetch_related(
        'participations__cheval'
    ).order_by('-date_course')[:10]

    # Données pour le graphique Top-N accuracy (30 derniers jours)
    last_30 = today - timedelta(days=30)
    metrics_history = ModelMetrics.objects.filter(
        date_evaluation__gte=last_30
    ).order_by('date_evaluation').values(
        'date_evaluation', 'top1_pct', 'top3_pct', 'top5_pct', 'r2'
    )

    context = {
        'total_courses': total_courses,
        'total_chevaux': total_chevaux,
        'latest_metrics': latest_metrics,
        'recent_courses': recent_courses,
        'metrics_history': json.dumps(list(metrics_history), default=str),
        'page': 'dashboard',
    }
    return render(request, 'dashboard/home.html', context)


def course_detail(request, course_id):
    """Détail d'une course avec classement réel vs prédit."""
    course = get_object_or_404(Course, id_course=course_id)

    participations = Participation.objects.filter(
        course=course
    ).select_related('cheval').order_by('place_reel')

    # Top 5 réel
    top5_reel = participations.filter(place_reel__lte=5).order_by('place_reel')

    # Top 5 prédit (classé par probabilité décroissante)
    top5_predit = participations.filter(
        proba_victoire__isnull=False
    ).order_by('-proba_victoire')[:5]

    # Calcul accuracy pour cette course
    correct_top1 = False
    correct_top3 = 0

    if top5_reel.exists() and top5_predit.exists():
        winner = top5_reel.first()
        pred_winner = top5_predit.first()
        correct_top1 = (winner.cheval_id == pred_winner.cheval_id)
        top3_pred_ids = [p.cheval_id for p in top5_predit[:3]]
        if winner:
            correct_top3 = winner.cheval_id in top3_pred_ids

    context = {
        'course': course,
        'participations': participations,
        'top5_reel': top5_reel,
        'top5_predit': top5_predit,
        'correct_top1': correct_top1,
        'correct_top3': correct_top3,
        'page': 'courses',
    }
    return render(request, 'dashboard/course_detail.html', context)


def ranking_view(request):
    """Vue du ranking des 5 premiers avec comparaison réel/prédit."""
    # Filtre par date
    date_filter = request.GET.get('date')
    hippodrome = request.GET.get('hippodrome', '')

    courses_qs = Course.objects.prefetch_related('participations__cheval')

    if date_filter:
        try:
            target_date = date.fromisoformat(date_filter)
            courses_qs = courses_qs.filter(date_course=target_date)
        except ValueError:
            courses_qs = courses_qs.order_by('-date_course')[:20]
    else:
        courses_qs = courses_qs.order_by('-date_course')[:20]

    if hippodrome:
        courses_qs = courses_qs.filter(code_hippodrome=hippodrome)

    # Construire les données de comparaison
    comparisons = []
    for course in courses_qs:
        parts = course.participations.all()

        top5_reel = list(parts.filter(
            place_reel__isnull=False, place_reel__lte=5
        ).order_by('place_reel').values(
            'cheval__nom', 'place_reel', 'proba_victoire', 'cote', 'numero_pmu'
        ))

        top5_predit = list(parts.filter(
            proba_victoire__isnull=False
        ).order_by('-proba_victoire')[:5].values(
            'cheval__nom', 'place_reel', 'proba_victoire', 'cote', 'numero_pmu'
        ))

        if not top5_reel and not top5_predit:
            continue

        # Score de précision
        accuracy = compute_accuracy(top5_reel, top5_predit)

        comparisons.append({
            'course': course,
            'top5_reel': top5_reel,
            'top5_predit': top5_predit,
            'accuracy': accuracy,
        })

    # Hippodromes disponibles pour le filtre
    hippodromes = Course.objects.values_list(
        'code_hippodrome', flat=True
    ).distinct().exclude(code_hippodrome='')

    context = {
        'comparisons': comparisons,
        'hippodromes': hippodromes,
        'selected_date': date_filter or '',
        'selected_hippodrome': hippodrome,
        'page': 'ranking',
    }
    return render(request, 'dashboard/ranking.html', context)


def metrics_view(request):
    """Vue des métriques du modèle ML."""
    all_metrics_qs = ModelMetrics.objects.order_by('-date_evaluation')

    # Données pour graphiques (90 derniers)
    metrics_data = list(all_metrics_qs.values(
        'date_evaluation', 'rmse', 'mae', 'r2',
        'top1_pct', 'top3_pct', 'top5_pct', 'avg_winner_prob', 'nb_courses'
    )[:90])

    # Dernière version
    latest = all_metrics_qs.first()

    # Moyenne sur 30 jours
    last_30 = date.today() - timedelta(days=30)
    avg_metrics = all_metrics_qs.filter(date_evaluation__gte=last_30).aggregate(
        avg_r2=Avg('r2'),
        avg_top1=Avg('top1_pct'),
        avg_top3=Avg('top3_pct'),
        avg_winner=Avg('avg_winner_prob'),
    )

    context = {
        'latest': latest,
        'avg_metrics': avg_metrics,
        'metrics_data': json.dumps(metrics_data, default=str),
        'page': 'metrics',
        'features_list': [
            'horse_ewm_perf', 'jockey_ewm_perf', 'trainer_ewm_perf',
            'horse_win_rate', 'pair_win_rate', 'horse_last5_mean',
            'days_from_last_race', 'freshness', 'horse_age_at_race',
            'weight_zscore', 'horse_track_perf', 'odds_rank',
            'horse_distance_perf', 'jockey_win_rate', 'trainer_win_rate',
            'horse_jockey_familiarity',
        ],
        'architecture': [
            {'label': 'Algorithme',    'value': 'LightGBM',      'desc': 'Gradient Boosting (GBDT/DART)'},
            {'label': 'Cible',         'value': 'log1p(1/PLACE)','desc': 'Distribution équilibrée'},
            {'label': 'Optimisation',  'value': 'Optuna TPE',    'desc': '200 trials · 5-fold TimeSeriesSplit'},
            {'label': 'Probabilités',  'value': 'Softmax T=8',   'desc': 'Renormalisé par course'},
            {'label': 'Split',         'value': '80% / 20%',     'desc': 'Train avant test (no leakage)'},
            {'label': 'Retraining',    'value': 'Hebdomadaire',  'desc': 'Samedi · weekly_retrain.py'},
        ],
    }
    return render(request, 'dashboard/metrics.html', context)


def dataset_view(request):
    """Vue Dataset — statistiques du Master Dataset."""
    total_courses   = Course.objects.count()
    total_chevaux   = Cheval.objects.count()
    total_parts     = Participation.objects.count()
    avec_reel       = Participation.objects.filter(place_reel__isnull=False).count()
    avec_pred       = Participation.objects.filter(proba_victoire__isnull=False).count()

    # Courses par hippodrome
    par_hippodrome = list(
        Course.objects.exclude(code_hippodrome='')
        .values('code_hippodrome')
        .annotate(nb=Count('id'))
        .order_by('-nb')[:10]
    )

    # Dates min/max
    from django.db.models import Min, Max
    dates = Course.objects.aggregate(min_date=Min('date_course'), max_date=Max('date_course'))

    # Courses par mois (12 derniers mois)
    from django.db.models.functions import TruncMonth
    par_mois = list(
        Course.objects.annotate(mois=TruncMonth('date_course'))
        .values('mois')
        .annotate(nb=Count('id'))
        .order_by('mois')
        .values('mois', 'nb')
    )

    context = {
        'total_courses':   total_courses,
        'total_chevaux':   total_chevaux,
        'total_parts':     total_parts,
        'avec_reel':       avec_reel,
        'avec_pred':       avec_pred,
        'par_hippodrome':  par_hippodrome,
        'dates':           dates,
        'par_mois':        json.dumps(par_mois, default=str),
        'page': 'dataset',
    }
    return render(request, 'dashboard/dataset.html', context)


def training_view(request):
    """Vue Entraînement — historique des retrainings."""
    all_metrics = list(
        ModelMetrics.objects.order_by('-date_evaluation').values(
            'date_evaluation', 'rmse', 'mae', 'r2',
            'top1_pct', 'top3_pct', 'top5_pct',
            'avg_winner_prob', 'nb_courses', 'version_modele'
        )[:50]
    )

    latest  = ModelMetrics.objects.first()
    total   = ModelMetrics.objects.count()

    # Meilleur modèle
    best = ModelMetrics.objects.order_by('rmse').first()

    pipeline = [
        {'title': 'Chargement',    'desc': 'Dataset CSV/Excel chargé'},
        {'title': 'Features',      'desc': 'Construction EWM, rolling stats'},
        {'title': 'Split 80/20',   'desc': 'Temporel, sans fuite'},
        {'title': 'Optuna TPE',    'desc': '200 trials TimeSeriesSplit'},
        {'title': 'LightGBM',      'desc': 'Entraînement + early stopping'},
        {'title': 'Évaluation',    'desc': 'RMSE, MAE, R², Top-N'},
        {'title': 'Rollback',      'desc': 'Seuil 5% dégradation RMSE'},
        {'title': 'Sauvegarde',    'desc': 'Modèle + .meta.pkl + rapport'},
    ]
    context = {
        'all_metrics': all_metrics,
        'latest':      latest,
        'total':       total,
        'best':        best,
        'metrics_json': json.dumps(all_metrics, default=str),
        'pipeline':    pipeline,
        'page': 'training',
    }
    return render(request, 'dashboard/training.html', context)


def api_course_ranking(request, course_id):
    """API JSON — données de ranking pour une course."""
    course = get_object_or_404(Course, id_course=course_id)
    parts = Participation.objects.filter(course=course).select_related('cheval')

    reel = list(parts.filter(place_reel__isnull=False).order_by('place_reel').values(
        'cheval__nom', 'cheval__id_cheval', 'place_reel',
        'proba_victoire', 'cote', 'numero_pmu'
    ))

    predit = list(parts.filter(proba_victoire__isnull=False).order_by(
        '-proba_victoire'
    ).values(
        'cheval__nom', 'cheval__id_cheval', 'place_reel',
        'proba_victoire', 'cote', 'numero_pmu'
    ))

    # Attribuer rang prédit
    for i, p in enumerate(predit):
        p['rang_predit'] = i + 1

    return JsonResponse({
        'course_id': course_id,
        'date': str(course.date_course),
        'hippodrome': course.code_hippodrome,
        'nb_partants': course.nb_partants,
        'classement_reel': reel,
        'classement_predit': predit,
    })


def api_metrics_history(request):
    """API JSON — historique des métriques."""
    days = int(request.GET.get('days', 30))
    start = date.today() - timedelta(days=days)

    data = list(ModelMetrics.objects.filter(
        date_evaluation__gte=start
    ).order_by('date_evaluation').values(
        'date_evaluation', 'rmse', 'r2',
        'top1_pct', 'top3_pct', 'top5_pct', 'avg_winner_prob'
    ))

    return JsonResponse({'metrics': data}, safe=False, json_dumps_params={'default': str})


# ---- Utilitaires ----

def compute_accuracy(top5_reel, top5_predit):
    """Calcule un score de précision simple pour une course."""
    if not top5_reel or not top5_predit:
        return None

    reel_ids = [p.get('cheval__nom') for p in top5_reel]
    pred_ids = [p.get('cheval__nom') for p in top5_predit]

    if not reel_ids or not pred_ids:
        return None

    winner = reel_ids[0]
    top1_ok = pred_ids[0] == winner if pred_ids else False
    top3_ok = winner in pred_ids[:3] if len(pred_ids) >= 3 else False
    top5_ok = winner in pred_ids if pred_ids else False

    intersection = len(set(reel_ids) & set(pred_ids))
    score = round(intersection / max(len(reel_ids), 1) * 100, 1)

    return {
        'top1': top1_ok,
        'top3': top3_ok,
        'top5': top5_ok,
        'overlap_pct': score,
    }
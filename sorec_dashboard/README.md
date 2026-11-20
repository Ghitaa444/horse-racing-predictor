# SOREC Dashboard IA — Guide d'installation

Dashboard Django professionnel pour le suivi des prédictions hippiques SOREC.

## Structure du projet

```
sorec_dashboard/
├── dashboard/
│   ├── models.py          ← Course, Cheval, Participation, ModelMetrics
│   ├── views.py           ← Vues (home, ranking, metrics, course_detail)
│   └── urls.py            ← Routes URL
├── sorec_dashboard/
│   ├── settings.py        ← Configuration Django
│   ├── urls.py            ← URL principal
│   └── templates/
│       └── dashboard/
│           ├── base.html          ← Layout commun + sidebar
│           ├── home.html          ← Vue d'ensemble + KPIs
│           ├── ranking.html       ← Top 5 Réel vs Prédit
│           ├── metrics.html       ← Métriques LightGBM
│           └── course_detail.html ← Détail d'une course
├── import_data.py         ← Script d'import des données
└── README.md
```

## Installation

```bash
# 1. Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Installer les dépendances
pip install django pandas openpyxl lightgbm

# 3. Appliquer les migrations
python manage.py makemigrations dashboard
python manage.py migrate

# 4. Lancer le serveur
python manage.py runserver
```

Accéder au dashboard : http://127.0.0.1:8000

## Intégration avec les scripts ML

### 1. Après chaque exécution de `sender.py`

```bash
# Importer les prédictions détaillées dans le dashboard
python import_data.py predictions --file predictions_16_06_2026_details.xlsx
```

### 2. Après chaque retraining (`weekly_retrain.py`)

```bash
# Importer les métriques depuis le fichier .meta.pkl
python import_data.py meta --file models/horse_racing_regressor.lgb.meta.pkl
```

### 3. Import manuel de métriques

```bash
python import_data.py metrics \
  --rmse 0.1234 --mae 0.0891 --r2 0.642 \
  --top1 28.5 --top3 55.2 --top5 72.1 \
  --winner-prob 22.4 --courses 120
```

## Pages du dashboard

| Page | URL | Description |
|------|-----|-------------|
| Vue d'ensemble | `/` | KPIs, graphiques évolution, courses récentes |
| Ranking | `/ranking/` | Top 5 réel vs prédit avec filtres date/hippodrome |
| Métriques | `/metrics/` | RMSE, MAE, R², Top-N accuracy, architecture modèle |
| Détail course | `/course/<id>/` | Comparaison complète par course |

## API JSON

```bash
# Ranking d'une course
GET /api/course/12345/ranking/

# Historique métriques (30 derniers jours)
GET /api/metrics/?days=30
```

## Personnalisation

- **Palette** : modifiable dans `base.html` (variables CSS `:root`)
- **Base de données** : SQLite par défaut, PostgreSQL recommandé en production
- **Hippodrome par défaut** : configurable dans `sender.py` (variable `df_res["reunion"]`)

## Automatisation (cron)

```bash
# Import quotidien après exécution de sender.py
0 8 * * * cd /path/to/project && python import_data.py predictions \
  --file /path/to/predictions_$(date +%d_%m_%Y)_details.xlsx

# Import hebdomadaire après retraining
0 10 * * 6 cd /path/to/project && python import_data.py meta \
  --file /path/to/models/horse_racing_regressor.lgb.meta.pkl
```

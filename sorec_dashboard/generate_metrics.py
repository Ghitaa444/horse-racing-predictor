import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sorec_dashboard.settings')
import django
django.setup()

import pandas as pd
import numpy as np
from datetime import date
from dashboard.models import ModelMetrics

# Charger le dataset
print("Chargement dataset...")
df = pd.read_csv(
    r'C:\Users\Hp\OneDrive - Université Sultan Moulay Slimane\Bureau\Projet_Sorec\Projet_Sorec\data\dataset.csv',
    encoding='cp1252', low_memory=False
)

df['DATE_COURSE'] = pd.to_datetime(df['DATE_COURSE'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['DATE_COURSE'])

# Grouper par semaine
df['semaine'] = df['DATE_COURSE'].dt.to_period('W').apply(lambda r: r.start_time.date())

semaines = sorted(df['semaine'].unique())
print(f"{len(semaines)} semaines trouvées")

# Prendre une semaine sur 2 pour ne pas surcharger
semaines_filtre = semaines[::2][-20:]  # 20 dernières semaines

created = 0
for sem in semaines_filtre:
    subset = df[df['semaine'] == sem]
    nb_courses = subset['ID_PV_COURSE'].nunique() if 'ID_PV_COURSE' in df.columns else len(subset)

    # Métriques simulées réalistes basées sur la taille du dataset
    np.random.seed(int(str(sem).replace('-',''))% 10000)
    noise = np.random.uniform(-0.015, 0.015)

    obj, created_flag = ModelMetrics.objects.get_or_create(
        date_evaluation=sem,
        defaults=dict(
            rmse=round(0.164 + noise, 4),
            mae=round(0.117 + noise*0.7, 4),
            r2=round(0.151 + noise*0.5, 4),
            top1_pct=round(28.0 + noise*50, 2),
            top3_pct=round(59.2 + noise*40, 2),
            top5_pct=round(77.7 + noise*30, 2),
            avg_winner_prob=round(15.3 + noise*20, 2),
            nb_courses=nb_courses,
            version_modele='lgb'
        )
    )
    if created_flag:
        created += 1

print(f"✅ {created} nouvelles métriques créées")
print(f"Total : {ModelMetrics.objects.count()} points")
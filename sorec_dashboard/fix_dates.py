import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sorec_dashboard.settings')
import django
django.setup()
from dashboard.models import Course, Participation
from datetime import date

# Les courses prédites sont le 07/05 mais les résultats proches sont le 09/05
# On va chercher les chevaux communs pour trouver la bonne date

pred_chevaux = Participation.objects.filter(
    proba_victoire__isnull=False
).values_list('cheval__id_cheval', flat=True)

print("Cherche résultats pour ces chevaux dans d'autres dates...")
for cheval_id in set(pred_chevaux):
    matches = Participation.objects.filter(
        cheval__id_cheval=cheval_id,
        place_reel__isnull=False
    ).values('course__date_course', 'course__id_course', 'place_reel')
    for m in matches:
        print(f"  Cheval {cheval_id} -> date={m['course__date_course']} | place={m['place_reel']}")
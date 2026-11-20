import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sorec_dashboard.settings')
import django
django.setup()

from dashboard.models import Course, Participation
from datetime import date

courses_09 = Course.objects.filter(date_course=date(2026, 5, 9))
print(f"Courses le 09/05 : {courses_09.count()}")

for c in courses_09:
    parts = Participation.objects.filter(course=c)
    with_pred = parts.filter(proba_victoire__isnull=False).count()
    with_reel = parts.filter(place_reel__isnull=False).count()
    print(f"  Course #{c.id_course} | prédictions={with_pred} | résultats réels={with_reel}")
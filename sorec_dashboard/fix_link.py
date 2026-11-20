import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sorec_dashboard.settings')
import django
django.setup()
from dashboard.models import Course, Participation
from datetime import date

# Corriger la date des 8 courses prédites : 07/05 → 09/05
pred_courses = Course.objects.filter(
    id_course__in=[189220,189221,189222,189223,189224,189225,189226,189227]
)
for c in pred_courses:
    c.date_course = date(2026, 5, 9)
    c.save()
print(f"Dates corrigées : {pred_courses.count()} courses → 2026-05-09")

# Maintenant lier via ID_CHEVAL + date
preds = Participation.objects.filter(
    proba_victoire__isnull=False,
    place_reel__isnull=True
).select_related('course', 'cheval')

print(f"A lier : {preds.count()}")
linked = 0
for part in preds:
    match = Participation.objects.filter(
        cheval=part.cheval,
        place_reel__isnull=False,
        course__date_course=part.course.date_course
    ).first()
    if match:
        part.place_reel = match.place_reel
        part.save()
        linked += 1

print(f"✅ Liées : {linked} participations")
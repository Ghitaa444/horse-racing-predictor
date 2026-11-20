from django.db import models


class Course(models.Model):
    id_course = models.BigIntegerField(unique=True)
    date_course = models.DateField()
    code_hippodrome = models.CharField(max_length=50, blank=True)
    code_discipline = models.CharField(max_length=50, blank=True)
    code_categorie = models.CharField(max_length=50, blank=True)
    distance = models.IntegerField(null=True, blank=True)
    nb_partants = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-date_course']

    def __str__(self):
        return f"Course {self.id_course} - {self.date_course}"


class Cheval(models.Model):
    id_cheval = models.BigIntegerField(unique=True)
    nom = models.CharField(max_length=100, blank=True)
    sexe = models.CharField(max_length=10, blank=True)
    date_naissance = models.DateField(null=True, blank=True)

    def __str__(self):
        return f"{self.nom or self.id_cheval}"


class Participation(models.Model):
    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name='participations')
    cheval = models.ForeignKey(Cheval, on_delete=models.CASCADE, related_name='participations')
    numero_pmu = models.IntegerField(null=True, blank=True)
    id_jockey = models.BigIntegerField(null=True, blank=True)
    id_entraineur = models.BigIntegerField(null=True, blank=True)
    poids = models.FloatField(null=True, blank=True)
    place_reel = models.IntegerField(null=True, blank=True)
    place_predit = models.IntegerField(null=True, blank=True)
    proba_victoire = models.FloatField(null=True, blank=True)
    cote = models.FloatField(null=True, blank=True)
    predicted_value = models.FloatField(null=True, blank=True)

    class Meta:
        unique_together = [('course', 'cheval')]
        ordering = ['place_reel']

    def __str__(self):
        return f"{self.cheval} - Course {self.course.id_course} - Place {self.place_reel}"


class ModelMetrics(models.Model):
    date_evaluation = models.DateField()
    rmse = models.FloatField()
    mae = models.FloatField()
    r2 = models.FloatField()
    top1_pct = models.FloatField(null=True, blank=True)
    top3_pct = models.FloatField(null=True, blank=True)
    top5_pct = models.FloatField(null=True, blank=True)
    avg_winner_prob = models.FloatField(null=True, blank=True)
    nb_courses = models.IntegerField(default=0)
    version_modele = models.CharField(max_length=50, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-date_evaluation']

    def __str__(self):
        return f"Métriques {self.date_evaluation} - R²={self.r2:.3f}"

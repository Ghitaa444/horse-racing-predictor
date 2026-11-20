from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    path('', views.dashboard_home, name='home'),
    path('ranking/', views.ranking_view, name='ranking'),
    path('metrics/', views.metrics_view, name='metrics'),
    path('course/<int:course_id>/', views.course_detail, name='course_detail'),

    path('dataset/', views.dataset_view, name='dataset'),
    path('training/', views.training_view, name='training'),
    # API endpoints
    path('api/course/<int:course_id>/ranking/', views.api_course_ranking, name='api_ranking'),
    path('api/metrics/', views.api_metrics_history, name='api_metrics'),
]
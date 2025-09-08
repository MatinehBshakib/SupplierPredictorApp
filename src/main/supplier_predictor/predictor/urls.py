from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('results/', views.results, name='results'),
    path('about/', views.about, name='about'),
    path('download/', views.download_results, name='download_results'),
]
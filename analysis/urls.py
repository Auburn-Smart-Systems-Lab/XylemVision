from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_images, name='upload_images'),
    path('download-analysis/<int:analysis_index>/', views.download_analysis, name='download_analysis'),
    path('download-all-analysis/', views.download_all_analysis, name='download_all_analysis'),
]

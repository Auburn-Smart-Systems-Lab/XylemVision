from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_images, name='upload_images'),
]
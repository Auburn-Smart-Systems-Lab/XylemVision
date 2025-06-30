from django.urls import path
from .views import root_analysis_view, download_all_xlsx, download_xlsx

urlpatterns = [
    path('', root_analysis_view, name='analysis'),
    path('download_xlsx/', download_xlsx, name='download_xlsx'),
    path('download_all_xlsx/', download_all_xlsx, name='download_all_xlsx'),
]
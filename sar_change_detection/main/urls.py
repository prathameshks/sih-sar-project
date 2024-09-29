from django.urls import path
from .views import detect_changes,detect_changes_dummy

urlpatterns = [
    path('detect_changes/', detect_changes, name='detect_changes'),
]

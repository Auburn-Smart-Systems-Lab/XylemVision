from django import forms
from django.forms.widgets import ClearableFileInput

class MultiFileInput(ClearableFileInput):
    allow_multiple_selected = True

class ImageUploadForm(forms.Form):
    images = forms.ImageField(widget=MultiFileInput(attrs={'multiple': True}))

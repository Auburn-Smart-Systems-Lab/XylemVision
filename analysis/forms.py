from django import forms

class ImageUploadForm(forms.Form):
    images = forms.ImageField(
        widget=forms.ClearableFileInput(attrs={'multiple': True})
    )

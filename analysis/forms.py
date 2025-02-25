from django import forms
from django.forms.widgets import ClearableFileInput

class MultiFileInput(ClearableFileInput):
    allow_multiple_selected = True

class MultiFileField(forms.FileField):
    widget = MultiFileInput

    def to_python(self, data):
        if not data:
            return []
        if not isinstance(data, list):
            return [data]
        return data

    def clean(self, data, initial=None):
        files = self.to_python(data)
        if self.required and not files:
            raise forms.ValidationError("No file was submitted. Check the encoding type on the form.")
        # Flatten in case any nested lists exist
        flattened = []
        for item in files:
            if isinstance(item, list):
                flattened.extend(item)
            else:
                flattened.append(item)
        return flattened

class ImageUploadForm(forms.Form):
    images = MultiFileField()

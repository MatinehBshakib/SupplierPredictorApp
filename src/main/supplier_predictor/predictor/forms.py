from django import forms

class ShipmentUploadForm(forms.Form):
    file = forms.FileField(
        label='Upload shipments file',
        help_text='Accepted formats: CSV, Excel'
    )
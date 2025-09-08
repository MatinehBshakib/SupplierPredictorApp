from django.db import models

class GeocodedLocation(models.Model):
    """
    A Django model to cache the results of geocoding lookups.
    
    Stores the latitude and longitude for a given postal code and country
    to avoid redundant API calls.
    """
    postal_code = models.CharField(max_length=20, default="UNKNOWN")
    country     = models.CharField(max_length=255)
    latitude    = models.FloatField()
    longitude   = models.FloatField()

    class Meta:
        unique_together = ('postal_code', 'country')

class PredictionResult(models.Model):
    """
    A Django model to store the results of a shipment prediction.

    Logs the input data and the predicted supplier/account, along with a
    timestamp for tracking and analysis.
    """
    shipment_id = models.CharField(max_length=255, primary_key=True)
    supplier_prediction = models.CharField(max_length=255)
    account_prediction = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
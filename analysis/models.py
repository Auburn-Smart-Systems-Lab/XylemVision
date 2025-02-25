from django.db import models

class ProcessedImage(models.Model):
    original_image = models.ImageField(upload_to='upload/')
    processed_image = models.ImageField(upload_to='upload/')
    vascular_bundle_image = models.ImageField(upload_to='upload/')
    total_root_image = models.ImageField(upload_to='upload/')
    xylem_image = models.ImageField(upload_to='upload/')

    vascular_area = models.FloatField()
    vascular_diameter = models.FloatField()
    xylem_count = models.IntegerField()
    xylem_diameter = models.FloatField()
    xylem_details = models.JSONField()

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Processed image {self.pk}"
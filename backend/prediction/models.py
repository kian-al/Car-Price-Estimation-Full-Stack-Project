from django.db import models

class CarPrediction(models.Model):
    brand = models.CharField(max_length=50)
    model_year = models.IntegerField()
    mileage = models.IntegerField()
    gearbox = models.CharField(max_length=20)
    fuel_type = models.CharField(max_length=20)
    city = models.CharField(max_length=50)

    predicted_price = models.BigIntegerField()

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.brand} - {self.predicted_price}"

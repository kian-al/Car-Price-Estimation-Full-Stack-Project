from django.db import models
from django.contrib.auth.models import User

class CarPrediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    
    # افزایش طول فیلدها برای جلوگیری از ارور
    brand = models.CharField(max_length=100)       # بود 50
    model_year = models.IntegerField()
    mileage = models.IntegerField()
    gearbox = models.CharField(max_length=100)     # بود 20 (مهم)
    fuel_type = models.CharField(max_length=100)   # بود 20 (مهم)
    
    body_condition = models.CharField(max_length=100, default='نامشخص')
    engine_condition = models.CharField(max_length=100, default='نامشخص')
    chassis_condition = models.CharField(max_length=100, default='نامشخص')

    city = models.CharField(max_length=100)        # بود 50
    predicted_price = models.BigIntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.brand} - {self.predicted_price}"
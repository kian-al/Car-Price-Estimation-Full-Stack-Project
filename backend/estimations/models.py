from django.db import models
from django.contrib.auth.models import User

class Estimation(models.Model):
 user = models.ForeignKey(User, on_delete=models.CASCADE)
 car_model = models.CharField(max_length=100)
 year = models.IntegerField()
 mileage = models.IntegerField()
 predicted_price = models.IntegerField()
 created_at = models.DateTimeField(auto_now_add=True)

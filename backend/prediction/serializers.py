from rest_framework import serializers
from .models import CarPrediction

class CarPredictionSerializer(serializers.ModelSerializer):
    # اتصال نام‌های ارسالی فرانت‌اند (حروف بزرگ) به فیلدهای دیتابیس (حروف کوچک)
    Brand = serializers.CharField(source='brand')
    Model_Year = serializers.IntegerField(source='model_year')
    Mileage = serializers.IntegerField(source='mileage')
    Gearbox = serializers.CharField(source='gearbox')
    Fuel_Type = serializers.CharField(source='fuel_type')
    Body_Condition = serializers.CharField(source='body_condition')
    Engine_Condition = serializers.CharField(source='engine_condition')
    Chassis_Condition = serializers.CharField(source='chassis_condition')
    City = serializers.CharField(source='city')

    class Meta:
        model = CarPrediction
        # فیلدهایی که در API رد و بدل می‌شوند
        fields = [
            'id', 
            'Brand', 'Model_Year', 'Mileage', 'Gearbox', 'Fuel_Type', 
            'Body_Condition', 'Engine_Condition', 'Chassis_Condition', 'City',
            'predicted_price', 'created_at'
        ]
        # این فیلدها فقط خواندنی هستند (توسط کاربر پر نمی‌شوند)
        read_only_fields = ['predicted_price', 'created_at']
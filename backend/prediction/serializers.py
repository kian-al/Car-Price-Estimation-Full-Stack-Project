from rest_framework import serializers

class CarPredictionSerializer(serializers.Serializer):
    Brand = serializers.CharField()
    Model_Year = serializers.IntegerField()
    Mileage = serializers.IntegerField()
    Gearbox = serializers.CharField()
    Fuel_Type = serializers.CharField()
    Body_Condition = serializers.CharField()
    Engine_Condition = serializers.CharField()
    Chassis_Condition = serializers.CharField()
    City = serializers.CharField()

from rest_framework import serializers
from .models import Estimation

class EstimationSerializer(serializers.ModelSerializer):
 class Meta:
  model = Estimation
  fields = '__all__'
  read_only_fields = ('user',)

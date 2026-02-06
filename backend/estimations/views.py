from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from prediction.models import CarPrediction
from prediction.serializers import CarPredictionSerializer

class UserEstimationsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        # فقط پیش‌بینی‌های مربوط به کاربر فعلی را بگیر
        estimations = CarPrediction.objects.filter(user=request.user).order_by('-created_at')
        serializer = CarPredictionSerializer(estimations, many=True)
        return Response(serializer.data)
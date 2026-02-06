from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.authentication import BasicAuthentication

from .serializers import CarPredictionSerializer
from .ml_model import predict_price
from .models import CarPrediction


class PredictCarPriceAPIView(APIView):
    authentication_classes = [BasicAuthentication]
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = CarPredictionSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(
                serializer.errors,
                status=status.HTTP_400_BAD_REQUEST
            )

        data = serializer.validated_data

        try:
            # پیش‌بینی قیمت
            predicted_price = predict_price(data)
        except Exception as e:
            return Response(
                {"error": f"ML model error: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # ذخیره کامل دیتا در دیتابیس
        car_prediction = CarPrediction.objects.create(
            brand=data["Brand"],
            model_year=data["Model_Year"],
            mileage=data["Mileage"],
            gearbox=data["Gearbox"],
            fuel_type=data["Fuel_Type"],
            body_condition=data["Body_Condition"],
            engine_condition=data["Engine_Condition"],
            chassis_condition=data["Chassis_Condition"],
            city=data["City"],
            predicted_price=predicted_price
        )

        return Response(
            {
                "id": car_prediction.id,
                "predicted_price": predicted_price
            },
            status=status.HTTP_200_OK
        )

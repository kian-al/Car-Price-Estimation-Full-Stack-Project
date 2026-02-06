from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated

from .serializers import CarPredictionSerializer
from .ml_model import predict_price
from .models import CarPrediction

class PredictCarPriceAPIView(APIView):
    # فقط کاربران لاگین شده اجازه دسترسی دارند
    permission_classes = [IsAuthenticated] 

    def post(self, request):
        serializer = CarPredictionSerializer(data=request.data)

        if not serializer.is_valid():
            # نمایش دقیق ارور برای دیباگ
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        # دریافت داده‌های تمیز شده (با حروف کوچک ذخیره شده‌اند)
        data = serializer.validated_data

        # آماده‌سازی داده برای مدل هوش مصنوعی (اگر مدل نیاز به کلیدهای حروف بزرگ دارد)
        ml_input = {
            "Brand": data['brand'],
            "Model_Year": data['model_year'],
            "Mileage": data['mileage'],
            "Gearbox": data['gearbox'],
            "Fuel_Type": data['fuel_type'],
            "Body_Condition": data['body_condition'],
            "Engine_Condition": data['engine_condition'],
            "Chassis_Condition": data['chassis_condition'],
            "City": data['city']
        }

        try:
            # محاسبه قیمت
            predicted_price = predict_price(ml_input)
        except Exception as e:
            return Response({"error": f"ML model error: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # ذخیره در دیتابیس به همراه کاربر و قیمت محاسبه شده
        serializer.save(user=request.user, predicted_price=predicted_price)

        return Response(
            {
                "id": serializer.instance.id,
                "predicted_price": predicted_price
            },
            status=status.HTTP_200_OK
        )
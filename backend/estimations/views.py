from rest_framework.viewsets import ModelViewSet
from rest_framework.permissions import IsAuthenticated

from .models import Estimation
from .serializers import EstimationSerializer


class EstimationViewSet(ModelViewSet):
    queryset = Estimation.objects.all()
    serializer_class = EstimationSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Estimation.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(
            user=self.request.user,
            predicted_price=0
        )

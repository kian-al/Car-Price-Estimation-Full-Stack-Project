from rest_framework.routers import DefaultRouter
from .views import EstimationViewSet

router = DefaultRouter()
router.register('', EstimationViewSet)

urlpatterns = router.urls

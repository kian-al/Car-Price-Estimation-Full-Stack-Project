# backend/estimations/urls.py
from django.urls import path
from .views import UserEstimationsView

urlpatterns = [
    path('', UserEstimationsView.as_view()), # چون در backend/urls.py پیشوند api/estimations/ دارد
]
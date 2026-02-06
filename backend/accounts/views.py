from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from rest_framework_simplejwt.tokens import RefreshToken

class LoginView(APIView):
 permission_classes = [AllowAny]

 def post(self, request):
  user = authenticate(
   username=request.data['email'],
   password=request.data['password']
  )
  if not user:
   return Response({'message': 'Invalid credentials'}, status=401)

  token = RefreshToken.for_user(user)
  return Response({'token': str(token.access_token)})

class RegisterView(APIView):
 permission_classes = [AllowAny]

 def post(self, request):
  User.objects.create_user(
   username=request.data['email'],
   email=request.data['email'],
   password=request.data['password']
  )
  return Response({'message': 'created'})

from django.shortcuts import render
from rest_framework import generics, mixins
from rest_framework.permissions import IsAuthenticated, AllowAny, BasePermission
from rest_framework.viewsets import GenericViewSet
from .serializers import UserSerializer, TransactionSerializer, RegisterSerializer
from .models import Transaction, User
from rest_framework import viewsets
from django.utils import timezone
from rest_framework import serializers
import uuid
from django.db import IntegrityError
from rest_framework import status
from .ml_model import generate_recommendations  
import pandas as pd

@api_view(['GET'])
def get_recommendations(request):
    transactions = Transaction.objects.all().values('amount', 'category', 'description', 'date')
    df = pd.DataFrame(transactions)
    
    recommendations = generate_recommendations(df)
    
    return Response({'recommendations': recommendations})

class IsOwner(BasePermission):
    def has_object_permission(self, request, view, obj):
        return obj.user == request.user

class UserProfileView(generics.RetrieveUpdateAPIView):
    queryset = User.objects.all()
    permission_classes = [IsAuthenticated, IsOwner]
    serializer_class = UserSerializer
    lookup_field = 'id'


class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    permission_classes = [AllowAny]
    serializer_class = RegisterSerializer


class TransactionViewSet(viewsets.ModelViewSet):
    queryset = Transaction.objects.all()
    serializer_class = TransactionSerializer
    permission_classes = [AllowAny]

    def perform_create(self, serializer):
        try:
            serializer.save(
                user=self.request.user,
                ref_no=str(uuid.uuid4())  
            )
        except IntegrityError:
            return Response(
                {"error": "Не удалось сохранить транзакцию. Возможно, дубликат ref_no."},
                status=status.HTTP_400_BAD_REQUEST
            )
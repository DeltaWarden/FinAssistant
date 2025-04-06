from rest_framework import serializers
from .models import User, Transaction
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from django.contrib.auth.password_validation import validate_password
from rest_framework.permissions import AllowAny


class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(
        write_only=True, required=True, validators=[validate_password]
    )
    password2 = serializers.CharField(write_only=True, required=True)

    class Meta:
        model = User
        fields = ('email', 'username', 'password', 'password2')

    def validate(self, attrs):
        if attrs['password'] != attrs['password2']:
            raise serializers.ValidationError(
                {"password": "Пароли не совпадают"}
            )

        return attrs
    
    def create(self, validated_data):
        user = User.objects.create(
            username=validated_data['username'],
            email=validated_data['email']
        )

        user.set_password(validated_data['password'])
        user.save()

        return user

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'first_name', 'last_name')
        read_only_fields = ('id',)

class MyTokenObtainPairSerializer(TokenObtainPairSerializer):
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)
        token['id'] = user.id
        token['username'] = user.username
        return token

from decimal import Decimal

class TransactionSerializer(serializers.ModelSerializer):
    permission_classes = [AllowAny]
    ref_no = serializers.UUIDField(format='hex_verbose', read_only=True) 
    operation_type = serializers.CharField()  
    user = serializers.PrimaryKeyRelatedField(queryset=User.objects.all()) 
    
    class Meta:
        model = Transaction
        fields = '__all__'

    from decimal import Decimal

    def validate(self, data):
        withdrawal_amount = data.get('withdrawal_amount', 0)
        deposit_amount = data.get('deposit_amount', 0)
    
        
        if withdrawal_amount < 0 or deposit_amount < 0:
            raise serializers.ValidationError("Суммы не могут быть отрицательными")
        
        
        if withdrawal_amount == 0 and deposit_amount == 0:
            raise serializers.ValidationError("Не указана сумма для операции (вывод или депозит)")
    
        
        data['withdrawal_amount'] = Decimal(str(withdrawal_amount))
        data['deposit_amount'] = Decimal(str(deposit_amount))
        
        return data
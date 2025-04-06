from django.urls import path, include
from . import views
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
    TokenVerifyView,
)
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register(r'transactions', views.TransactionViewSet, basename='transaction')

urlpatterns = [
    path('recommendations/', views.get_recommendations, name='get_recommendations'),
    path('token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('token/verify/', TokenVerifyView.as_view(), name='token_verify'),
    path('user/<int:id>', views.UserProfileView.as_view(), name='user_details'),
    path('register/', views.RegisterView.as_view(), name='register'),
    path('', include(router.urls)),
]
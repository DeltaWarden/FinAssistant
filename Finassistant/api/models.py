from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils.translation import gettext_lazy as _
from django.utils import timezone
import uuid
from django.contrib.auth.models import User

class User(AbstractUser):
    username = models.CharField(
        max_length=100,
        help_text=_("Имя пользователя для отображения в системе")
    )
    email = models.EmailField(
        unique=True,
        db_index=True,
        help_text=_("Email пользователя, используется для входа"),
        error_messages={
            'unique': "Пользователь с таким email уже существует"
        }
    )

    
    groups = models.ManyToManyField(
        'auth.Group',
        related_name='custom_user_set',
        blank=True,
        help_text=_('The groups this user belongs to.'),
        verbose_name=_('groups'),
    )
    user_permissions = models.ManyToManyField(
        'auth.Permission',
        related_name='custom_user_set',
        blank=True,
        help_text=_('Specific permissions for this user.'),
        verbose_name=_('user permissions'),
    )

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']

    class Meta:
        verbose_name = _("пользователь")
        verbose_name_plural = _("пользователи")

    @property
    def full_name(self):
        return f'{self.first_name} {self.last_name}'

    def __str__(self) -> str:
        return self.full_name

OPERATION_TYPE_CHOICES = [
    ('income', 'Income'),
    ('expense', 'Expense'),
    ]

class Transaction(models.Model):
    description = models.TextField(blank=True, null=True)
    ref_no = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    date = models.DateTimeField(auto_now_add=True)
    category = models.CharField(max_length=255)
    operation_type = models.CharField(max_length=50)
    balance = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    deposit_amount = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    withdrawal_amount = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    def __str__(self):
        return f"{self.user.username} - {self.ref_no}"
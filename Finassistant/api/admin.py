from django.contrib import admin
from .models import User, Transaction

class UserAdmin(admin.ModelAdmin):
    search_fields = ['username', 'email', 'full_name']
    list_display = ['id', 'username', 'full_name', 'email']
    list_display_links = ['id', 'username', 'full_name']

class TransactionAdmin(admin.ModelAdmin):
    list_display = ['ref_no', 'user', 'date', 'category', 'operation_type', 'balance']
    list_display_links = ['ref_no', 'user', 'date', 'category', 'operation_type', 'balance']

admin.site.register(User, UserAdmin)
admin.site.register(Transaction, TransactionAdmin)

from django.contrib import admin

# Register your models here.
from app.models import Alarm, Stream, Control

admin.site.register(Alarm)
admin.site.register(Stream)
admin.site.register(Control)

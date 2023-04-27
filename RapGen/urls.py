from django.urls import path
from .views import home, run_generate_rap, index, indexrap
from django.conf.urls import include
from django.shortcuts import redirect
# app_name = 'RapGen'
urlpatterns = [
    #path('', lambda req: redirect('/')),
    path('', index),
    #path('run-generate-rap/', indexrap),
    path('home/', home, name='home'),
    path('run-generate-rap/', run_generate_rap, name='run-generate-rap')
]

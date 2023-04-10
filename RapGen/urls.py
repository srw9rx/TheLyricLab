from django.urls import path
from .views import home, run_generate_rap

urlpatterns = [
    path('home/', home, name='home'),
    path('run-generate-rap/', run_generate_rap, name='run-generate-rap')
]

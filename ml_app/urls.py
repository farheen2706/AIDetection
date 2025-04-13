"""project_settings URL Configuration
"""
from django.contrib import admin
from django.urls import path, include
from . import views
from .views import about, index, predict_page,cuda_full

app_name = 'ml_app'
handler404 = views.handler404

urlpatterns = [
    path('', views.landingpage, name='landingpage'),
    path('register/', views.register, name='register'),
    path('login/', views.login, name='login'),
    path('logout/', views.logoutusers, name='logout'),
    path('index/', index, name='home'),
    path('about/', about, name='about'),
    path('model/',views.model,name='model'),
    path('model_db',views.model_db,name='model_db'),
    path('video_db',views.video_db,name='video_db'),
    path('videod',views.videod,name='videod'),
    path('predict/', predict_page, name='predict'),
    path('modell/',views.modell,name='modell'),
    path('Database/',views.Database,name='Database'),
    path('cuda_full/',cuda_full,name='cuda_full'),
]

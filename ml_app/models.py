from django.db import models

# Create your models here.
from django.db import models

class AudioPrediction(models.Model):
    id = models.BigAutoField(primary_key=True)
    audio_file = models.FileField(upload_to='audio_files/')
    prediction = models.CharField(max_length=50)
    upload_time = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.prediction
    
    
from django.db import models

class VideoPrediction(models.Model):
    id = models.BigAutoField(primary_key=True)
    video_file = models.FileField(upload_to='uploaded_videos/')
    prediction_output = models.CharField(max_length=10)  # E.g., 'REAL' or 'FAKE'
    confidence = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.video_file.name} - {self.prediction_output} ({self.confidence})"
    
from PIL import Image  
class UserImageModel(models.Model):
    image = models.ImageField(upload_to = 'images/',blank=True)
    label = models.CharField(max_length=20,default='data')
    def __str__(self):
        return str(self.image)





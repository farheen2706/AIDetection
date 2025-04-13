from django import forms

class VideoUploadForm(forms.Form):

    upload_video_file = forms.FileField(label="Select Video", required=True,widget=forms.FileInput(attrs={"accept": "video/*"}))
    sequence_length = forms.IntegerField(label="Sequence Length", required=True)
    
    
    
    
from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.forms import ModelForm


class CreateUserForm(UserCreationForm):
    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']
        
        
from django import forms
from .models import AudioPrediction

class AudioForm(forms.ModelForm):
    class Meta:
        model = AudioPrediction
        fields = ['audio_file']
        
        
from . models import UserImageModel
class UserImageForm(forms.ModelForm):

    class Meta:
        model = UserImageModel
        fields = ['image']


from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import os

app = FastAPI()

# Load the model
model = load_model("C:/Users/kvrus/Downloads/Kiosk_Model1.h5")
scaler = StandardScaler()

# Define audio processing functions
def add_noise(data):
    noise_value = 0.015 * np.random.uniform() * np.amax(data)
    data = data + noise_value * np.random.normal(size=data.shape[0])
    return data

def stretch_process(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=0.8)

def pitch_process(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

def extract_process(data, sample_rate):
    output_result = np.array([])
    mean_zero = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    output_result = np.hstack((output_result, mean_zero))
    stft_out = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft_out, sr=sample_rate).T, axis=0)
    output_result = np.hstack((output_result, chroma_stft))
    mfcc_out = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40).T, axis=0)
    output_result = np.hstack((output_result, mfcc_out))
    root_mean_out = np.mean(librosa.feature.rms(y=data).T, axis=0)
    output_result = np.hstack((output_result, root_mean_out))
    mel_spectogram = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    output_result = np.hstack((output_result, mel_spectogram))
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=data, sr=sample_rate).T, axis=0)
    output_result = np.hstack((output_result, spectral_contrast))
    return output_result

def preprocess_audio(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast', duration=42, offset=0.6)
    features = extract_process(audio, sample_rate)
    noise_out = add_noise(audio)
    features = np.vstack((features, extract_process(noise_out, sample_rate)))
    stretch_pitch = pitch_process(stretch_process(audio, 0.8), sample_rate, pitch_factor=0.7)
    features = np.vstack((features, extract_process(stretch_pitch, sample_rate)))
    return features

def prepare_features_for_model(file_path):
    features = preprocess_audio(file_path)
    features = features.reshape(-1, 189)
    features = scaler.fit_transform(features)
    features = np.expand_dims(features, axis=2)
    return features

@app.get("/predict_local/")
async def predict_local():
    try:
        # Replace with your actual file path
        file_path = "C:/Users/kvrus/Downloads/Kisok Lung/archive/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/223_1b1_Pl_sc_Meditron.wav"
        
        features = prepare_features_for_model(file_path)
        prediction = model.predict(features)
        pred_label = np.argmax(np.mean(prediction, axis=0))
        
        labels = ['Bronchiectasis', 'Bronchiolitis', 'COPD', 'Healthy', 'Pneumonia', 'URTI']
        result = labels[pred_label]
        output_file = "Kiosk_Outputs\\result.txt"
        with open(output_file,"w+") as f:
            f.write(f"classification,{result}")
        return {"classification": result}
    except Exception as e:
        return {"error": str(e)}

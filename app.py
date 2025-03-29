import streamlit as st
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import load_model
from keras.utils import to_categorical

# Load the trained model
model = load_model("sound_model.h5")

# Function to extract features
def extract_features(data, sample_rate):
    result = np.array([])
    
    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    # Chroma Stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40).T, axis=0)
    result = np.hstack((result, mfcc))

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    # MelSpectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))
    
    return result

# Streamlit UI
st.title("🎵 Speech Emotion Recognition App")
st.write("Upload a `.wav` audio file to predict the emotion.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    # Load the audio file
    data, sample_rate = librosa.load(uploaded_file, duration=2.5, offset=0.6)
    
    # Extract features
    features = extract_features(data, sample_rate)
    features = np.expand_dims(features, axis=0)  # Reshape for model input

    # Make prediction
    predictions = model.predict(features)
    predicted_label = np.argmax(predictions)

    # Emotion categories (update based on your dataset)
    emotions = ["angry", "calm", "fear", "happy", "neutral", "sad", "surprise"]

    st.write(f"**Predicted Emotion: {emotions[predicted_label].upper()}**")

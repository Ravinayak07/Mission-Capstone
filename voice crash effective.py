import librosa
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os

# Load the trained model
model = load_model("model/fallback_model.keras")

# Define function to convert audio to spectrogram image
def create_spectrogram(audio_path, temp_img_path="temp_spectrogram.png"):
    wave, sr = librosa.load(audio_path, duration=3.0)
    augmented_wave = wave + 0.005 * np.random.randn(len(wave))  # same noise as training
    plt.figure(figsize=(3, 3))
    plt.specgram(augmented_wave, Fs=sr)
    plt.axis('off')
    plt.savefig(temp_img_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return temp_img_path

# Function to predict
def predict_audio(audio_path, img_size=(128, 128)):
    # Step 1: Convert audio to spectrogram
    img_path = create_spectrogram(audio_path)

    # Step 2: Load and preprocess image
    img = load_img(img_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 128, 128, 3)

    # Step 3: Make prediction
    prediction = model.predict(img_array)
    classes = ["Female", "Male", "Noizeus"]
    predicted_class = classes[np.argmax(prediction)]

    return predicted_class, prediction[0]

# Example usage:
audio_file = "sample.wav"  # Replace with your audio file path
predicted_class, probabilities = predict_audio(audio_file)

print("Predicted Class:", predicted_class)
print("Class Probabilities:", dict(zip(["Female", "Male", "Noizeus"], probabilities)))

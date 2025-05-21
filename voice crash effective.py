# ----------- 1_data_prep.py -----------
# Run this only once to generate spectrograms

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Data Augmentation

def add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y)) * noise_factor
    return y + noise

# Paths
AUDIO_PATH = 'Data/Audio'
OUTPUT_PATH = 'spectrogram'
CATEGORIES = ["Female", "Male", "Noizeus"]

for dirpath, _, filenames in os.walk(AUDIO_PATH):
    for category in CATEGORIES:
        if category in dirpath:
            relative_path = dirpath.split("Audio/")[-1]
            save_dir = os.path.join(OUTPUT_PATH, relative_path)
            os.makedirs(save_dir, exist_ok=True)
            for filename in filenames:
                if filename.lower().endswith(('.wav', '.mp3')):
                    file_path = os.path.join(dirpath, filename)
                    wave, sr = librosa.load(file_path, duration=3.0)
                    augmented_wave = add_noise(wave)
                    plt.figure(figsize=(3, 3))
                    plt.specgram(augmented_wave, Fs=sr)
                    plt.axis('off')
                    base_name = os.path.splitext(filename)[0]
                    save_path = os.path.join(save_dir, base_name + ".png")
                    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                    plt.close()

print("✅ Spectrograms generated and saved successfully.")


# ----------- 2_train_model.py -----------

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras import Sequential, Model, layers, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, BatchNormalization

# Directories
SPEC_DIR = "spectrogram"
CHECKPOINT_DIR = "model/ckpt"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Image settings
IMG_SIZE = (128, 128)
BATCH_SIZE = 8

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    SPEC_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    classes=["Female", "Male", "Noizeus"]
)

val_generator = train_datagen.flow_from_directory(
    SPEC_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=True
)

# Encoder architecture
def get_encoder(input_size):
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), input_shape=input_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(2, 2))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Dropout(0.4))

    model.add(layers.GlobalMaxPool2D())
    return model

input_shape = train_generator.image_shape
encoder = get_encoder(input_shape)
input_layer = Input(input_shape)
encoded = encoder(input_layer)
output_layer = layers.Dense(3, activation='softmax')(encoded)
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Callbacks
checkpoint_path = os.path.join(CHECKPOINT_DIR, 'checkpoint.model.keras')
model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(patience=5, min_delta=0.001, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# Training
try:
    history = model.fit(
        train_generator,
        epochs=30,
        validation_data=val_generator,
        callbacks=[model_checkpoint, early_stopping, reduce_lr],
        workers=2,
        use_multiprocessing=False
    )
except Exception as e:
    print("❌ Training crashed:", e)
    model.save("model/fallback_model.keras")

# Save model and evaluation
model.save("model/final_voice_model.keras")
val_loss, val_acc = model.evaluate(val_generator, batch_size=BATCH_SIZE)
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Validation Loss: {val_loss:.4f}")

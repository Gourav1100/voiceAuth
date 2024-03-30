import os
import pyaudio
import wave
import numpy as np
import tensorflow as tf
import librosa
import librosa.display
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def record_audio(output_folder, person_name, num_recordings=16, duration=2, sample_rate=44100, channels=2, format=pyaudio.paInt16):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i in range(num_recordings):
        audio = pyaudio.PyAudio()
        filename = os.path.join(output_folder, f"{person_name}_{i+1}.wav")
        if num_recordings == 1:
            filename = os.path.join(output_folder, f"{person_name}.wav")
        print(f"Recording {i+1}/{num_recordings}...")
        stream = audio.open(format=format,
                             channels=channels,
                             rate=sample_rate,
                             input=True,
                             frames_per_buffer=1024)
        frames = []
        for _ in range(0, int(sample_rate / 1024 * duration)):
            data = stream.read(1024)
            frames.append(data)
        stream.stop_stream()
        stream.close()
        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        print(f"Recording saved as {filename}")
        audio.terminate()

def load_audio(parent_dir, username = None):
    audio_files = []
    labels = []
    directories = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    for directory in directories:
        is_auth_user = True if directory == username else False
        _files = [os.path.join(parent_dir, directory, f) for f in os.listdir(os.path.join(parent_dir, directory)) if f.endswith(".wav")]
        for audio_file in _files:
            y, sr = librosa.load(audio_file, sr=None)  # Load audio file
            mfccs = librosa.feature.mfcc(y=y, sr=sr)  # Extract MFCC features
            audio_files.append(mfccs.T)  # Transpose MFCCs to have shape (time_steps, num_mfcc)
            labels.append(directory if is_auth_user else "other")
    _l = labels
    label_set = list(set(labels))
    label_to_index = {label: i for i, label in enumerate(label_set)}
    labels = [label_to_index[label] for label in labels]
    labels = to_categorical(labels, num_classes=len(label_set))
    return np.array(audio_files), np.array(labels), _l


def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        
        layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        
        layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        
        layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        
        layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dense(4096, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def register(name):
    if name != "":
        _store_name = f"recordings/{name}"
        record_audio(_store_name, name)

def login(name):
    batch_size = 16
    epochs = 100
    _parent_dir = "recordings"
    auth_username = name
    if auth_username not in os.listdir("recordings"):
        raise Exception("Invalid username")
    _audio_files, _labels, _l = load_audio(_parent_dir, auth_username)
    X_train, X_test, y_train, y_test = train_test_split(_audio_files, _labels, test_size=0.25, random_state= 42)
    input_shape = (X_train.shape[1], X_train.shape[2], 1)  # Shape of input data for Conv2D layers
    num_classes = len(set(_l))
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
    model = build_model(input_shape, num_classes)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))
    record_audio("testing", "auth", 1)
    _audio, _, __ = load_audio(".")
    _audio = _audio.reshape((1, _audio.shape[1], _audio.shape[2], 1))
    res = model.predict(_audio).tolist()
    _options = list(set(_l))
    _predicted = res[0].index(max(res[0]))
    print(_options)
    print(res[0])
    if _options[_predicted] == auth_username:
        print("Authenticated")
    else:
        print("Not Authenticated")
        
if __name__ == "__main__":
    login("sinchan")
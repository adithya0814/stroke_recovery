import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import os

# Load the saved landmark sequences
def load_landmark_data(data_dir='landmark_sequences'):
    all_data = []
    for file in os.listdir(data_dir):
        if file.endswith('.npy'):
            path = os.path.join(data_dir, file)
            landmarks = np.load(path)
            all_data.append(landmarks)
    return np.array(all_data)

# Load data
landmark_sequences = load_landmark_data()

# Example: Assuming the last 20 frames are labels for recovery prediction
# Labels should be based on user feedback or clinical assessment
labels = np.array([1, 0, 1, 0])  # 1 for recovered, 0 for not recovered (example)

# Preprocessing
X = landmark_sequences  # Shape: (num_samples, num_frames, num_landmarks, 3)
X = X.reshape(X.shape[0], X.shape[1], -1)  # Flatten the 3D landmarks for CNN input
y = labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN+LSTM model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save('../recovery_cnn_lstm_model.h5')

import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('models/recovery_cnn_lstm_model.h5')

def predict_recovery(landmark_sequence):
    # Preprocess the input sequence (flatten landmarks)
    landmark_sequence = np.array(landmark_sequence)
    landmark_sequence = landmark_sequence.reshape(1, landmark_sequence.shape[0], -1)
    
    # Predict recovery
    prediction = model.predict(landmark_sequence)
    return prediction[0][0]  # Recovery score between 0 and 1

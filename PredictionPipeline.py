import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = tf.keras.models.load_model("/Users/kavyasmac/Desktop/Time Pass/DeepFakeDetection/DeepfakeDetection/Models/Xception_LSTM/xception_lstm.keras",compile=False)  
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


def extract_frames(video_path, num_frames=30):
    """Extracts evenly spaced frames from the video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // num_frames)

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (299, 299))
        frame = img_to_array(frame) / 255.0  # Normalize pixels
        frames.append(frame)

    cap.release()

    if len(frames) < num_frames:
        print("Warning: Not enough frames extracted!")
        return None

    return np.array(frames)

def predict_deepfake(video_path):
    """Predict if a given video is real or fake."""
    frames = extract_frames(video_path)
    if frames is None:
        print("Could not process the video correctly.")
        return

    frames = np.expand_dims(frames, axis=0)  # Add batch dimension
    prediction = model.predict(frames)

    if prediction[0][0] > 0.5:
        print(f"Prediction: 🚨 Fake Video (Confidence: {prediction[0][0]:.2f})")
    else:
        print(f"Prediction: ✅ Real Video (Confidence: {1 - prediction[0][0]:.2f})")

# Example usage
video_path = "/Users/kavyasmac/Desktop/Time Pass/DeepFakeDetection/Data/ModelData/Test/fake/02_07__walking_down_street_outside_angry__O4SXNLRL.mp4"  # Update with your video path
predict_deepfake(video_path)

import cv2
import os
from mtcnn import MTCNN
import imageio

detector = MTCNN()

def extract_faces(video_path, output_folder, output_video_path, frame_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        faces = detector.detect_faces(frame_rgb)  # Detect faces
        
        if faces:
            x, y, width, height = faces[0]['box']  # Extract face bounding box
            x, y = max(0, x), max(0, y)  # Ensure coordinates are positive
            face_crop = frame_rgb[y:y+height, x:x+width]  # Crop the face
            
            if face_crop.size > 0:
                face_resized = cv2.resize(face_crop, frame_size)  # Resize face
                frames.append(face_resized)  # Store processed frame

    cap.release()

    if frames:
        # Save frames as video
        output_path = os.path.join(output_folder, output_video_path)
        imageio.mimwrite(output_path, frames, fps=30)
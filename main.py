import cv2
import numpy as np
import os
from datetime import datetime
import pandas as pd

# Path to the directory containing face images
FACES_DIR = 'faces'
ATTENDANCE_FILE = 'attendance.csv'

def get_face_detector():
    # Use OpenCV's pre-trained Haar Cascade classifier
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("Error: Could not load Haar Cascade Classifier XML.")
        # Fallback for some systems or custom paths if needed, but usually cv2.data.haarcascades works
    return face_cascade

def train_model(path):
    print("Training LBPH model from known faces...")
    
    # Check if faces dir exists
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}. please add images using capture_faces.py")
        return None, {}, []

    # Initialize LBPH Face Recognizer
    # Note: This requires opencv-contrib-python
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = get_face_detector()
    
    face_samples = []
    ids = []
    names = {} # Map ID (int) -> Name (str)
    
    # We need numeric IDs for LBPH, so we'll assign an arbitrary ID to each unique name
    current_id = 0
    name_to_id = {}

    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_paths:
        print("No images found in 'faces' directory.")
        return None, {}, []

    for image_path in image_paths:
        # Read image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        # Extract name from filename
        filename = os.path.split(image_path)[-1]
        name = os.path.splitext(filename)[0]
        
        # Handle IDs
        if name not in name_to_id:
            name_to_id[name] = current_id
            names[current_id] = name
            current_id += 1
        
        label_id = name_to_id[name]
        
        # Detect face in the training image to ensure we train on just the face
        faces = detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            face_samples.append(img[y:y+h, x:x+w])
            ids.append(label_id)
            # We assume one face per training image for simplicity as per capture_faces.py convention
            break 
            
    if not face_samples:
        print("No faces found in the training images.")
        return None, {}, []

    recognizer.train(face_samples, np.array(ids))
    print(f"Model trained with {len(np.unique(ids))} unique people.")
    return recognizer, names, ids

def mark_attendance(name):
    # Check if file exists, if not create with header
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'w') as f:
            f.write('Name,Time,Date\n')
            
    # Read existing attendance to avoid duplicates for the day
    try:
        df = pd.read_csv(ATTENDANCE_FILE)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=['Name', 'Time', 'Date'])
    
    now = datetime.now()
    time_string = now.strftime('%H:%M:%S')
    date_string = now.strftime('%Y-%m-%d')
    
    # Check if user already marked for today
    if not df.empty:
        today_records = df[(df['Name'] == name) & (df['Date'] == date_string)]
        if not today_records.empty:
            return # Already marked

    with open(ATTENDANCE_FILE, 'a') as f:
        f.write(f'{name},{time_string},{date_string}\n')
        print(f"Attendance marked for {name}")

def main():
    recognizer, names, _ = train_model(FACES_DIR)
    detector = get_face_detector()
    
    if recognizer is None:
        print("Could not train model. Taking you to capture mode? No, just exiting.")
        print("Please run capture_faces.py first to add some faces.")
        # We allow running anyway to show the camera, but recognition won't work
    
    cap = cv2.VideoCapture(0)
    
    print("Starting Webcam...")
    print("Press 'q' to quit")
    
    # Minimum confidence to consider a match. 
    # In LBPH, LOWER distance means BETTER match.
    # 0 is a perfect match. 100+ is usually no match.
    # A good threshold is around 40-60 depending on lighting.
    CONFIDENCE_THRESHOLD = 60 
    
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        for (x, y, w, h) in faces:
            name = "Unknown"
            confidence_text = "  ?  "
            
            if recognizer:
                try:
                    # predict returns (label_id, confidence)
                    id_num, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                    
                    # Convert confidence to a percentage-like score for display (optional)
                    # For LBPH: 0 is perfect match.
                    if confidence < 100:
                        if confidence < CONFIDENCE_THRESHOLD:
                            name = names.get(id_num, "Unknown")
                        else:
                            name = "Unknown"
                        confidence_text = f" {round(100 - confidence)}%"
                    else:
                        name = "Unknown"
                        confidence_text = "  0%"
                except Exception as e:
                    pass

            if name != "Unknown":
                mark_attendance(name)
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            
            # Draw box
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(img, (x, y-35), (x+w, y), color, cv2.FILLED)
            cv2.putText(img, name, (x+6, y-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            # cv2.putText(img, str(confidence), (x+5, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 1)
            
        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

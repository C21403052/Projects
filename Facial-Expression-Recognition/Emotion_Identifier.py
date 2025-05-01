# Import Libraries
import cv2
import numpy as np
import psycopg2
import torch
from CNN_Model import create_CNN_model
from datetime import datetime

# Database Connections
DB_params = {
    "dbname": "Emotions",
    "user": "Admin",
    "password": "password1",
    "host": "34.147.132.176",
    "port": "5432"
}

# Connect to Database
def connect_to_db():
    return psycopg2.connect(**DB_params)

# Get the new user and insert it into users table
def get_or_create_user(username = "Anonymous"):
    conn = connect_to_db()
    cur = conn.cursor()
    cur.execute("INSERT INTO users (name) VALUES (%s) RETURNING user_id;", (username,))
    user_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return user_id

# Start the session
def start_session(user_id):
    conn = connect_to_db()
    cur = conn.cursor()
    cur.execute("INSERT INTO sessions (user_id, start_time) VALUES (%s, %s) RETURNING session_id;", (user_id, datetime.now()))
    session_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return session_id

# Stop the session
def stop_session(session_id):
    conn = connect_to_db()
    cur = conn.cursor()
    cur.execute("UPDATE sessions SET end_time = %s WHERE session_id = %s;", (datetime.now(), session_id))
    conn.commit()
    cur.close()
    conn.close()

# Store Emotions on the average intensities of frames
def store_emotion(session_id, buffer):

    if not buffer:
        return

    avg_intensities = np.mean(buffer, axis=0).tolist()
    conn = connect_to_db()
    cur = conn.cursor()
    cur.execute("INSERT INTO emotion_records (session_id, timestamp, happy, neutral, surprised, disgust, anger, fear, sad) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);", (session_id, datetime.now(), *avg_intensities))
    conn.commit()
    cur.close()
    conn.close()

# Function to run model on camera and identify emotion intensities
def emotion_identifier():
    # load the trained model
    num_classes = 7
    model = create_CNN_model(num_classes)
    model.load_state_dict(torch.load("BackUp Files/Best_FER_Model.pth"))
    # Set model to evaluate
    model.eval()

    # Turn on device camera
    cap = cv2.VideoCapture(0)

    # Load the Haar cascade for face detection
    face_detection = cv2.CascadeClassifier(r"/FACIAL_EXPRESSION_RECOGNITION_APP\haarcascade_frontalface_default.xml")

    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    name = input("Enter Session Name:")
    user_id = get_or_create_user(name)
    session_id = start_session(user_id)
    print(f"Session {session_id} started")

    buffer = []
    last_save_time = datetime.now()

    # While loop to capture frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the gray frame
        faces = face_detection.detectMultiScale(gray, 1.3, 5)

        intensity_window = np.zeros((350, 450, 3), dtype=np.uint8)

        for (x, y, w, h) in faces:
            # Draw bounding box around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (140, 240, 140), 2)

            # Crop to the bounding box
            face = gray[y:y+h, x:x+w]

            # Resize frame
            face_resize = cv2.resize(face, (48, 48))


            # Normalize frame
            face_normalized = face_resize /255.0

            # face_normalized is shape [48, 48]
            face_tensor = torch.tensor(face_normalized, dtype=torch.float32)

            # Add one dimension for 'channels' and another for 'batch':
            # So final shape is [1, 1, 48, 48].
            face_tensor = face_tensor.unsqueeze(0).unsqueeze(0)

            # Disable gradient computation (Reduce memory usage)
            with torch.no_grad():
                # Get prediction with the image
                prediction = model(face_tensor)
                intensities = torch.softmax(prediction, dim=1).squeeze().tolist()

            # Pick the top emotion
            label_idx = torch.argmax(prediction, dim=1).item()
            emotion = emotions[label_idx]
            intensity = intensities[label_idx]

            # Draw on the image
            cv2.putText(frame, f"{emotion}: {intensity:.2f}",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

            # Draw a horizontal bar for each emotion
            bar_height = 30
            max_bar_width = 300
            start_y = 40

            for i, emotion_name in enumerate(emotions):
                prob = intensities[i]
                bar_width = int(prob * max_bar_width)

                top_left = (20, start_y + i * 40)
                bottom_right = (20 + bar_width, start_y + i * 40 + bar_height)
                cv2.rectangle(intensity_window, top_left, bottom_right, (0, 255, 0), -1)

                text = f"{emotion_name}: {prob:.2f}"
                cv2.putText(
                    intensity_window, text,
                    (25, start_y + i * 40 + int(bar_height * 0.7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2
                )

            buffer.append(intensities)

        # Show Frames
        cv2.imshow('FER', frame)
        cv2.imshow("Emotion Intensities", intensity_window)

        # Create buffer to gather emotion intensities in a certain timeframe
        if (datetime.now() - last_save_time).total_seconds() >= 2:
            store_emotion(session_id, buffer)
            buffer = []
            last_save_time = datetime.now()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stop_session(session_id)
    print(f"Session {session_id} stopped")
    cap.release()
    cv2.destroyAllWindows()
    
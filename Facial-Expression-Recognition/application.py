import os
import signal
import threading
import time

import cv2
import numpy as np
import psycopg2
import torch
from django.utils.datetime_safe import datetime
from flask import Flask, request, render_template, redirect, url_for, jsonify, Response, session, send_file
from datetime import datetime
import bcrypt
import matplotlib
import pdfkit
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from CNN_Model import create_CNN_model

app = Flask(__name__, static_folder='static')
app.secret_key = os.urandom(24)

DB_params = {
    "dbname": "Emotions",
    "user": "Admin",
    "password": "password1",
    "host": "34.147.132.176",
    "port": "5432"
}

# Upload Config
upload_folder = 'static/uploads/'
app.config['upload_folder'] = upload_folder
os.makedirs(upload_folder, exist_ok=True)
config = pdfkit.configuration(wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe")


# Load in the trained Model
num_classes = 7
model = create_CNN_model(num_classes)
model.load_state_dict(torch.load("Best_FER_Model.pth", weights_only=True))
model.eval()

# Use OpenCV face detection haarcascade
face_detection = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Define the emotions
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Define the global Variables
recording = False
paused = False
session_name = ""
session_start_time = None
active_session_id = None

# Color map
color_map = {
    'anger': 'red',
    'disgust': 'green',
    'fear': 'lightblue',
    'happy': 'gold',
    'neutral': 'brown',
    'sad': 'blue',
    'surprised': 'purple'
}

# Buffer for intensities
intensities_buffer = []
last_save_time = None
save_interval = 2.0

# Left Panel bar chart dictionary
current_intensities = {emo: 0.0 for emo in emotions}

# Function to store emotion intensities
def store_emotion(session_id, intensities_buffer):
    if not intensities_buffer:
        return

    # Average the intensities
    avg_array = np.mean(intensities_buffer, axis=0)

    # Connect to the database and insert the intensities
    try:
        conn = psycopg2.connect(**DB_params)
        cur = conn.cursor()

        cur.execute(
            "INSERT INTO emotion_records (session_id, timestamp, anger, disgust, fear, happy, neutral, sad, surprised) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (session_id, datetime.now(), *avg_array)
        )

        conn.commit()
        cur.close()
        conn.close()

    except Exception as e:
        print("Error storing emotion intensities:", e)

# Begin generating frames for the create session page
def generate_frames():
    # Reference global variables
    global recording, paused, active_session_id, intensities_buffer, last_save_time, current_intensities

    # Turn on camera
    cap = cv2.VideoCapture(0)

    # While loop to process frames
    while cap.isOpened():
        success, frame = cap.read()

        # Break if frames aren't detected
        if not success:
            break

        # Reset intensities
        intensities = None

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            # Process Frames
            (x, y, w, h) = faces[0]
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (48, 48))
            face_normalized = face_resized / 255.0
            face_tensor = torch.tensor(face_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            # Predict emotion
            with torch.no_grad():
                prediction = model(face_tensor)
                intensities = torch.softmax(prediction, dim=1).squeeze().tolist()

            # Update current_intensities for the bar chart
            for i, emo in enumerate(emotions):
                current_intensities[emo] = intensities[i]

            # Get most dominant emotion
            label_idx = torch.argmax(prediction, dim=1).item()
            emotion = emotions[label_idx]

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (140, 240, 140), 2)
            cv2.putText(frame, f"{emotion}: {intensities[label_idx]:.2f}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

        # if recording add intensities to the buffer
        if recording and not paused and active_session_id is not None and intensities is not None:
            intensities_buffer.append(intensities)
            now = datetime.now()

            # Check if interval has been passed
            if last_save_time and (now - last_save_time).total_seconds() >= save_interval:
                store_emotion(active_session_id, intensities_buffer)
                intensities_buffer = []
                last_save_time = now
        else:
            # No face detected
            pass

        # Convert to JPEG format for web display
        ret_jpeg, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        if not ret_jpeg:
            break

        # Send frame to frontend
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Function to process a frame for prediction
def preprocessing(img_path):
    img = cv2.resize(img_path, (224, 224))
    face = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face, (48, 48))
    face_norm = face_resized / 255.0
    face_tensor = torch.tensor(face_norm, dtype=torch.float32)
    face_tensor = face_tensor.unsqueeze(0).unsqueeze(0)
    return face_tensor

# Function to get the emotion intensity window
def generate_emotion_intensity():
    while True:
        intensity_window = np.zeros((350, 450, 3), dtype=np.uint8)

        # Draw emotion bars
        start_y = 40
        max_bar_width = 300
        bar_height = 30

        # For loop to constantly update the emotion intensity frame
        for i, emotion in enumerate(emotions):
            # Get intensity for each emotion
            prob = current_intensities[emotion]
            bar_width = int(prob * max_bar_width)

            # Draw the bar based on the bar width variable
            cv2.rectangle(intensity_window, (20, start_y + i * 40), (20 + bar_width, start_y + i * 40 + bar_height),
                          (0, 255, 0), -1)
            cv2.putText(intensity_window, f"{emotion}: {prob:.2f}", (25, start_y + i * 40 + int(bar_height * 0.7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        _, buffer_intensity = cv2.imencode('.jpg', intensity_window)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer_intensity.tobytes() + b'\r\n')

# Generate bar chart for a session
def generate_bar_chart(session_id):
    # Connect to database and get the average intensities from each emotion
    try:
        conn = psycopg2.connect(**DB_params)
        query = """
            SELECT 
                AVG(happy) AS avg_happy,
                AVG(neutral) AS avg_neutral,
                AVG(surprised) AS avg_surprised,
                AVG(disgust) AS avg_disgust,
                AVG(anger) AS avg_anger,
                AVG(fear) AS avg_fear,
                AVG(sad) AS avg_sad
            FROM emotion_records
            WHERE session_id = %s;
            """
        df = pd.read_sql(query, conn, params=(session_id,))
        conn.close()
    except Exception as e:
        print("DB error:", e)
        return None

    # If no data exist return nothing
    if df.empty:
        return None

    # Convert any None into 0.0
    anger = safe_float(df['avg_anger'][0])
    disgust = safe_float(df['avg_disgust'][0])
    fear = safe_float(df['avg_fear'][0])
    happy = safe_float(df['avg_happy'][0])
    neutral = safe_float(df['avg_neutral'][0])
    sad = safe_float(df['avg_sad'][0])
    surprised = safe_float(df['avg_surprised'][0])

    # reorder the emotions based on the emotion labels
    values = [anger, disgust, fear, happy, neutral, sad, surprised]
    labels = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']

    # Plot the graph
    plt.figure(figsize=(6,6))
    plt.bar(labels, values, color=[color_map[label] for label in labels])
    plt.xlabel("Emotions")
    plt.ylabel("Intensity")
    plt.title(f"Emotion Intensity for Session {session_name}")
    plt.ylim(0,1)

    # Save to folder
    save_path = f"static/charts/bar_{session_name}.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return save_path

# Function to generate box plot graph for the session
def generate_box_plot(session_id, session_name):
    # Connect to database and get all intensities for each emotion
    try:
        conn = psycopg2.connect(**DB_params)
        query = "SELECT anger, disgust, fear, happy, neutral, sad, surprised FROM emotion_records WHERE session_id = %s;"
        df = pd.read_sql(query, conn, params=(session_id,))
        conn.close()
    except Exception as e:
        print("DB error:", e)
        return None

    if df.empty:
        return None

    # Plot the data on a box plot using plt
    plt.figure(figsize=(8,6))
    data = [df[emotion].dropna() for emotion in color_map.keys()]
    box = plt.boxplot(data, patch_artist=True, labels=color_map.keys())

    for patch, emotion in zip(box['boxes'], color_map.keys()):
        patch.set_facecolor(color_map[emotion])
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)

    plt.title(f"Box Plot for Session {session_name}")
    plt.ylabel("Emotion Intensity")

    # Make directory if doesn't exist and save plot
    chart_dir = "static/charts"
    os.makedirs(chart_dir, exist_ok=True)
    save_path = f"{chart_dir}/boxplot_{session_name}.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return save_path

# Generate a histogram for a session
def generate_histogram(session_id, session_name, emotion_choice="all"):
    # Connect to a database and get all the intensities for each emotion
    try:
        conn = psycopg2.connect(**DB_params)
        query = "SELECT anger, disgust, fear, happy, neutral, sad, surprised FROM emotion_records WHERE session_id = %s;"
        df = pd.read_sql(query, conn, params=(session_id,))
        conn.close()
    except Exception as e:
        print("DB error:", e)
        return None

    # If the dataframe is empty return nothing
    if df.empty:
        return None

    # If and else statement to determine which histogram to generate based on user input
    if emotion_choice == "all":
        # Generate each emotions histogram for frequency and intensity appeared in the session and subplot them
        fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(16,12))
        ax = ax.flatten()

        columns = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
        for i, col in enumerate(columns):
            ax[i].hist(df[col], bins = 12, color=color_map[col], edgecolor='black')
            ax[i].set_title(col.capitalize())
            ax[i].set_xlim(0,1)
            ax[i].set_ylim(bottom=0)
            ax[i].set_xlabel("Intensity")
            ax[i].set_ylabel("Frequency")

        if len(columns) < len(ax):
            fig.delaxes(ax[-1])

        fig.suptitle(f"Histogram for Session {session_name}")

    else:
        # For the selected emotion generate and display the one graph
        if emotion_choice not in color_map:
            return None
        plt.figure(figsize=(10, 8))
        plt.hist(df[emotion_choice], bins=12, color=color_map[emotion_choice], edgecolor='black')
        plt.title(f"{emotion_choice.capitalize()} Histogram for Session {session_name}")
        plt.xlim(0,1)
        plt.ylim(bottom=0)
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")

    # Save each graph
    chart_dir = "static/charts"
    os.makedirs(chart_dir, exist_ok=True)

    # Name for each graph
    if emotion_choice == "all":
        save_path = f"{chart_dir}/histogram_{session_name}_all.png"
    else:
        save_path = f"{chart_dir}/histogram_{session_name}_{emotion_choice}.png"

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return save_path

# Function to generate a line graph for all emotions and selected emotions
def generate_line_graph(session_id, session_name, emotion_choice='all'):
    # Connect to the database and get the timestamp and every emotion intensity from the emotion records
    try:
        conn = psycopg2.connect(**DB_params)
        query = "SELECT timestamp, anger, disgust, fear, happy, neutral, sad, surprised FROM emotion_records WHERE session_id = %s ORDER BY timestamp;"
        df = pd.read_sql(query, conn, params=(session_id,))
        conn.close()
    except Exception as e:
        print("DB error:", e)
        return None

    # Check if dataframe is empty
    if df.empty:
        return None

    # Convert to datetime if not already stored, error check
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Create a empty list
    emotion_cols = []

    # Determine what line graph to make
    if emotion_choice == 'all':
        emotion_cols = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
    else:
        emotion_cols = [emotion_choice] if emotion_choice in color_map else []

    # Plot graph
    plt.figure(figsize=(12,6))
    for col in emotion_cols:
        plt.plot(
            df['timestamp'],
            df[col],
            label=col.capitalize(),
            color=color_map.get(col, 'black')
        )

    # Determine title for graph based on user input
    if emotion_cols:
        plt.title(f"Line Graph - {session_name} ({emotion_choice.capitalize()})")
    else:
        plt.title(f"Line Graph - {session_name}")

    # Plot graph
    plt.xlabel("Time")
    plt.ylabel("Emotion Intensity")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    # Save graph to static folder
    chart_dir = "static/charts"
    os.makedirs(chart_dir, exist_ok=True)

    # Determine graph name saved as
    if emotion_choice == "all":
        save_path = f"{chart_dir}/line_graph_{session_name}_all.png"
    else:
        save_path = f"{chart_dir}/line_graph_{session_name}_{emotion_choice}.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return save_path

# Function to generate a pie chart for a session
def generate_pie_chart(session_id, session_name):
    # Connect to the database and get the average intensities for each emotion in a session
    try:
        conn = psycopg2.connect(**DB_params)
        query = ("""SELECT
                 AVG(happy) AS avg_happy,
                 AVG(neutral) AS avg_neutral,
                 AVG(surprised) AS avg_surprised,
                 AVG(sad) AS avg_sad,
                 AVG(anger) AS avg_anger,
                 AVG(disgust) AS avg_disgust,
                 AVG(fear) AS avg_fear
                 FROM emotion_records
                 WHERE session_id = %s;""")
        df = pd.read_sql(query, conn, params=(session_id,))
        conn.close()
    except Exception as e:
        print("DB error:", e)
        return None

    # Check if dataframe is empty
    if df.empty:
        return None

    # Convert any None into 0.0
    anger = safe_float(df['avg_anger'][0])
    disgust = safe_float(df['avg_disgust'][0])
    fear = safe_float(df['avg_fear'][0])
    happy = safe_float(df['avg_happy'][0])
    neutral = safe_float(df['avg_neutral'][0])
    sad = safe_float(df['avg_sad'][0])
    surprised = safe_float(df['avg_surprised'][0])

    # Organize the labels and values
    labels = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
    values = [anger, disgust, fear, happy, neutral, sad, surprised]

    # Create empty lists to fill in the label values and color
    filtered_labels = []
    filtered_values = []
    filtered_colors = []
    for lab, val in zip(labels, values):
        if val is not None and val > 0:
            filtered_labels.append(lab)
            filtered_values.append(val)
            filtered_colors.append(color_map.get(lab, 'gray'))

    # Plot the pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(
        filtered_values,
        labels=filtered_labels,
        colors=filtered_colors,
        autopct='%1.1f%%',
        startangle=140
    )
    plt.title(f"Pie Chart - Session '{session_name}'")

    # Save the pie chart
    chart_dir = "static/charts"
    os.makedirs(chart_dir, exist_ok=True)
    save_path = f"{chart_dir}/pie_chart_{session_name}.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return save_path

# Function to prevent null emotions from stopping chart generation
def safe_float(val):
    if val is None:
        return 0.0
    return float(val)

# Function to generate all the charts for when the session is created
def generate_all_charts(session_id, session_name):
    # Pre-generate charts for the session and store them
    try:
        chart_dir = "static/charts"
        os.makedirs(chart_dir, exist_ok=True)
        generate_bar_chart(active_session_id)
        generate_box_plot(active_session_id, session_name)
        generate_pie_chart(active_session_id, session_name)
        generate_histogram(active_session_id, session_name, "all")
        for emo in ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']:
            generate_histogram(active_session_id, session_name, emo)
        generate_line_graph(active_session_id, session_name, "all")
        for emo in ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']:
            generate_line_graph(active_session_id, session_name, emo)
    except Exception as chart_error:
        print("Chart generation error:", chart_error)


# ----------------ROUTES---------------- #
# Stream the emotion intensities
@app.route('/intensity_feed')
def intensity_feed():
    return Response(generate_emotion_intensity(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Home page
@app.route('/')
def home():
    # Redirect to login page if not logged in
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Create name variable for each user
    name = None
    # Connect to database to get name and store it
    try:
        conn = psycopg2.connect(**DB_params)
        cur = conn.cursor()
        cur.execute("SELECT name FROM users WHERE user_id = %s", (session['user_id'],))
        row = cur.fetchone()
        if row:
            name = row[0]
        cur.close()
        conn.close()
    except Exception as e:
        print("Error fetching users name from database:", e)

    # Return the generated template
    return render_template(
        "index.html",
        css_file="index.css",
        logged_in=True,
        name=name
    )

# Route to generate account page with session data
@app.route('/account')
def account():
    # Check if user is logged in
    if 'user_id' not in session:

        return redirect(url_for('login'))

    # Create local variables to store data
    user_id = session['user_id']
    user_name = None
    session_data = []

    # Connect ot database to retrieve user and session data
    try:
        conn = psycopg2.connect(**DB_params)
        cur = conn.cursor()

        # Get users name
        cur.execute("SELECT name FROM users WHERE user_id = %s", (user_id,))
        user_name_row = cur.fetchone()
        if user_name_row:
            user_name = user_name_row[0]

        # Get any sessions attached to the account
        cur.execute("""
            SELECT session_id, session_name, start_time, end_time, created_at
            FROM sessions
            WHERE user_id = %s
            ORDER BY created_at ASC
            """, (user_id,))
        rows = cur.fetchall()

        # Organise the data
        for (sid, sname, stime, etime, ctime) in rows:
            # Calculate the duration
            if stime and etime:
                sessiontime = etime - stime
                total_seconds = int(sessiontime.total_seconds())
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                seconds = total_seconds % 60
                if hours:
                    duration = f"{hours}h {minutes}m {seconds}s"
                elif minutes:
                    duration = f"{minutes}m {seconds}s"
                else:
                    duration = f"{seconds}s"
            else:
                duration = "N/A"

            # Append each session to the session_data
            session_data.append({
                "session_id": sid,
                "session_name": sname,
                "duration": duration,
                "created_at": ctime.strftime("%Y-%m-%d %H:%M:%S") if ctime else "N/A"
            })

        cur.close()
        conn.close()
    except Exception as e:
        print("Error fetching users session information from database:", e)

    # Render the account page
    return render_template(
        "account.html",
        user_name=user_name,
        sessions=session_data
    )

# Route to rename the session
@app.route("/rename_Session", methods=['POST'])
def rename_session():
    # Check if user is logged in
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # get data from the data json
    data = request.get_json()
    session_id = data.get("session_id")
    new_name = data.get("new_name")

    # Error check to ensure all variables have data
    if not session_id or not new_name:
        return jsonify({"status": "error", "message": "Missing new name or user"})

    # Connect to database and update name
    try:
        conn = psycopg2.connect(**DB_params)
        cur = conn.cursor()
        cur.execute("""
            UPDATE sessions
            SET session_name = %s
            WHERE session_id = %s AND user_id = %s
            """, (new_name, session_id, session['user_id']))
        if cur.rowcount == 0:
            conn.rollback()
            cur.close()
            conn.close()
            return jsonify({"status": "error", "message": "Session does not exist"})

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({"status": "success", "message": "Session has been renamed"})
    except Exception as e:
        print("Error fetching users session information from database:", e)
        return jsonify({"status": "error", "message": "str(e)"})

# Route to allow user to delete a session
@app.route("/delete_session", methods=['POST'])
def delete_session():
    # Check user is logged in
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Get the data from the json
    data = request.get_json()
    session_id = data.get("session_id")

    # Check a session id was stored
    if not session_id:
        return jsonify({"status": "error", "message": "Missing session id"})

    # Connect to database and delete the session
    try:
        conn = psycopg2.connect(**DB_params)
        cur = conn.cursor()
        cur.execute("DELETE FROM emotion_records WHERE session_id = %s", (session_id,))
        cur.execute("DELETE FROM sessions WHERE session_id = %s AND user_id = %s", (session_id, session['user_id']))

        if cur.rowcount == 0:
            conn.rollback()
            cur.close()
            conn.close()
            return jsonify({"status": "error", "message": "Session does not exist"})

        conn.commit()
        cur.close()
        conn.close()
        return jsonify({"status": "success", "message": "Session has been deleted"})
    except Exception as e:
        print("Error fetching users session information from database:", e)
        return jsonify({"status": "error", "message": "str(e)"})




# Stream the video captured by camera back
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Bring the user to the create session page
@app.route('/create_session')
def create_session():
    return render_template("create_session.html", css_file="create_session.css")

# Start the session when the user clicks start session
@app.route('/start_session', methods=['POST'])
def start_session():
    # Create global variables for session info and recording status
    global recording, paused, session_name, session_start_time, active_session_id, intensities_buffer, last_save_time
    global active_session_id, intensities_buffer, last_save_time

    # Check user is logged in
    if 'user_id' not in session:
        return jsonify({"status": "error", "message": "User not logged in"})

    # Get the session name and user id
    s_name = request.form.get('session_name')
    user_id = session['user_id']

    # Check the session name is entered
    if not s_name:
        return jsonify({"status": "error", "message": "Session name required before beginning a session"})

    # Check if there is an active session id in the memory
    if active_session_id is not None:
        # Check if the session is currently paused
        if paused:
            paused = False
            recording = True
            session_name = s_name
            return jsonify({"status": "success", "message": f"Session {session_name} resumed"})
        else:
            return jsonify({"status": "error", "message": "Session is already recording"})

    # Declare variables for recording session
    recording = True
    paused = False
    session_name = s_name
    session_start_time = datetime.now()
    intensities_buffer = []
    last_save_time = datetime.now()

    # Store session in the database
    try:
        conn = psycopg2.connect(**DB_params)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO sessions (user_id, session_name, start_time, created_at) VALUES (%s, %s, %s, %s) RETURNING session_id",
            (user_id, session_name, session_start_time, datetime.now()))
        active_session_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({"status": "success", "message": f"Session {session_name} Started!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Route to pause the session when button is clicked
@app.route('/pause_session', methods=['POST'])
def pause_session():
    # Get global variables
    global recording, paused, active_session_id

    # Ensure user is logged in
    if 'user_id' not in session:
        return jsonify({"status": "error", "message": "User not logged in"})

    # Check if there's an active session
    if active_session_id is None:
        return jsonify({"status": "error", "message": "No active session to pause."})

    # Check if the session is recording
    if recording:
        # Update status of variables
        recording = False
        paused = True

        # Notify user
        return jsonify({"status": "success", "message": "Session paused!"})
    else:
        return jsonify({"status": "error", "message": "No active session to pause"})

# Route to login user
@app.route('/login', methods=['GET', 'POST'])
def login():
    # Get email and password
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Connect to database and get the user details matched to the email inputted
        try:
            conn = psycopg2.connect(**DB_params)
            cur = conn.cursor()
            cur.execute("SELECT user_id, password FROM users WHERE email = %s", (email,))
            user = cur.fetchone()
            cur.close()
            conn.close()

            # Check if user exists
            if user:
                user_id, db_hashed_password = user
                # Check the hashed password is stored as a string
                if isinstance(db_hashed_password, str):
                    # Convert to bytes if is
                    db_hashed_password = db_hashed_password.encode('utf-8')

                # Compare entered passworded hashed to hashed password stored
                if bcrypt.checkpw(password.encode('utf-8'), db_hashed_password):
                    # If match store user id as the session number
                    session['user_id'] = user_id
                    # Redirect to home page
                    return redirect(url_for('home'))

                # Else inform user invalid input
                else:
                    return "Invalid credentials"
            else:
                return "Invalid credentials"  # No such user
        except Exception as e:
            return f"Error: {str(e)}"
    return render_template('login.html')

# Route to log out the user and terminate session
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

# Route to create a new user account
@app.route('/signup', methods=['GET', 'POST'])
def sign_up():
    if request.method == 'POST':
        # Get inputted email, name and password
        email = request.form['email']
        name = request.form['name']
        password = request.form['password']

        # Encrypt the bassword into bytes
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        # Connect to database and insert the inputted information
        try:
            conn = psycopg2.connect(**DB_params)
            cur = conn.cursor()
            cur.execute("INSERT INTO users (email, name, password) VALUES (%s, %s, %s)", (email, name, hashed_password))
            conn.commit()
            cur.close()
            conn.close()
            return redirect(url_for('login'))
        except Exception as e:
            return f"Error: {str(e)}"

    return render_template('signup.html')

@app.route('/resume_session', methods=['POST'])
def resume_session():
    # Call the global variables
    global recording, paused, active_session_id, session_name, session_start_time

    # Check if the user is logged in
    if 'user_id' not in session:
        return jsonify({"status": "error", "message": "User not logged in"})

    # If there's no active session_id, can't resume
    if active_session_id is None:
        return jsonify({"status": "error", "message": "No paused session to resume."})

    # If it's not paused, can't resume either
    if not paused:
        return jsonify({"status": "error", "message": "Session is not paused."})

    # Update recording status and button status
    paused = False
    recording = True

    return jsonify({"status": "success", "message": f"Session {session_name} resumed successfully."})

# Route to end session and send the saved data to the database
@app.route('/end_session', methods=['POST'])
def end_session():
    # Call the global variables
    global recording, session_name, session_start_time, paused, active_session_id, intensities_buffer

    # Check if the user is logged in
    if 'user_id' not in session:
        return jsonify({"status": "error", "message": "User not logged in"})

    # if no active session cant end it
    if active_session_id is None:
        return jsonify({"status": "error", "message": "No active session."})

    # Check the recording is at least 10 seconds long
    elapsed = (datetime.now() - session_start_time).total_seconds()
    if elapsed < 10:
        return jsonify({"status": "error", "message": "Session too short!"})

    # Stop recording
    recording = False
    paused = False

    # Check the lists and session id are not empty
    if intensities_buffer and active_session_id is not None:
        # Call the store emotion function
        store_emotion(active_session_id, intensities_buffer)
        # Reset the intensity buffer
        intensities_buffer = []

    # Save the end time
    end_time = datetime.now()

    # Connect to the database and update the end time of the session
    try:
        conn = psycopg2.connect(**DB_params)
        cur = conn.cursor()
        cur.execute("UPDATE sessions SET end_time = %s WHERE session_id = %s", (end_time, active_session_id))
        conn.commit()
        cur.close()
        conn.close()

        # Capture for thread to run in background
        thread_sid = active_session_id
        thread_name = session_name

        # Clear active session id to prevent further recording
        active_session_id = None

        # Function to call the generating of charts
        def background_chart_generation():
            generate_all_charts(thread_sid, thread_name)

        # Start the thread
        thread = threading.Thread(target=background_chart_generation)
        thread.start()

        return jsonify({"status": "success",
                        "message": f"Session {session_name} Uploaded Successfully! Generating charts in background...",
                        "session_id": thread_sid})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Route to close the app
@app.route('/shutdown', methods=['POST'])
def shutdown():
    def delayed_shutdown():
        time.sleep(5)
        os.kill(os.getpid(), signal.SIGTERM)

    # Start the shutdown delay in a background thread
    threading.Thread(target=delayed_shutdown).start()

    # Render the goodbye screen
    return render_template("shutdown.html")

# Route to redirect the user to the visualize data page with the stored sessions
@app.route('/visualize_data')
def visualize_data():
    # Check user is signed in
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Get user id from session
    user_id  = session['user_id']

    # Connect to the database and get all sessions linked to the account
    try:
        conn = psycopg2.connect(**DB_params)
        cur = conn.cursor()
        cur.execute("SELECT session_id, session_name FROM sessions WHERE user_id = %s", (user_id,))
        sessions = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as e:
        print("Error fetching sessions:", e)
        sessions = []

    return render_template('visualize_data.html', sessions=sessions)

#
@app.route('/bar_chart')
def bar_chart():
    # Pass session_id
    sid = request.args.get('session_id')

    # if no session id inform user
    if not sid:
        return "No session_id provided", 400

    # Validate session id is an integer
    try:
        sid = int(sid)
    except ValueError:
        return "Invalid session_id", 400

    # Connect to database and get session names created by the user
    try:
        conn = psycopg2.connect(**DB_params)
        cur = conn.cursor()
        cur.execute("SELECT session_name FROM sessions WHERE session_id = %s", (sid,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        # If a session exist or row get equal the result to session_name
        if row:
            session_name = row[0]
        else:
            return "Session does not exist", 404
    except Exception as e:
        print("Error fetching session name:", e)
        return "Database error", 500

    # Generate the bar chart with the session id
    img_path = generate_bar_chart(sid)

    # If no image is generated give error message
    if not img_path:
        return "No data for bar chart", 500

    # Render a bar chart html template
    return render_template(
        "bar_chart.html",
        session_id=active_session_id,
        session_name=session_name,
        bar_chart_path="/"+img_path
    )

# Route to call the bar chart function for the session selected
@app.route('/box_plot')
def box_plot():
    #  Get the session id selected
    sid_str = request.args.get('session_id')

    # Check if there was a session id
    if not sid_str:
        return "No session_id provided", 400

    # Validate the session id is an integer
    try:
        sid = int(sid_str)
    except ValueError:
        return "Invalid session_id", 400

    # Connect to the database and get the session name
    try:
        conn = psycopg2.connect(**DB_params)
        cur = conn.cursor()
        cur.execute("SELECT session_name FROM sessions WHERE session_id = %s", (sid,))
        row = cur.fetchone()
        cur.close()
        conn.close()

        # Check if a session exists and equal it to the sess_name variable
        if row:
            sess_name = row[0]
        else:
            return "Session does not exist", 404
    except Exception as e:
        print("Error fetching session name:", e)
        return "Database error", 500

    # Generate the box plot
    img_path = generate_box_plot(sid, sess_name)
    if not img_path:
        return "No data for box plot or DB error", 500

    # Render a new template box_plot.html
    return render_template(
        "box_plot.html",
        session_id=sid,
        session_name=sess_name,
        box_plot_path="/" + img_path
    )

# Route to call the generate histogram function based on the selected session
@app.route('/histogram')
def histogram_page():
    # Get the session id selected from the session
    sid_str = request.args.get('session_id')

    # Check if the session id exists
    if not sid_str:
        return "No session_id provided", 400

    # Validate the session id is an int
    try:
        sid = int(sid_str)
    except ValueError:
        return "Invalid session_id", 400

    # Pass the emotions selected from the user to emotion parameter variable
    emotion_param = request.args.get('emotion', 'all').lower()

    # Connect to the database and get the session names
    try:
        conn = psycopg2.connect(**DB_params)
        cur = conn.cursor()
        cur.execute("SELECT session_name FROM sessions WHERE session_id = %s", (sid,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        # If a row exists equal the session name to the variable
        if row:
            sess_name = row[0]
        else:
            return "Session does not exist", 404
    except Exception as e:
        print("Error fetching session name:", e)
        return "Database error", 500

    # Generate the histogram from the user inputs
    img_path = generate_histogram(sid, sess_name, emotion_param)
    if not img_path:
        return "No data available for histogram", 500

    # Render a new "histogram.html" template
    return render_template(
        "histogram.html",
        session_id=sid,
        session_name=sess_name,
        histogram_path="/" + img_path
    )

# Route to call the generate line graph from the user inputs
@app.route('/line_graph')
def line_graph_route():
    # Get the session id selected
    sid_str = request.args.get('session_id')

    # Check if the session id exists
    if not sid_str:
        return "No session_id provided", 400

    # Validate the session id is an int
    try:
        sid = int(sid_str)
    except ValueError:
        return "Invalid session_id", 400

    # Get the emotions inputted for generation by the user
    emotion_param = request.args.get('emotion', 'all').lower()

    # Connect to the database and get the session names
    try:
        conn = psycopg2.connect(**DB_params)
        cur = conn.cursor()
        cur.execute("SELECT session_name FROM sessions WHERE session_id = %s", (sid,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            sess_name = row[0]
        else:
            return "Session does not exist", 404
    except Exception as e:
        print("Error fetching session name:", e)
        return "Database error", 500

    # Generate the line graph from the user inputs gathered
    img_path = generate_line_graph(sid, sess_name, emotion_param)
    if not img_path:
        return "No data for line graph or DB error", 500

    # Render the line_graph.html template
    return render_template(
        "line_graph.html",
        session_id=sid,
        session_name=sess_name,
        line_graph_path="/" + img_path
    )

# Route to generate the pie chart for the session selected
@app.route('/pie_chart')
def pie_chart():
    # Retrieve the session id from the user input
    sid_str = request.args.get('session_id')

    # Check if session id exists
    if not sid_str:
        return "No session_id provided", 400

    # Validate the session id is a int
    try:
        sid = int(sid_str)
    except ValueError:
        return "Invalid session_id", 400

    # Connect to database and get the session name from the session id
    try:
        conn = psycopg2.connect(**DB_params)
        cur = conn.cursor()
        cur.execute("SELECT session_name FROM sessions WHERE session_id = %s", (sid,))
        row = cur.fetchone()
        cur.close()
        conn.close()

        # Check if row exists and equal row to sess_name variable
        if row:
            sess_name = row[0]
        else:
            return "Session does not exist", 404
    except Exception as e:
        print("Error fetching session name:", e)
        return "Database error", 500

    # Generate line graph with the session id and name
    img_path = generate_pie_chart(sid, sess_name)
    if not img_path:
        return "No data for pie chart or DB error", 500

    # Render the line_graph.html template
    return render_template(
        "pie_chart.html",
        session_id=sid,
        session_name=sess_name,
        pie_chart_path="/" + img_path
    )

# Route to render the pdf form
@app.route('/pdf_form')
def pdf_form():
    # Check if user id is in session
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # check if there's a get request
    if request.method == 'GET':
        # Get the session ID from the users selection
        sid_str = request.args.get('session_id')

        # Check if there isn't a session id
        if not sid_str:
            return "No session_id provided", 400
        # Validate the session id as an int
        try:
            sid = int(sid_str)
        except ValueError:
            return "Invalid session_id", 400

        # Connect to databases retrieves the session name from session id gathered
        try:
            conn = psycopg2.connect(**DB_params)
            cur = conn.cursor()
            cur.execute("SELECT session_name FROM sessions WHERE session_id = %s", (sid,))
            row = cur.fetchone()
            cur.close()
            conn.close()
            # checks if a row was returned and equals it to sess_name
            if row:
                sess_name = row[0]
            else:
                return "Session does not exist", 404
        except Exception as e:
            print("Error fetching session name:", e)
            return "Database error", 500

        # Render pdf_form.html template using session id and session name
        return render_template("pdf_form.html", session_id=sid, session_name=sess_name)
    else:
        session_id = request.form.get('session_id')
        charts_selected = request.form.getlist('charts')
        histogram_choices = request.form.getlist('histogram')
        linegraph_choices = request.form.getlist('linegraph')
        extras = request.form.getlist('extras')

    # Prepare query string for redirecting to the generate_pdf
    qs = {
        'session_id': session_id,
        'charts': ','.join(charts_selected),
        'histogram': ','.join(histogram_choices),
        'linegraph': ','.join(linegraph_choices),
        'extras': ','.join(extras)
    }
    return redirect(url_for('generate_pdf', **qs))


# Route to generate the pdf
@app.route('/generate_pdf', methods=['GET', 'POST'])
def generate_pdf():
    # Check if user is logged if not redirect to login
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Get user id from session id
    user_id = session['user_id']
    # Equal session id to the variable
    session_id = request.form.get('session_id')

    # Check if session id is present
    if not session_id:
        return "No session_id provided", 400

    # Parse selected checkboxes from the form
    charts_selected = request.form.getlist('charts')
    histogram_choices = request.form.getlist('histogram')
    linegraph_choices = request.form.getlist('linegraph')
    extras = request.form.getlist('extras')

    # Query DB to get session info
    try:
        conn = psycopg2.connect(**DB_params)
        cur = conn.cursor()

        # Get the session info and username info with join statements
        cur.execute("""
            SELECT s.session_name, u.name, s.start_time, s.end_time, s.created_at
            FROM sessions s
            JOIN users u ON s.user_id = u.user_id
            WHERE s.session_id = %s
        """, (session_id,))
        session_data = cur.fetchone()
        # Get user name from user id
        cur.execute("SELECT name FROM users WHERE user_id = %s", (user_id,))
        user_data = cur.fetchone()
        cur.close()
        conn.close()

        # Check if session data exists
        if not session_data:
             return "Session does not exist", 404
        # If session data exists equal the data into variables
        if session_data:
            session_name, user_name, start_time, end_time, created_at = session_data
        user_name = user_data[0] if user_data else "Unknown"
    except Exception as e:
        print("Database error:", e)
        return "Database error", 500

    # Create an empty dictionary for average intensities
    avg_intensities = {}
    # Connect to data base and get average emotions from the session
    try:
        conn = psycopg2.connect(**DB_params)
        query = """
        SELECT AVG(anger), AVG(disgust), AVG(fear), AVG(happy), AVG(neutral), AVG(sad), AVG(surprised)
        FROM emotion_records WHERE session_id = %s"""
        df = pd.read_sql(query, conn, params=(session_id,))
        conn.close()
        # Check if data frame is not empty
        if not df.empty:
            # Input the average intensities for the session
            avg_intensities = {
                "Anger": df.iloc[0, 0],
                "Disgust": df.iloc[0, 1],
                "Fear": df.iloc[0, 2],
                "Happy": df.iloc[0, 3],
                "Neutral": df.iloc[0, 4],
                "Sad": df.iloc[0, 5],
                "Surprised": df.iloc[0, 6],
            }
    except Exception as e:
        print("Database error:", e)

    # Generate chart images
    chart_paths = {}
    chart_base_path = os.path.abspath("static/charts")

    # Check if bar was selected and generate it into the pdf template
    if "bar" in charts_selected:
        bar_chart_path = generate_bar_chart(session_id)
        if bar_chart_path:
            chart_paths["Bar Chart"] = f"file:///{os.path.abspath(bar_chart_path)}"

    # Check if box was selected and generate it for the pdf file
    if "box" in charts_selected:
        box_chart_path = generate_box_plot(session_id, session_name)
        if box_chart_path:
            chart_paths["Box Plot"] = f"file:///{os.path.abspath(box_chart_path)}"

    # Check if pie was selected and generate it for the pdf file
    if "pie" in charts_selected:
        pie_chart_path = generate_pie_chart(session_id, session_name)
        if pie_chart_path:
            chart_paths["Pie Chart"] = f"file:///{os.path.abspath(pie_chart_path)}"

    # For loop to check what emotions where selected in histogram
    for emotion in histogram_choices:
        hist_path = generate_histogram(session_id, session_name, emotion)
        if hist_path:
            chart_paths[f"Histogram ({emotion.capitalize()})"] = f"file:///{os.path.abspath(hist_path)}"

    # For loop to check what emotions where selected for line graph
    for emotion in linegraph_choices:
        line_path = generate_line_graph(session_id, session_name, emotion)
        if line_path:
            chart_paths[f"Line Graph ({emotion.capitalize()})"] = f"file:///{os.path.abspath(line_path)}"

    # Only include existing chart paths
    chart_paths = {title: path for title, path in chart_paths.items() if os.path.exists(path.replace("file:///", ""))}

    # Add the duration
    if start_time and end_time:
        total_seconds = int((end_time - start_time).total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        session_duration = f"{hours} hrs {minutes} mins {seconds} secs" if hours else \
            f"{minutes} mins {seconds} secs" if minutes else \
                f"{seconds} secs"
    else:
        session_duration = "N/A"

    # Format the created at in years months days hours and minutes
    formatted_created_at = created_at.strftime('%Y-%m-%d %H:%M') if created_at else "N/A"

    # Render HTML to PDF
    rendered_html = render_template(
        "pdf_template.html",
        session_name=session_name,
        user_name=user_name,
        session_duration=session_duration,
        created_at=formatted_created_at,
        avg_intensities=avg_intensities,
        chart_paths = {title: path for title, path in chart_paths.items() if path}  # Remove None paths

    )
    # Options for generating pdf file
    options = {
        "enable-local-file-access": "",
        "disable-smart-shrinking": "",
        "no-stop-slow-scripts": "",
        "javascript-delay": "1000",  # Give time for JavaScript to load
        "load-error-handling": "ignore",
        "load-media-error-handling": "ignore"
    }
    # Convert HTML to PDF
    pdf_path = f"static/reports/{session_name.replace(' ', '_')}.pdf"
    pdfkit.from_string(rendered_html, pdf_path, options=options, configuration=config)
    return send_file(pdf_path, as_attachment=True, download_name=f"Session_Report_{session_name}.pdf")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
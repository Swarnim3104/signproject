from flask import Flask, Response, jsonify, render_template
import cv2
import pickle
import mediapipe as mp
import numpy as np
import webbrowser
import threading

app = Flask(__name__)

# Load the trained model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Label dictionary (A-Z)
labels_dict = {i: chr(65 + i) for i in range(26)}

# Open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open the camera")
    exit()

predicted_character = "?"  # Store the latest recognized letter

# Function to stream video frames
def generate_frames():
    global predicted_character
    while True:
        success, frame = cap.read()
        if not success or frame is None:
            print("Error: Unable to capture frame")
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        predicted_character = "?"  # Reset each frame

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                x_ = [lm.x for lm in hand_landmarks.landmark]
                y_ = [lm.y for lm in hand_landmarks.landmark]
                data_aux = [coord for i in range(len(hand_landmarks.landmark)) for coord in (x_[i] - min(x_), y_[i] - min(y_))]

                try:
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict[int(prediction[0])]
                except Exception as e:
                    print(f"Error in prediction: {e}")

                cv2.putText(frame, predicted_character, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Route to serve the UI
@app.route('/')
def index():
    return render_template('index.html')

# Route to serve the live video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to get the latest predicted letter
@app.route('/predict', methods=['GET'])
def get_prediction():
    global predicted_character
    return jsonify({'predicted_character': predicted_character})

# Function to open the browser automatically
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == '__main__':
    threading.Timer(1, open_browser).start()  # Open UI after 1 second
    app.run(debug=True)

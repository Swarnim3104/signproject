import pickle
import cv2
import mediapipe as mp
import numpy as np
import threading
import base64
import flask
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import webbrowser

app = Flask(__name__)
CORS(app)  # Allow frontend to communicate with backend

# Load the model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# Label dictionary
labels_dict = {0: 'A', 1: 'B', 2: 'L', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 
               11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 
               21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

saved_text = ""  # Stores recognized letters
cap = cv2.VideoCapture(0)

def generate_frames():
    global saved_text
    while True:
        data_aux = []
        x_, y_ = [], []
        success, frame = cap.read()
        if not success:
            continue

        # Process frame
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        predicted_character = "?"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                for i in range(len(hand_landmarks.landmark)):
                    x, y = hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

            if x_ and y_:
                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                    data_aux.append(hand_landmarks.landmark[i].y - min(y_))

                try:
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict[int(prediction[0])]
                except Exception as e:
                    print(f"Error in model prediction: {e}")

        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        yield f"data:image/jpeg;base64,{frame_base64}|{predicted_character}"

@app.route('/video_feed')
def video_feed():
    return flask.Response(generate_frames(), mimetype='text/event-stream')

@app.route('/save_character', methods=['POST'])
def save_character():
    global saved_text
    data = request.json
    saved_text += data.get('character', '')
    return jsonify({'saved_text': saved_text})

@app.route('/get_text', methods=['GET'])
def get_text():
    return jsonify({'saved_text': saved_text})

if __name__ == '__main__':
    threading.Timer(1.25, lambda: webbrowser.open('http://localhost:3000')).start()
    app.run(debug=True, port=5000)

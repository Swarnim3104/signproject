# -*- coding: utf-8 -*-

import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# ✅ Unicode text drawing function using Pillow
def draw_text_unicode(img, text, position, font_path=r'D:\Swarnim\signtest\sign\sign-language-detector-python-master\sign-language-detector-python-master\Lohit-Devanagari.ttf', font_size=32, color=(0, 0, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# Load the model with error handling
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# Marathi label dictionary
labels_dict = {
    0: "नमस्कार", 1: "धन्यवाद", 2: "मी", 3: "तू", 4: "हो", 5: "नाही", 6: "सर्वांना", 7: "शुभेच्छा", 8: "माफ करा",
    9: "आभारी आहे", 10: "काय", 11: "कसे", 12: "कोण", 13: "कधी", 14: "कुठे", 15: "का", 16: "होय", 17: "नाही",
    18: "आता", 19: "नंतर", 20: "आज", 21: "उद्या", 22: "काल", 23: "सकाळ", 24: "संध्याकाळ", 25: "रात्र"
}

saved_text = ""

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Could not read frame")
        continue

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    predicted_character = "?"  # Default

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

        if not x_ or not y_:
            print("Error: No valid hand landmarks found")
            continue

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        try:
            prediction = model.predict([np.asarray(data_aux)])
            pred_index = int(prediction[0])
            if pred_index in labels_dict:
                predicted_character = labels_dict[pred_index]
            else:
                print(f"Prediction index {pred_index} not in labels_dict")
                predicted_character = "?"
        except Exception as e:
            print(f"Error in model prediction: {e}")
            predicted_character = "?"


        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

        # ✅ Draw Marathi text using PIL for proper rendering
        try:
            frame = draw_text_unicode(frame, predicted_character, (x1, y1 - 40), font_path='Lohit-Devanagari.ttf')
        except Exception as e:
            print(f"Error drawing unicode text: {e}")
            # fallback to OpenCV default if font fails
            cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  
        saved_text += predicted_character + " "  # Add a space after each word
        with open("output.txt", "w", encoding='utf-8') as f:
            f.write(saved_text.strip())  # Strip to remove trailing spaces
        print(f"Saved: {saved_text}")  # Show saved text in console


    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import os
import pickle
import mediapipe as mp
import cv2

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)  # Increased confidence

DATA_DIR = './data'

# Check if dataset directory exists
if not os.path.exists(DATA_DIR):
    print(f"Error: Data directory '{DATA_DIR}' not found!")
    exit()

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue  # Skip files that are not directories

    for img_path in os.listdir(dir_path):
        data_aux = []
        x_ = []
        y_ = []

        img_full_path = os.path.join(dir_path, img_path)
        img = cv2.imread(img_full_path)

        if img is None:
            print(f"Skipping {img_path}: Unable to read image")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)
        else:
            print(f"No hands detected in {img_full_path}")

# Ensure data is not empty before saving
if len(data) > 0:
    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print(f"Pickle file saved successfully with {len(data)} samples!")
else:
    print("No valid hand data found. Pickle file not created.")

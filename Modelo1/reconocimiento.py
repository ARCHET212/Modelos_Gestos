import cv2
import mediapipe as mp
import pickle
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Cargar modelo entrenado
with open("Modelo1\modelo1_v1.pkl", "rb") as f:
    clf = pickle.load(f)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

THRESHOLD = 0.7  # confianza mínima

def normalizar_landmarks(landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    wrist = coords[0]  # origen en muñeca
    coords -= wrist
    escala = np.linalg.norm(coords[9])  # muñeca → dedo medio
    coords /= escala if escala > 0 else 1
    return coords.flatten()

with mp_hands.Hands(static_image_mode=False, max_num_hands=1) as hands:
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
              
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Normalizar antes de predecir
                data = normalizar_landmarks(hand_landmarks.landmark)

              
                probs = clf.predict_proba([data])[0]
                pred_index = np.argmax(probs)
                confidence = probs[pred_index]

                if confidence >= THRESHOLD:
                    pred_label = clf.classes_[pred_index]
                    cv2.putText(frame, f"Gesto: {pred_label} ({confidence:.2f})", 
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Cam', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

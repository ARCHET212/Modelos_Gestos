import cv2
import mediapipe as mp
import pickle
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Cargar modelo entrenado
with open("Modelo3/modelo3_v1.pkl", "rb") as f:
    clf = pickle.load(f)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Umbral de confianza
THRESHOLD = 0.7

with mp_hands.Hands(static_image_mode=False, max_num_hands=2) as hands:
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibujar siempre los landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                data = []
                for lm in hand_landmarks.landmark:
                    data.extend([lm.x, lm.y, lm.z])

                # Probabilidades de predicción
                probs = clf.predict_proba([data])[0]
                pred_index = np.argmax(probs)
                confidence = probs[pred_index]

                # Solo mostrar texto si confianza suficiente
                if confidence >= THRESHOLD:
                    pred_label = clf.classes_[pred_index]
                    cv2.putText(frame, f"Gesto: {pred_label}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Cam', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

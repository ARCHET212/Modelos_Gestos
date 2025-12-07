from fastapi import FastAPI, UploadFile, File
import cv2
import mediapipe as mp
import numpy as np
import pickle
from PIL import Image
import io

app = FastAPI()

# Cargar modelo entrenado
with open("modelo_lsg_estatico_dataset0.pkl", "rb") as f:
    clf = pickle.load(f)

# Inicializar mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

THRESHOLD = 0.6

@app.get("/")
def root():
    return {"message": "Servidor de reconocimiento de señas activo 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Leer imagen recibida
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(img)

    # Procesar con Mediapipe
    results = hands.process(img_array)

    if not results.multi_hand_landmarks:
        return {"error": "No se detectó ninguna mano"}

    # Usamos solo la primera mano detectada
    hand_landmarks = results.multi_hand_landmarks[0]

    data = []
    for lm in hand_landmarks.landmark:
        data.extend([lm.x, lm.y, lm.z])

    # Predicción
    probs = clf.predict_proba([data])[0]
    pred_index = np.argmax(probs)
    confidence = probs[pred_index]

    print("Predicción:", clf.classes_[pred_index], "Confianza:", confidence)

    if confidence < THRESHOLD:
        return {"prediction": "Desconocido", "confidence": float(confidence)}

    pred_label = clf.classes_[pred_index]
    return {"prediction": str(pred_label), "confidence": float(confidence)}

#python -m uvicorn server:app --reload --host 0.0.0.0 --port 8000
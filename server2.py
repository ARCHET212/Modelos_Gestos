from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pickle
import cv2
import mediapipe as mp
import numpy as np
import tempfile

# Cargar modelo
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# Inicializar mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'E' ,
               4: 'K', 5: 'L', 6: 'M', 7: 'N' ,
               8: 'O', 9: 'R', 10: 'U', 11: 'V' ,
               12: 'W', 13: 'Y'}

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name

    frame = cv2.imread(temp_path)
    if frame is None:
        return JSONResponse({"error": "No se pudo leer la imagen"}, status_code=400)

    data_aux = []
    x_ = []
    y_ = []

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
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

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]
        return {"prediction": predicted_character}

    return {"prediction": None}

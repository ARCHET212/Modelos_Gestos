from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import cv2
import mediapipe as mp
import numpy as np
import pickle
import io


app = FastAPI(title="API Modelo 3 - Detección de Gestos por Imagen (Coordenadas Absolutas)")


MODELO_PATH = "Modelo3/modelo3_v1.pkl"
EXPECTED_FEATURES = 63  # 21 landmarks * (x, y, z)

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
# Usamos un objeto Hands para procesar las imágenes estáticas que recibimos
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Carga del modelo (se ejecuta una sola vez al iniciar la API)
clf = None
try:
    with open(MODELO_PATH, "rb") as f:
        clf = pickle.load(f)
    print(f"Modelo 3 cargado exitosamente desde: {MODELO_PATH}")
except FileNotFoundError:
    print(f"ERROR CRÍTICO: Archivo de modelo no encontrado en {MODELO_PATH}. La API no funcionará.")
except Exception as e:
    print(f"ERROR CRÍTICO al cargar el modelo: {e}")


def process_image_and_predict(image_bytes: bytes):
    """
    Decodifica la imagen, detecta la mano con MediaPipe, extrae las features (Modelo 3: x,y,z absolutas)
    y realiza la predicción.
    """
    if clf is None:
        raise HTTPException(status_code=503, detail="El modelo no está cargado. Revise los logs del servidor.")

    # 1. Convertir bytes a imagen OpenCV (BGR)
    np_array = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen. Asegúrese de que sea un formato válido (JPG, PNG).")

    # 2. Convertir BGR a RGB para MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 3. Procesar con MediaPipe
    results = hands.process(frame_rgb)

    if not results.multi_hand_landmarks:
        # No se detectó ninguna mano
        return {
            "success": False,
            "prediction": None,
            "confidence": 0.0,
            "message": "No se detectó ninguna mano en la imagen."
        }
    
    # 4. Extraer las 63 features (Coordenadas Absolutas para el Modelo 3)
    data = []
    # Usamos la primera mano detectada
    hand_landmarks = results.multi_hand_landmarks[0] 
    
    for lm in hand_landmarks.landmark:
 
        data.extend([lm.x, lm.y, lm.z])

    # 5. Validación de features (deberían ser 63)
    if len(data) != EXPECTED_FEATURES:
        raise HTTPException(status_code=500, detail=f"Error interno: Se esperaba {EXPECTED_FEATURES} features, pero se obtuvieron {len(data)}. Revise la implementación de MediaPipe.")

    # 6. Predicción
    try:
        
        probs = clf.predict_proba([data])[0]
        pred_index = np.argmax(probs)
        confidence = probs[pred_index]
        pred_label = clf.classes_[pred_index]

        return {
            "success": True,
            "prediction": pred_label,
            "confidence": round(confidence, 4),
            "model_type": "Modelo 3 (Absoluto - Por Imagen)"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la predicción del modelo: {str(e)}")


# --- 3. ENDPOINT ---
@app.post("/predict", summary="Predice el gesto de una mano enviada como imagen.")
async def predict_image(file: UploadFile = File(..., description="Imagen de la mano (JPG o PNG).")):
    """
    Recibe un archivo de imagen, procesa los landmarks de la mano, y devuelve el gesto predicho.
    """
    
    image_bytes = await file.read()
    
   
    result = process_image_and_predict(image_bytes)
    
    return JSONResponse(content=result)



if __name__ == '__main__':
    import uvicorn
    print("\n--- INSTRUCCIONES ---")
    print("Para iniciar el servidor, ejecute el siguiente comando en su terminal:")
    print("uvicorn api_modelo3_imagen:app --reload")
    print("Luego, acceda a http://127.0.0.1:8000/docs para ver la documentación y probar el endpoint con el 'Try it out'.")
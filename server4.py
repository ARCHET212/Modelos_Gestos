from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pickle
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
import shutil # Para manejar archivos temporales

# ¡IMPORTANTE! MODIFIQUE ESTA RUTA/NOMBRE SI ES NECESARIO
MODELO_PKL = "Modelo4/modelo4_v3.pkl"
THRESHOLD = 0.7 
# ****************************************************************************

# Inicialización de la aplicación FastAPI
app = FastAPI(title="Servidor de Reconocimiento de Gestos LSC (83 Features)")

# Inicialización de MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Variables globales para el modelo
clf = None
hands_processor = None



def calcular_angulo(p1, p2, p3):
    """Calcula el ángulo en grados formado por 3 puntos (p1, p2, p3) con el vértice en p2."""
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    v1 = p1 - p2
    v2 = p3 - p2
    
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    angle_rad = np.arccos(cosine_angle)
    return np.degrees(angle_rad)

def normalizar_landmarks(landmarks):
    """
    Procesa los 21 landmarks de MediaPipe para extraer las 83 features:
    63 (XYZ normalizados) + 5 (distancias tips) + 15 (ángulos).
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    wrist = coords[0]
    
    # 1. Normalización de Traslación y Escala
    coords_trasladas = coords - wrist
    # Usamos la distancia 0 a 9 (muñeca a base del dedo medio) como escala
    escala = np.linalg.norm(coords_trasladas[9]) 
    coords_normalizadas = coords_trasladas / (escala if escala > 0.001 else 1.0)
    
    datos_base = coords_normalizadas.flatten() # 63 features
    
    # 2. Extracción de 5 Distancias a Puntas de Dedos (Punta a muñeca normalizada)
    tips_indices = [4, 8, 12, 16, 20]
    distancias_tips = [np.linalg.norm(coords_normalizadas[i]) for i in tips_indices]
    
    # 3. Extracción de 15 Ángulos de Bending (Articulaciones)
    # Tenga en cuenta que sus ángulos están definidos en su script de entrenamiento
    angulos_puntos = [
        # Pulgar (0, 1, 2)
        (coords_normalizadas[0], coords_normalizadas[1], coords_normalizadas[2]), 
        (coords_normalizadas[1], coords_normalizadas[2], coords_normalizadas[3]),
        (coords_normalizadas[2], coords_normalizadas[3], coords_normalizadas[4]), 
        
        # Índice (5, 6, 7)
        (coords_normalizadas[5], coords_normalizadas[6], coords_normalizadas[7]),
        (coords_normalizadas[6], coords_normalizadas[7], coords_normalizadas[8]),
        (coords_normalizadas[0], coords_normalizadas[5], coords_normalizadas[6]), # Angulo entre la base del indice y muñeca
        
        # Medio (9, 10, 11)
        (coords_normalizadas[9], coords_normalizadas[10], coords_normalizadas[11]),
        (coords_normalizadas[10], coords_normalizadas[11], coords_normalizadas[12]),
        (coords_normalizadas[0], coords_normalizadas[9], coords_normalizadas[10]), # Angulo entre la base del medio y muñeca
        
        # Anular (13, 14, 15)
        (coords_normalizadas[13], coords_normalizadas[14], coords_normalizadas[15]),
        (coords_normalizadas[14], coords_normalizadas[15], coords_normalizadas[16]),
        (coords_normalizadas[0], coords_normalizadas[13], coords_normalizadas[14]), # Angulo entre la base del anular y muñeca
        
        # Meñique (17, 18, 19)
        (coords_normalizadas[17], coords_normalizadas[18], coords_normalizadas[19]),
        (coords_normalizadas[18], coords_normalizadas[19], coords_normalizadas[20]),
        (coords_normalizadas[0], coords_normalizadas[17], coords_normalizadas[18]) # Angulo entre la base del meñique y muñeca
    ]
    
    datos_angulares = [calcular_angulo(p1, p2, p3) for p1, p2, p3 in angulos_puntos]
    
    # 4. Concatenación: 63 (XYZ) + 5 (Distancias) + 15 (Ángulos) = 83 features
    datos_completos = np.concatenate((datos_base, distancias_tips, datos_angulares))
    
    return datos_completos


# --- 3. EVENTOS DE INICIO Y CIERRE DEL SERVIDOR ---

@app.on_event("startup")
async def load_model():
    """Carga el modelo de Machine Learning y el procesador de manos de MediaPipe al inicio."""
    global clf, hands_processor

    try:
        print(f"[*] Cargando modelo desde: {MODELO_PKL}...")
        with open(MODELO_PKL, "rb") as f:
            clf = pickle.load(f)
        print("[+] Modelo cargado exitosamente.")
        
      
        hands_processor = mp_hands.Hands(
            static_image_mode=True, # Modo de imagen estática para mejor detección inicial
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        print("[+] MediaPipe Hands inicializado.")

    except FileNotFoundError:
        print(f"!!! ERROR: No se encontró el modelo en la ruta: {MODELO_PKL}")
      
        exit(1) # Forzar la salida si el modelo no existe
    except Exception as e:
        print(f"!!! ERROR al cargar el modelo o inicializar MediaPipe: {e}")
        exit(1)

@app.on_event("shutdown")
async def shutdown_event():
    """Cierra recursos al apagar el servidor."""
    global hands_processor
    if hands_processor:
        hands_processor.close()
    print("[*] Servidor apagado. Recursos liberados.")


# --- 4. ENDPOINT PRINCIPAL ---

@app.post("/predict")
async def predict_gesture(file: UploadFile = File(...)):
    """
    Recibe una imagen (desde React Native), detecta la mano, extrae 83 features 
    y predice el gesto usando el modelo de Random Forest.
    """
    if clf is None:
        return JSONResponse(status_code=503, content={"error": "El modelo no está cargado o el servidor no se inicializó correctamente."})

    # Crear un archivo temporal para guardar la imagen
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Error al guardar la imagen temporal: {e}"})

    
    try:
        # Leer la imagen con OpenCV (BGR por defecto)
        image = cv2.imread(tmp_path)
        
        if image is None:
            return JSONResponse(status_code=400, content={"error": "El archivo subido no es una imagen válida."})

        # Convertir a RGB (requerido por MediaPipe)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Procesar la imagen con MediaPipe
        results = hands_processor.process(image_rgb)
        
        if results.multi_hand_landmarks:
            # Procesar solo la primera mano detectada
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # --- Extracción de Features y Predicción ---
            
            # 1. Normalizar a 83 features
            features = normalizar_landmarks(hand_landmarks.landmark)
            
            # 2. Predecir
            probs = clf.predict_proba([features])[0]
            pred_index = np.argmax(probs)
            confidence = float(probs[pred_index])
            
            # 3. Obtener la etiqueta (Gesto)
            # clf.classes_ contiene las etiquetas de las clases en el orden correcto
            pred_label = clf.classes_[pred_index]
            
            # 4. Aplicar Umbral de Confianza
            if confidence >= THRESHOLD:
                return JSONResponse(content={
                    "prediction": pred_label,
                    "confidence": round(confidence, 4)
                })
            else:
                return JSONResponse(content={
                    "prediction": "Desconocido",
                    "confidence": round(confidence, 4),
                    "message": "Predicción bajo umbral de confianza. Intente de nuevo."
                })

        else:
            # No se detectó ninguna mano en la imagen
            return JSONResponse(content={
                "prediction": "No-Mano",
                "confidence": 0.0,
                "message": "No se detectó ninguna mano en la imagen."
            })
            
    except Exception as e:
        print(f"Error en el procesamiento: {e}")
        return JSONResponse(status_code=500, content={"error": f"Error interno del servidor durante el procesamiento: {e}"})
        
    finally:
        # Asegurarse de eliminar el archivo temporal
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
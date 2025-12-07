import cv2
import mediapipe as mp
import pickle
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


MODELO_PKL = "Modelo4/modelo4_v3.pkl"
THRESHOLD = 0.7 


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

# --- FUNCIÓN DE NORMALIZACIÓN ROBUSTA CON DISTANCIAS Y ÁNGULOS (83 Features) ---
def normalizar_landmarks(landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    wrist = coords[0]
    
    # 1. Normalización de Traslación y Escala
    coords_trasladas = coords - wrist
    escala = np.linalg.norm(coords_trasladas[9])
    coords_normalizadas = coords_trasladas / (escala if escala > 0.001 else 1.0)
    
    datos_base = coords_normalizadas.flatten() # 63 features
    
    # 2. Extracción de 5 Distancias a Puntas de Dedos
    tips_indices = [4, 8, 12, 16, 20]
    distancias_tips = [np.linalg.norm(coords_normalizadas[i]) for i in tips_indices]
    
    # 3. Extracción de 15 Ángulos de Bending (Articulaciones)
    angulos_puntos = [
        # Pulgar (0, 1, 2, 3, 4)
        (coords_normalizadas[0], coords_normalizadas[1], coords_normalizadas[2]), 
        (coords_normalizadas[1], coords_normalizadas[2], coords_normalizadas[3]),
        (coords_normalizadas[2], coords_normalizadas[3], coords_normalizadas[4]), 
        
        # Índice (5, 6, 7, 8)
        (coords_normalizadas[5], coords_normalizadas[6], coords_normalizadas[7]),
        (coords_normalizadas[6], coords_normalizadas[7], coords_normalizadas[8]),
        (coords_normalizadas[0], coords_normalizadas[5], coords_normalizadas[6]), 
        
        # Medio (9, 10, 11, 12)
        (coords_normalizadas[9], coords_normalizadas[10], coords_normalizadas[11]),
        (coords_normalizadas[10], coords_normalizadas[11], coords_normalizadas[12]),
        (coords_normalizadas[0], coords_normalizadas[9], coords_normalizadas[10]),
        
        # Anular (13, 14, 15, 16)
        (coords_normalizadas[13], coords_normalizadas[14], coords_normalizadas[15]),
        (coords_normalizadas[14], coords_normalizadas[15], coords_normalizadas[16]),
        (coords_normalizadas[0], coords_normalizadas[13], coords_normalizadas[14]),
        
        # Meñique (17, 18, 19, 20)
        (coords_normalizadas[17], coords_normalizadas[18], coords_normalizadas[19]),
        (coords_normalizadas[18], coords_normalizadas[19], coords_normalizadas[20]),
        (coords_normalizadas[0], coords_normalizadas[17], coords_normalizadas[18])
    ]
    
    datos_angulares = [calcular_angulo(p1, p2, p3) for p1, p2, p3 in angulos_puntos]
    
    # 4. Concatenación: 63 + 5 + 15 = 83 features
    datos_completos = np.concatenate((datos_base, distancias_tips, datos_angulares))
    
    return datos_completos


try:
    # Cargar modelo entrenado
    with open(MODELO_PKL, "rb") as f:
        clf = pickle.load(f)
except FileNotFoundError:
    print(f"Error: No se encontró el modelo en {MODELO_PKL}. Asegúrese de haber ejecutado el script de entrenamiento.")
    exit()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with mp_hands.Hands(static_image_mode=False, max_num_hands=1) as hands:
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        
                data = normalizar_landmarks(hand_landmarks.landmark)

            
                probs = clf.predict_proba([data])[0]
                pred_index = np.argmax(probs)
                confidence = probs[pred_index]

                if confidence >= THRESHOLD:
                    pred_label = clf.classes_[pred_index]
                    cv2.putText(frame, f"Gesto: {pred_label} ({confidence:.2f})", 
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Camara - Reconocimiento (Modelo 4 v3: 83 features)', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
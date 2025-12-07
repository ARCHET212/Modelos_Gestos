import cv2
import mediapipe as mp
import csv
import os
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- RUTAS Y CONFIGURACIÓN DEL MODELO 4 (v3: 83 FEATURES) ---
ARCHIVO_CSV = "Modelo4/modelo4_v3.csv"
FRAMES_POR_RAFAGA = 300 

# --- FUNCIÓN DE UTILIDAD PARA CALCULAR ÁNGULO EN 3D ---
def calcular_angulo(p1, p2, p3):
    """Calcula el ángulo en grados formado por 3 puntos (p1, p2, p3) con el vértice en p2."""
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    v1 = p1 - p2
    v2 = p3 - p2
    
    # Evitar la división por cero o problemas numéricos
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0 # Retornar 0 si algún vector tiene longitud cero

    cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
    
    # Asegurar que el coseno esté en el rango [-1, 1]
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    angle_rad = np.arccos(cosine_angle)
    return np.degrees(angle_rad)

# --- FUNCIÓN DE NORMALIZACIÓN ROBUSTA CON DISTANCIAS Y ÁNGULOS (83 Features) ---
def normalizar_landmarks(landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    wrist = coords[0]
    
    # 1. Normalización de Traslación y Escala
    coords_trasladas = coords - wrist
    escala = np.linalg.norm(coords_trasladas[9]) # Muñeca (0) al Nudillo Medio (9)
    coords_normalizadas = coords_trasladas / (escala if escala > 0.001 else 1.0)
    
    # 63 features (x, y, z) normalizadas
    datos_base = coords_normalizadas.flatten()
    
    # 2. Extracción de 5 Distancias a Puntas de Dedos (respecto a la muñeca normalizada)
    tips_indices = [4, 8, 12, 16, 20]
    distancias_tips = [np.linalg.norm(coords_normalizadas[i]) for i in tips_indices]
    
    # 3. Extracción de 15 Ángulos de Bending (Articulaciones)
    
    angulos_puntos = [
        # Pulgar (0, 1, 2, 3, 4)
        (coords_normalizadas[0], coords_normalizadas[1], coords_normalizadas[2]), # Ángulo 1
        (coords_normalizadas[1], coords_normalizadas[2], coords_normalizadas[3]), # Ángulo 2
        (coords_normalizadas[2], coords_normalizadas[3], coords_normalizadas[4]), # Ángulo 3
        
        # Índice (5, 6, 7, 8)
        (coords_normalizadas[5], coords_normalizadas[6], coords_normalizadas[7]),
        (coords_normalizadas[6], coords_normalizadas[7], coords_normalizadas[8]),
        (coords_normalizadas[0], coords_normalizadas[5], coords_normalizadas[6]), # Ángulo MCP, ref a muñeca
        
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
    
    # 4. Concatenación: 63 (XYZ) + 5 (Distancias) + 15 (Ángulos) = 83 features
    datos_completos = np.concatenate((datos_base, distancias_tips, datos_angulares))
    
    return datos_completos

# ----------------- INICIO DEL SCRIPT -----------------

# Crear directorio y CSV si no existe
if not os.path.exists("Modelo4"):
    os.makedirs("Modelo4")
if not os.path.exists(ARCHIVO_CSV):
    with open(ARCHIVO_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        
        # 63 columnas base
        base_header = [f"{i}_{coord}" for i in range(21) for coord in ("x","y","z")]
        # 5 columnas de distancia
        dist_header = [f"D_{i}" for i in [4, 8, 12, 16, 20]] 
        # 15 columnas de ángulo
        angle_header = [f"A_{i}" for i in range(15)]
        header = ["label"] + base_header + dist_header + angle_header # 83 COLUMNAS TOTALES
        
        writer.writerow(header)

cap = cv2.VideoCapture(0)

with mp_hands.Hands(static_image_mode=False, max_num_hands=1) as hands: 
    etiqueta_actual = None
    contador_rafagas = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        texto_info = f"Etiqueta actual: {etiqueta_actual or 'Ninguna'} | Ráfagas: {contador_rafagas}"
        cv2.putText(frame, texto_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow('Camara - Captura de Datos (Modelo 4 v3: 83 features)', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('e'):
            etiqueta_actual = input("Etiqueta para capturar: ")
            contador_rafagas = 0

        if key == ord('r') and etiqueta_actual:
            print(f"Capturando ráfaga de {FRAMES_POR_RAFAGA} frames para '{etiqueta_actual}'...")
            contador_rafagas += 1

            for i in range(FRAMES_POR_RAFAGA):
                ret, frame = cap.read()
                if not ret: break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # --- USAR LA FUNCIÓN DE NORMALIZACIÓN MEJORADA (83 FEATURES) ---
                        datos_normalizados = normalizar_landmarks(hand_landmarks.landmark)
                        
                        fila = [etiqueta_actual]
                        fila.extend(datos_normalizados) # Ahora son 83 features

                        with open(ARCHIVO_CSV, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(fila)

                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                cv2.imshow('Camara - Captura de Datos (Modelo 4 v3: 83 features)', frame)
                cv2.waitKey(10)

            print("Ráfaga completada.")

        if key == 27: break

cap.release()
cv2.destroyAllWindows()
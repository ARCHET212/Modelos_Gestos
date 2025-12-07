import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

ARCHIVO_CSV = "Modelo3/modelo3_v1.csv"
FRAMES_POR_RAFAGA = 300 

# Crear archivo CSV si no existe
if not os.path.exists(ARCHIVO_CSV):
    with open(ARCHIVO_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["label"] + [f"{i}_{coord}" for i in range(21) for coord in ("x","y","z")]
        writer.writerow(header)

cap = cv2.VideoCapture(0)

with mp_hands.Hands(static_image_mode=False, max_num_hands=2) as hands:
    etiqueta_actual = None
    contador_rafagas = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Mostrar info en pantalla
        texto_info = f"Etiqueta actual: {etiqueta_actual or 'Ninguna'} | Ráfagas: {contador_rafagas}"
        cv2.putText(frame, texto_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow('Cam', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('e'):  # Cambiar etiqueta
            etiqueta_actual = input("Etiqueta para capturar: ")
            contador_rafagas = 0

        if key == ord('r') and etiqueta_actual:  # Iniciar ráfaga
            print(f"Capturando ráfaga de {FRAMES_POR_RAFAGA} frames para '{etiqueta_actual}'...")
            contador_rafagas += 1

            for i in range(FRAMES_POR_RAFAGA):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        fila = [etiqueta_actual]
                        for lm in hand_landmarks.landmark:
                            fila.extend([lm.x, lm.y, lm.z])
                        with open(ARCHIVO_CSV, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(fila)

                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                cv2.imshow('Cam', frame)
                cv2.waitKey(100)  # 100ms entre frames

            print("Ráfaga completada.")

        if key == 27:  # ESC para salir
            break

cap.release()
cv2.destroyAllWindows()

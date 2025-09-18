import cv2
import numpy as np
import os

# Webcam local IP Webcam "http://<IP:PUERTO>/video"
cap = cv2.VideoCapture(0)

#rango colores
rangos_colores = {
    "Rojo": [np.array([0, 120, 70]), np.array([10, 255, 255])],
    "Verde": [np.array([40, 40, 40]), np.array([80, 255, 255])],
    "Azul": [np.array([90, 50, 50]), np.array([130, 255, 255])]
}

#CARPETA DE REFERENCIA
LIBRERIA_DIR = "libreria_objetos"
os.makedirs(LIBRERIA_DIR, exist_ok=True)

#FUNCIONES AUXILIARES
def cargar_referencias():
    """Carga todas las imágenes de la librería y extrae sus contornos"""
    referencias = {}
    for folder in os.listdir(LIBRERIA_DIR):
        path = os.path.join(LIBRERIA_DIR, folder)
        if os.path.isdir(path):
            referencias[folder] = []
            for file in os.listdir(path):
                img_path = os.path.join(path, file)
                ref_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if ref_img is None:
                    continue
                ref_img = cv2.resize(ref_img, (300, 300))
                _, ref_thresh = cv2.threshold(ref_img, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(ref_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    referencias[folder].append(contours[0])
    return referencias

def guardar_objeto(frame, nombre_objeto):
    """Guarda un frame en la carpeta del objeto dado"""
    path = os.path.join(LIBRERIA_DIR, nombre_objeto)
    os.makedirs(path, exist_ok=True)
    count = len(os.listdir(path)) + 1
    filename = os.path.join(path, f"{nombre_objeto}_{count}.png")
    cv2.imwrite(filename, frame)
    print(f"[INFO] Imagen guardada en {filename}")

#CARGAR REFERENCIAS 
referencias = cargar_referencias()

print("[INFO] Presiona 's' para guardar un objeto")
print("[INFO] Presiona 'q' para salir")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

 # Pprocesamiento color 
    for nombre_color, (lower, upper) in rangos_colores.items():
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:
                x, y, w, h = cv2.boundingRect(cnt)
                objeto_detectado = frame[y:y+h, x:x+w]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Comparación con todas las referencias
                mejor_match = None
                mejor_similitud = 1.0

                for nombre_objeto, contornos_ref in referencias.items():
                    for ref_contour in contornos_ref:
                        similarity = cv2.matchShapes(ref_contour, cnt, 1, 0.0)
                        if similarity < mejor_similitud:
                            mejor_similitud = similarity
                            mejor_match = nombre_objeto

                if mejor_match and mejor_similitud < 0.2:
                    texto = f"{nombre_color} - {mejor_match}"
                    color_texto = (255, 0, 0)
                else:
                    texto = f"{nombre_color} - No coincide"
                    color_texto = (0, 0, 255)

                cv2.putText(frame, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color_texto, 2)

        cv2.imshow(f"Mascara {nombre_color}", mask)

# Mostrar video procesado
    cv2.imshow("Video", frame)

#TECLAS
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        nombre = input("Nombre del objeto a guardar: ")
        guardar_objeto(frame, nombre)
        referencias = cargar_referencias()  # recargar referencias

cap.release()
cv2.destroyAllWindows()

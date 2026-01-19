# detector_letras_colores_serial.py
import cv2
import sqlite3
import pickle
import numpy as np
import time
from collections import Counter, deque

# opcional: pyserial; activar si vas a usar puerto serial real
try:
    import serial
except Exception:
    serial = None

# ==========================
# CONFIG
# ==========================
DB = "contours.db"
CAM_INDEX = 1            # cambia si tu cámara no es 1
SERIAL_PORT = "COM4"     # cambia si usas serial
BAUDRATE = 115200

K = 5
BUFFER_SIZE = 15
buffer_letras = deque(maxlen=BUFFER_SIZE)
buffer_color = deque(maxlen=BUFFER_SIZE)

# filtros contornos (los tuyos)
AREA_MIN = 200
AREA_MAX = 8000
RATIO_MIN = 0.3
RATIO_MAX = 2.0
MARGEN_BORDES = 10

# umbral para considerar "negro" en V
UMBRALES = {
    "v_negro": 40,
    "sat_color_min": 40,
    "val_min": 40
}

# ---------------------- serial (opcional) ----------------------
ser = None
if serial is not None:
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)
        time.sleep(0.05)
        print(f"[SERIAL] Abierto {SERIAL_PORT} @{BAUDRATE}")
    except Exception as e:
        print(f"[SERIAL] No se pudo abrir {SERIAL_PORT}: {e}")
        ser = None
else:
    print("[SERIAL] pyserial no disponible; funcionando sin serial.")

# ==========================
# BD: cargar plantillas (tu función)
# ==========================
def cargar_base():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT label, contour FROM templates")
    datos = c.fetchall()
    conn.close()

    base = []
    for label, contour_pickle in datos:
        contour = pickle.loads(contour_pickle)
        x, y, w, h = cv2.boundingRect(contour)
        c_norm = contour.copy()
        c_norm[:,0,0] -= x
        c_norm[:,0,1] -= y
        c_norm = c_norm.astype(np.float32)
        c_norm *= 64.0 / max(w,h)
        hu = cv2.HuMoments(cv2.moments(c_norm)).flatten()
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
        base.append((label, hu))
    print(f"[DB] Cargadas {len(base)} plantillas.")
    return base

# ==========================
# Tu función exacta para extraer contornos (letras)
# ==========================
def extraer_contorno(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV,11,2)
    kernel = np.ones((3,3), np.uint8)
    th = cv2.dilate(th, kernel, 1)
    th = cv2.erode(th, kernel, 1)
    contornos, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contornos) == 0:
        return None, th
    contornos_validos = []
    h_frame, w_frame = frame.shape[:2]
    for c in contornos:
        x,y,w,h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if h == 0: continue
        ratio = w/h
        if area < AREA_MIN or area > AREA_MAX:
            continue
        if ratio < RATIO_MIN or ratio > RATIO_MAX:
            continue
        if x < MARGEN_BORDES or y < MARGEN_BORDES or x+w > w_frame - MARGEN_BORDES or y+h > h_frame - MARGEN_BORDES:
            continue
        contornos_validos.append(c)
    if len(contornos_validos) == 0:
        return None, th
    c = max(contornos_validos, key=cv2.contourArea)
    return c, th

# ==========================
# Función para extraer contorno basado en color (si no hay borde negro)
# ==========================
def extraer_contorno_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, UMBRALES["sat_color_min"], UMBRALES["val_min"]])
    upper = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.medianBlur(mask, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    conts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not conts:
        return None, mask
    conts = [c for c in conts if cv2.contourArea(c) >= AREA_MIN]
    if not conts:
        return None, mask
    c = max(conts, key=cv2.contourArea)
    return c, mask

# ==========================
# KNN letras (tu función)
# ==========================
def comparar_knn(hu_actual, base):
    distancias = [(l, np.linalg.norm(hu_actual-h)) for l,h in base]
    distancias.sort(key=lambda x: x[1])
    vecinos = distancias[:K]
    letras = [v[0] for v in vecinos]
    letra_pred = Counter(letras).most_common(1)[0][0]
    porc = np.mean([max(0,100 - v[1]*25) for v in vecinos])
    dist_prom = np.mean([v[1] for v in vecinos])
    return letra_pred, porc, dist_prom, vecinos

# ==========================
# Clasificación color simple + confianza por saturación
# ==========================
def clasificar_color_por_hsv_patch(hsv_patch):
    if hsv_patch is None or hsv_patch.size == 0:
        return "DESCONOCIDO", 0.0
    h_mean = float(np.mean(hsv_patch[:,:,0]))
    s_mean = float(np.mean(hsv_patch[:,:,1]))
    v_mean = float(np.mean(hsv_patch[:,:,2]))

    if v_mean < UMBRALES["v_negro"]:
        return "NEGRO", 100.0
    if s_mean < 30 and v_mean > 200:
        return "BLANCO", 100.0
    if (h_mean <= 10 or h_mean >= 170) and s_mean > 60:
        return "ROJO", min(100.0, (s_mean/255.0)*100.0)
    if 20 <= h_mean <= 35 and s_mean > 60:
        return "AMARILLO", min(100.0, (s_mean/255.0)*100.0)
    if 36 <= h_mean <= 85 and s_mean > 50:
        return "VERDE", min(100.0, (s_mean/255.0)*100.0)

    return "DESCONOCIDO", min(100.0, (s_mean/255.0)*100.0)

# ==========================
# Envío por serial (nunca se rompe)
# ==========================
def enviar_serial(msg):
    global ser
    line = (msg + "\n").encode()
    if ser is not None:
        try:
            ser.write(line)
        except Exception:
            print("[SERIAL] Error enviando; ignorando y continuando.")
            ser = None  # se desconectó → continuar sin romper
    print("[OUT]", msg)

# ==========================
# MAIN
# ==========================
def main():
    base = cargar_base()
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir cámara index {CAM_INDEX}")
        return

    print("[INFO] Iniciado. Presiona '1' para forzar obtener_color, 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        cont_letra, mask_letra = extraer_contorno(frame)

        cont_color = None
        mask_color = None
        if cont_letra is None:
            cont_color, mask_color = extraer_contorno_color(frame)

        mask_show = mask_color if (mask_color is not None) else mask_letra

        x=y=w=h=None
        if cont_letra is not None:
            x,y,w,h = cv2.boundingRect(cont_letra)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        elif cont_color is not None:
            x,y,w,h = cv2.boundingRect(cont_color)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        cmd = ""
        if ser is not None:
            try:
                raw = ser.readline()
                if raw:
                    cmd = raw.decode(errors="ignore").strip()
            except Exception:
                cmd = ""

        key = cv2.waitKey(1) & 0xFF
        if key == ord("1"):
            cmd = "obtener_color"

        if cmd == "obtener_color":
            if cont_letra is not None:
                x,y,w,h = cv2.boundingRect(cont_letra)
                roi = frame[y:y+h, x:x+w]
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                v_mean = float(np.mean(hsv_roi[:,:,2]))

                if v_mean < UMBRALES["v_negro"]:
                    c_norm = cont_letra.copy()
                    c_norm[:,0,0] -= x
                    c_norm[:,0,1] -= y
                    c_norm = c_norm.astype(np.float32)
                    c_norm *= 64.0 / max(w,h)
                    hu = cv2.HuMoments(cv2.moments(c_norm)).flatten()
                    hu = -np.sign(hu)*np.log10(np.abs(hu)+1e-12)

                    letra_top, porc_top, dist_top, _ = comparar_knn(hu, base)
                    if dist_top <= 4:
                        resultado = letra_top
                        confianza = porc_top
                    else:
                        resultado = "NEGRO"
                        confianza = 100.0
                else:
                    resultado, confianza = clasificar_color_por_hsv_patch(hsv_roi)

            elif cont_color is not None:
                x,y,w,h = cv2.boundingRect(cont_color)
                roi = frame[y:y+h, x:x+w]
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                resultado, confianza = clasificar_color_por_hsv_patch(hsv_roi)

            else:
                resultado = "BLANCO"
                confianza = 100.0

            enviar_serial(f"{resultado}|{confianza:.1f}")

        if x is not None:
            roi_tmp = frame[y:y+h, x:x+w]
            if roi_tmp is not None and roi_tmp.size != 0:
                hsv_tmp = cv2.cvtColor(roi_tmp, cv2.COLOR_BGR2HSV)
                lab_tmp, conf_tmp = clasificar_color_por_hsv_patch(hsv_tmp)
                cv2.putText(frame, f"{lab_tmp} {conf_tmp:.0f}%", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)

        cv2.imshow("Camara", frame)
        if mask_show is None:
            blank = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            cv2.imshow("Mascara", blank)
        else:
            cv2.imshow("Mascara", mask_show)

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ============================================================
# *** FINAL CORRECTO ***
# ============================================================
if __name__ == "__main__":
    main()
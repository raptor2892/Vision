import cv2
import sqlite3
import pickle
import numpy as np
from collections import Counter, deque

DB = "contours.db"
K = 5
BUFFER_SIZE = 15
buffer_letras = deque(maxlen=BUFFER_SIZE)

# Filtros
AREA_MIN = 200
AREA_MAX = 8000
RATIO_MIN = 0.3
RATIO_MAX = 2.0
MARGEN_BORDES = 10  # px para ignorar contornos pegados al borde

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
    return base

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
        ratio = w/h
        if area < AREA_MIN or area > AREA_MAX:
            continue
        if ratio < RATIO_MIN or ratio > RATIO_MAX:
            continue
        if x<MARGEN_BORDES or y<MARGEN_BORDES or x+w>w_frame-MARGEN_BORDES or y+h>h_frame-MARGEN_BORDES:
            continue
        contornos_validos.append(c)
    if len(contornos_validos)==0:
        return None, th
    c = max(contornos_validos, key=cv2.contourArea)
    return c, th

def comparar_knn(hu_actual, base):
    distancias = [(l, np.linalg.norm(hu_actual-h)) for l,h in base]
    distancias.sort(key=lambda x: x[1])
    vecinos = distancias[:K]
    letras = [v[0] for v in vecinos]
    letra_pred = Counter(letras).most_common(1)[0][0]
    porc = np.mean([max(0,100 - v[1]*25) for v in vecinos])
    dist_prom = np.mean([v[1] for v in vecinos])
    return letra_pred, porc, dist_prom, vecinos

def main():
    base = cargar_base()
    cam = cv2.VideoCapture(1)
    print("[INFO] Detector PRO filtrado listo. Q para salir.")

    while True:
        ret, frame = cam.read()
        if not ret:
            continue
        contorno, th = extraer_contorno(frame)
        letra_actual = "ERROR"
        porc_actual = 0
        if contorno is not None:
            x,y,w,h = cv2.boundingRect(contorno)
            c_norm = contorno.copy()
            c_norm[:,0,0] -= x
            c_norm[:,0,1] -= y
            c_norm = c_norm.astype(np.float32)
            c_norm *= 64.0/max(w,h)
            hu = cv2.HuMoments(cv2.moments(c_norm)).flatten()
            hu = -np.sign(hu)*np.log10(np.abs(hu)+1e-12)
            letra_top, porc_top, dist_top, vecinos = comparar_knn(hu, base)
            if dist_top>4:
                letra_top="ERROR"
                porc_top=0
            buffer_letras.append(letra_top)
            letra_actual = Counter(buffer_letras).most_common(1)[0][0]
            porc_actual = porc_top
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.drawContours(frame,[contorno],-1,(0,255,0),2)
            cv2.putText(frame,f"{letra_actual} | {porc_actual:.1f}%",(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv2.imshow("Camara", frame)
        cv2.imshow("Mascara", th)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()

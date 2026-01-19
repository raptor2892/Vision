import cv2
import numpy as np
from collections import deque, Counter
import tkinter as tk
from PIL import Image, ImageTk

# ---------- CONFIG ----------
cam_index = 1
PATCH_MIN_AREA = 500
SAT_MIN = 40
VAL_MIN = 40
BUFFER_SIZE = 10
label_buffer = deque(maxlen=BUFFER_SIZE)

# ---------- Mapeo HSV → Nombre ----------
def nombre_color(h, s, v):
    if v < 40:
        return "negro"
    if s < 30 and v > 200:
        return "blanco"
    if (h <= 10 or h >= 170) and s > 60:
        return "rojo"
    if 11 <= h <= 25 and s > 60:
        return "naranja"
    if 26 <= h <= 35 and s > 60:
        return "amarillo"
    if 36 <= h <= 85 and s > 50:
        return "verde"
    if 86 <= h <= 140 and s > 50:
        return "azul"
    if 141 <= h <= 170 and s > 50:
        return "morado"
    return "desconocido"

# ---------- Tkinter + OpenCV ----------
class App:
    def __init__(self):
        self.cap = cv2.VideoCapture(cam_index)
        self.root = tk.Tk()
        self.root.title("Detector de Color con Bounding Box")

        self.lframe = tk.Label(self.root)
        self.lframe.pack()

        self.update_frame()
        self.root.mainloop()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_frame)
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # máscara general de "color"
        lower = np.array([0, SAT_MIN, VAL_MIN])
        upper = np.array([179, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        if contours:
            best = max(contours, key=cv2.contourArea)
            if cv2.contourArea(best) < PATCH_MIN_AREA:
                best = None

        label = "Ninguno"

        if best is not None:
            x, y, w, h = cv2.boundingRect(best)

            # recortar patch
            patch = frame[y:y+h, x:x+w]
            hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

            # HSV promedio
            h_mean, s_mean, v_mean = np.mean(hsv_patch.reshape(-1, 3), axis=0)

            # nombre
            label = nombre_color(int(h_mean), int(s_mean), int(v_mean))

            # dibujar bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,255), 2)
            cv2.putText(frame, f"{label}",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0,255,255), 2)

        # suavizado
        label_buffer.append(label)
        stable = Counter(label_buffer).most_common(1)[0][0]

        # texto final
        cv2.putText(frame, f"Detectado: {stable}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255,255,255), 2)

        # convertir a Tkinter
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.lframe.imgtk = img
        self.lframe.configure(image=img)

        self.root.after(1, self.update_frame)

App()

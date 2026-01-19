import cv2
import numpy as np
import sqlite3
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

DB_NAME = "colores.db"

# ----------------- BASE DE DATOS -----------------
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS colores(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            etiqueta TEXT,
            h INTEGER,
            s INTEGER,
            v INTEGER
        )
    """)
    conn.commit()
    conn.close()

def guardar_hsv(etiqueta, h, s, v):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO colores(etiqueta, h, s, v) VALUES (?, ?, ?, ?)",
              (etiqueta, int(h), int(s), int(v)))
    conn.commit()
    conn.close()

# ----------------- DETECTOR DE COLOR -----------------
# Rango "colorido" general para aislar el objeto
SAT_MIN = 40
VAL_MIN = 40

def obtener_cuadro_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # máscara simple para colores (no blanco/negro)
    mask = cv2.inRange(hsv, (0, SAT_MIN, VAL_MIN), (179, 255, 255))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, None

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 400:
        return None, None, None, None

    x, y, w, h = cv2.boundingRect(c)

    return x, y, w, h, mask

def obtener_hsv_centro(frame, x, y, w, h):
    cx = x + w//2
    cy = y + h//2
    pixel = frame[cy, cx]
    hsv = cv2.cvtColor(np.uint8([[pixel]]), cv2.COLOR_BGR2HSV)
    h, s, v = hsv[0][0]
    return h, s, v, (cx, cy)

# ----------------- TKINTER -----------------
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Captura HSV (200 por color)")

        init_db()

        self.etiqueta = tk.StringVar(value="rojo")
        self.capturando = False
        self.contador = 0

        ttk.Label(root, text="Color actual:").pack()

        ttk.Combobox(root, textvariable=self.etiqueta,
                     values=["rojo", "verde", "amarillo"]).pack()

        ttk.Button(root, text="Iniciar captura (200)",
                   command=self.iniciar_captura).pack(pady=10)

        self.video_label = ttk.Label(root)
        self.video_label.pack()

        self.cap = cv2.VideoCapture(1)
        self.actualizar_video()

    def iniciar_captura(self):
        self.capturando = True
        self.contador = 0
        print(f"[INFO] Capturando 200 HSV para '{self.etiqueta.get()}' ...")

    def actualizar_video(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.actualizar_video)
            return

        x, y, w, h, mask = obtener_cuadro_color(frame)

        if x is not None:
            # dibujar cuadro
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,255), 2)

            # pixel central
            h_c, s_c, v_c, (cx, cy) = obtener_hsv_centro(frame, x, y, w, h)
            cv2.circle(frame, (cx, cy), 4, (0,0,255), -1)

            # CAPTURA AUTOMÁTICA
            if self.capturando and self.contador < 200:
                guardar_hsv(self.etiqueta.get(), h_c, s_c, v_c)
                self.contador += 1
                print(f"[{self.etiqueta.get()}] HSV {self.contador}/200  -> {h_c}, {s_c}, {v_c}")

                if self.contador == 200:
                    print("✔ Captura completa")
                    self.capturando = False

        # convertir para TK
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(img))
        self.video_label.imgtk = img
        self.video_label.configure(image=img)

        self.root.after(10, self.actualizar_video)

# ----------------- MAIN -----------------
root = tk.Tk()
app = App(root)
root.mainloop()

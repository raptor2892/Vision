import sqlite3

DB = "contours.db"

def crear_base():
    conn = sqlite3.connect(DB)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS templates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        label TEXT,
        contour BLOB,
        width INTEGER,
        height INTEGER
    )
    """)

    conn.commit()
    conn.close()
    print("✔ Base de datos creada correctamente.")

crear_base()

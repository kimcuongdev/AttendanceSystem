import sqlite3
import numpy as np
import base64
import cv2
from config import DB_PATH, SCHEMA_PATH, DB_DIR

def create_db():
    conn = sqlite3.connect(DB_PATH)
    with open(SCHEMA_PATH, 'r', encoding= 'utf-8') as f:
        schema_sql = f.read()
    conn.executescript(schema_sql)
    conn.commit()
    conn.close()

def insert_face(name, image, embedding):
    conn = sqlite3.connect(DB_PATH)

    # encode image -> base64
    _, buffer = cv2.imencode(".jpg", image)
    image_base64 = base64.b64encode(buffer).decode("utf-8")

    # embedding -> BLOB
    embedding_bytes = embedding.astype(np.float32).tobytes()

    with open(DB_DIR / 'insert_faces.sql', 'r', encoding= 'utf-8') as f:
        insert_sql = f.read() 

    conn.execute(insert_sql, (name, image_base64, embedding_bytes))
    conn.commit()
    conn.close()

def load_faces():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    with open(DB_DIR / "select_faces.sql", 'r', encoding= 'utf-8') as f:
        select_sql = f.read()
    c.execute(select_sql)
    rows = c.fetchall()
    conn.close()

    results = []
    for row in rows:
        id_, name, image_base64, embedding_blob = row
        image = cv2.imdecode(
            np.frombuffer(base64.b64decode(image_base64), np.uint8),
            cv2.IMREAD_COLOR
        )
        embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        results.append((id_, name, image, embedding))
    return results

if __name__ == "__main__":
    create_db()
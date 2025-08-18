from pathlib import Path

# Thư mục gốc của project
BASE_DIR = Path(__file__).resolve().parent

# Data
# Data Dir
DATA_DIR = BASE_DIR / "data"
# faces path
FACES_PATH = DATA_DIR / "faces"

# Đường dẫn database
DB_PATH = BASE_DIR / "database" / "face_recognition.db"
SCHEMA_PATH = BASE_DIR / "database" / "schema.sql"

# Đường dẫn model ArcFace (ONNX)
ARCFACE_MODEL_PATH = BASE_DIR / "data" / "models" / "w600k_r50.onnx"
# Đường dẫn detector (det_10g.onnx chẳng hạn)
FACE_DETECTOR_MODEL_PATH = BASE_DIR / "data" / "models" / "det_10g.onnx"
# LANDMARK
REFERENCE_LANDMARKS = [
    [38.2946, 51.6963], # left eye
    [73.5318, 51.5014], # right eye
    [56.0252, 71.7366], # nose tip
    [41.5493, 92.3655], # left mouth corner
    [70.7299, 92.2041], # right mouth corner
]
# Input det
INPUT_SIZE_DET = (640, 640)


# Ngưỡng nhận diện (cosine similarity)
RECOGNITION_THRESHOLD = 0.5

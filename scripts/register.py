import os
import cv2
import argparse
from database.database import create_db, insert_face
from face_detection.face_detector import FaceDetector
from face_embedding.extractor import Extractor
from config import FACES_PATH
FACES_DIR = FACES_PATH

def register_person(person_name, detector, extractor):
    detector = FaceDetector(min_detection_confidence= 0.7)
    extractor = Extractor()
    person_dir = os.path.join(FACES_DIR, person_name)
    if not os.path.isdir(person_dir):
        print(f"[WARN] Không tìm thấy folder cho {person_name}")
        return

    for filename in os.listdir(person_dir):
        img_path = os.path.join(person_dir, filename)
        image = cv2.imread(img_path)
        if image is None:
            continue

        detection_results = detector.detect_faces(image)
        if not detection_results:
            print(f"[WARN] Không phát hiện được khuôn mặt trong {img_path}")
            continue

        for detection in detection_results.detections:
            embedding = extractor.extract(image, detection)
            insert_face(person_name, image, embedding)
            print(f"[INFO] Đăng ký {person_name} từ {filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Đăng ký tất cả người trong data/faces")
    parser.add_argument("--name", type=str, help="Đăng ký cho 1 người cụ thể")
    args = parser.parse_args()

    detector = FaceDetector(min_detection_confidence=0.5)
    extractor = Extractor()

    if args.all:
        for person_name in os.listdir(FACES_DIR):
            if os.path.isdir(os.path.join(FACES_DIR, person_name)):
                register_person(person_name, detector, extractor)
    elif args.name:
        register_person(args.name, detector, extractor)
    else:
        print("❌ Bạn cần truyền --all hoặc --name <person_name>")


if __name__ == "__main__":
    main()

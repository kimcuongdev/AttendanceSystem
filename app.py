import cv2
import sys
from face_detection.face_detector import FaceDetector
from face_detection.preprocessor import Preprocessor

def main():
    # Load ảnh gốc
    image = cv2.imread('05022025.jpg')
    if image is None:
        print(f"Không thể load ảnh:")
        return

    # Khởi tạo detector + preprocessor
    detector = FaceDetector()
    preprocessor = Preprocessor(image_size=(112, 112))

    # Detect mặt
    faces = detector.detect_faces(image)

    if not faces:
        print("Không phát hiện khuôn mặt nào.")
        return

    for idx, face in enumerate(faces):
        landmarks = face['landmarks']
        bbox = face['bbox']
        # Align khuôn mặt đầu tiên
        aligned = preprocessor.align_face(image, landmarks)

        # Vẽ bbox + landmarks trên ảnh gốc
        (x1, y1, x2, y2) = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for (lx, ly) in landmarks:
            cv2.circle(image, (int(lx), int(ly)), 2, (0, 0, 255), -1)

        # Show kết quả
        cv2.imshow("Original with detection", image)
        cv2.imshow(f"Aligned Face {idx+1}", aligned)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

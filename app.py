import cv2
from face_detection.face_detector import FaceDetector
from face_detection.preprocessor import Preprocessor
from face_embedding.extractor import Extractor
from face_recognition.retriver import Retriever
from face_recognition.recognizer import Recognizer
from face_recognition.similarity import Cosine_Similarity
from ui.drawer import Drawer

def main():
    cap = cv2.VideoCapture(0)
    detector = FaceDetector(min_detection_confidence= 0.7)
    extractor = Extractor()
    retriever = Retriever()
    similarity = Cosine_Similarity()
    recognizer = Recognizer(similarity)
    all_faces = retriever.retrieve(image= True)
    drawer = Drawer()
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Camera Error')
            break
        detection_results = detector.detect_faces(frame)
        detection_drawing = detector.draw_faces(frame, detection_results)
        # cv2.imshow("Face Detection", cv2.flip(detection_drawing, 1))
        if not detection_results:
            print('Face not found')
            break
        for detection in detection_results.detections:
            vector_embedding = extractor.extract(frame, detection)
            best_score, best_item = recognizer.recognize(vector_embedding, all_faces)
            # cv2.imshow("Best Similarity", cv2.flip(best_item['image'], 1))
            print(f"Name: {best_item['name']}, Score: {best_score}")
            result = drawer.draw_item(frame, best_item, detection, best_score)
            cv2.imshow("App", result)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


from face_embedding.arcface_model import ArcFaceModel
from face_detection.preprocessor import Preprocessor
class Extractor:
    def __init__(self):
        self.embedding_model = ArcFaceModel()
        self.preprocessor = Preprocessor()

    def extract(self, image, detection):
        """
        Input: raw image, detection
        Output: embedding face as detection
        """
        preprocessed_image = self.preprocessor.preprocessing(image, detection)
        embedding = self.embedding_model.get_embedding(preprocessed_image)
        return embedding
    
# test
if __name__ == "__main__":
    import cv2
    from face_detection.face_detector import FaceDetector
    cap = cv2.VideoCapture(0)
    detector = FaceDetector(min_detection_confidence= 0.5)
    extractor = Extractor()
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Loi doc camera')
            break
        detection_results = detector.detect_faces(frame)
        detection_drawing = detector.draw_faces(frame, detection_results)
        cv2.imshow("Face Detection", cv2.flip(detection_drawing, 1))
        if not detection_results:
            print('Face not found')
            break
        for detection in detection_results.detections:
            vector_embedding = extractor.extract(frame, detection)
            # print(vector_embedding.shape)
            print(vector_embedding[:5])
            print("-"*50)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
            

# face_detector.py
import cv2
import mediapipe as mp
from config import IMAGE_SHAPE

class FaceDetector:
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        """
        min_detection_confidence: ngưỡng tin cậy (0.0-1.0)
        model_selection:
            0 = dùng model cho khoảng cách gần (~2m)
            1 = dùng model cho khoảng cách xa (>2m)
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection= model_selection,
            min_detection_confidence = min_detection_confidence
        )

    def detect_faces(self, image):
        """
        Return: results
        + results.detections: list kết quả detection ứng với mỗi khuôn mặt phát hiện được
            + .score: điểm
            + location_data:
                + .relative_bounding_box: tọa độ normalize nằm trong khoảng [0,1] của bbox
                    + .xmin, .ymin: tọa độ góc trái trên
                        => x_min_pixel = xmin * image_width
                        => y_min_pixel = ymin * image_height
                    + .width, .height: chiều rộng, cao của bbox
                        => box_width = width * image_width
                        => box_height = height * image_height
                + .relative_keypoints: tọa độ normalize nằm trong khoảng [0,1] của keypoints trên khuôn mặt
                    + [0]: left eye
                    + [1]: right eye
                    + [2]: nose
                    + [3]: mouth
                    + [4]: left_ear
                    + [5]: right_ear

        """
        results = self.face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.detections:
            print('Faces not found')
            return
        return results
            

    def draw_faces(self, image, results):
        h, w, _ = image.shape
        if results.detections:
            for detection in results.detections:
                self.mp_drawing.draw_detection(image, detection)
                # keypoints = detection.location_data.relative_keypoints
                # left_eye     = (int(keypoints[0].x * w), int(keypoints[0].y * h))
                # right_eye    = (int(keypoints[1].x * w), int(keypoints[1].y * h))
                # nose         = (int(keypoints[2].x * w), int(keypoints[2].y * h))
                # mouth        = (int(keypoints[3].x * w), int(keypoints[3].y * h))
                # left_ear     = (int(keypoints[4].x * w), int(keypoints[4].y * h))
                # right_ear    = (int(keypoints[5].x * w), int(keypoints[5].y * h))
                # for pt in [left_eye, right_eye, nose, mouth, left_ear, right_ear]:
                #     cv2.circle(image, pt, 3, (0,0,255), -1)
        return image


# test
if __name__ == "__main__":
    from face_detection.preprocessor import Preprocessor
    preprocessor = Preprocessor()
    cap = cv2.VideoCapture(0)  # webcam
    detector = FaceDetector(min_detection_confidence=0.7)

    # detected = False

    while True:
        # if detected:
        #     break
        ret, frame = cap.read()
        if not ret:
            break

        detection_results = detector.detect_faces(frame) # results
        if not detection_results:
            print('Faces not found')
            break
        # detection_drawing = detector.draw_faces(frame, detection_results)

        # cv2.imshow("Face Detection", cv2.flip(detection_drawing,1))

        # view cropped faces, aligned_faces
        # for detection in detection_results.detections:
        #     bbox = detection.location_data.relative_bounding_box  # get bbox
        #     face_cropped = preprocessor.crop_face(frame, bbox)

        #     keypoints = detection.location_data.relative_keypoints
        #     left_eye  = (keypoints[0].x, keypoints[0].y)
        #     right_eye = (keypoints[1].x, keypoints[1].y)
        #     nose_tip = (keypoints[2].x, keypoints[2].y)

        #     aligned_face = preprocessor.align_face_3point(face_cropped, left_eye, right_eye, nose_tip)

        #     cv2.imshow("Face Detection", cv2.flip(aligned_face,1))

        # view crop + resize face:
        for detection in detection_results.detections:
            bbox = detection.location_data.relative_bounding_box
            face_cropped = preprocessor.crop_face(frame, bbox)
            face_resized = preprocessor.resize_face(face_cropped)
            cv2.imshow("Cropped Face: ", cv2.flip(face_cropped,1))
            cv2.imshow("ArcFace Input: ", cv2.flip(face_resized,1))
            # cv2.imwrite('output.jpg',face_resized)
            # detected = True
            # break

        if cv2.waitKey(1) & 0xFF == 27:  # ESC để thoát
            break

    cap.release()
    cv2.destroyAllWindows()

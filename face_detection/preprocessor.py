# preprocessor.py
import cv2
import numpy as np
from config import REFERENCE_LANDMARKS

class Preprocessor:
    def __init__(self, image_size=(112, 112)):
        self.image_size = image_size
        self.ref_landmarks = np.array(REFERENCE_LANDMARKS, dtype=np.float32)

    def align_face_3point(self, image, left_eye, right_eye, nose_tip):
        """
        Input:
            + image: ảnh đầu vào
            + left_eye: relative_key_point mắt trái
            + right_eye: relative_key_point mắt phải
            + nose_tip: relative_key_point mũi
            => chỉ có 3 point do dùng mediapipe không có khóe miệng trái phải
        Output:
            + ảnh được align theo chuẩn training set ArcFace
        """
        # image = cv2.resize(image, self.image_size)
        h, w, _ = image.shape
        left_eye = (int(left_eye[0] * w), int(left_eye[1] * h))
        right_eye = (int(right_eye[0] * w), int(right_eye[1] * h))
        nose_tip = (int(nose_tip[0] * w), int(nose_tip[1] * h))
        ref_3pts = self.ref_landmarks[:3] # lấy 3 point đầu
        src_3pts = np.float32([left_eye, right_eye, nose_tip])
        # Tính affine transform từ 3 điểm
        M = cv2.getAffineTransform(src_3pts, ref_3pts)

        # Warp về chuẩn ArcFace
        output_size = self.image_size
        aligned_image = cv2.warpAffine(image, M, output_size, flags=cv2.INTER_LINEAR)

        return aligned_image

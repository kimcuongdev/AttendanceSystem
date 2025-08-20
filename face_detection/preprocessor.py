# preprocessor.py
import cv2
import numpy as np
from config import REFERENCE_LANDMARKS, IMAGE_SHAPE

class Preprocessor:
    def __init__(self, image_size=(112, 112)):
        self.image_size = image_size
        self.ref_landmarks = np.array(REFERENCE_LANDMARKS, dtype=np.float32)

    def crop_face(self, image, bbox):
        """
        Input: image, bbox: [xmin, ymin, width, height] (normalized)
        Output: face crop from bbox
        """
        h, w, _ = image.shape
        xmin = int(bbox.xmin * w)
        ymin = int(bbox.ymin * h)
        box_w = int(bbox.width * w)
        box_h = int(bbox.height * h)

        x1 = max(0, xmin)
        y1 = max(0, ymin)
        x2 = min(w, xmin + box_w)
        y2 = min(h, ymin + box_h)

        face_crop = image[y1:y2, x1:x2]
        return face_crop

    def resize_face(self, face):
        face_resize = cv2.resize(face, IMAGE_SHAPE)
        # print(face_resize.shape)
        return face_resize

    def to_float32(self, face):
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32)
        return face

    def normalize_face(self, face):
        # [0; 255] -> [-1; 1]
        face = (face / 127.5) - 1.0
        return face
    
    def transpose_face(self, face):
        # H, W, C -> C, H, W
        face = np.transpose(face, (2, 0, 1))
        return face
    
    def expand_dim(self, face):
        # add batch dim
        face = np.expand_dims(face, axis= 0)
        return face

    def preprocessing(self, image, detection):
        bbox = detection.location_data.relative_bounding_box
        face_cropped = self.crop_face(image, bbox)
        face_resize = self.resize_face(face_cropped)
        face_float32 = self.to_float32(face_resize)
        face_normalize = self.normalize_face(face_float32)
        face_transpose = self.transpose_face(face_normalize)
        face_batch_dim = self.expand_dim(face_transpose)
        return face_batch_dim

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

import cv2

class Drawer:
    def draw_item(self, image, item, detection, score):
        """
        image: ảnh gốc (numpy array BGR)
        item: dict (ví dụ {'id': 1, 'name': 'Alice', 'embedding': ...})
        detection: đối tượng chứa bbox (từ mediapipe hoặc detector khác)
        score: độ tương đồng (float), optional
        """
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = image.shape
        xmin = int(bbox.xmin * w)
        ymin = int(bbox.ymin * h)
        box_w = int(bbox.width * w)
        box_h = int(bbox.height * h)

        x1 = max(0, xmin)
        y1 = max(0, ymin)
        x2 = min(w, xmin + box_w)
        y2 = min(h, ymin + box_h)

        # Vẽ bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Chuẩn bị text: name + score
        name = item.get("name", "Unknown")
        text = name
        if score is not None:
            text += f" ({score:.2f})"

        image = cv2.flip(image, 1)
        # Vẽ text trên ảnh
        cv2.putText(
            image,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        return image

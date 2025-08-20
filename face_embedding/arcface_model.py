from config import ARCFACE_MODEL_PATH
import onnxruntime as ort
import numpy as np

class ArcFaceModel:
    def __init__(self, model_path= ARCFACE_MODEL_PATH, providers= ["CPUExecutionProvider"]):
        self.session = ort.InferenceSession(model_path, providers= providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def get_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Input: ảnh đầu vào (1, H, W, C): batch, float32, normalized
        Output: vector embedding (1, 512)
        """
        if image.ndim == 3:
            image = np.expand_dims(image, axis= 0)
        embedding = self.session.run([self.output_name], {self.input_name: image})[0][0]
        return embedding
        
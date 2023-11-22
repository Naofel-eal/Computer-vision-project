from ultralytics import YOLO
from torch import cuda

class FaceDetector:
    def __init__(self, model_path: str = "weights/model.pt") -> None:
        self.model = YOLO(model_path)
        device = 'cuda' if cuda.is_available() else 'cpu'
        self.model.to(device)

    def detect(self, img):
        results = self.model.predict(img, show=True, save=False, verbose=False)
        original_image = results[0].orig_img
        bounding_boxes =results[0].boxes.xyxy.tolist()

    def crop(self, img, bounding_boxes: list):
        pass
from ultralytics import YOLO

class Model:
    def __init__(self, model_path: str = "model/yolov8/model.pt") -> None:
        self.model = YOLO(model_path)

    def detect(self, img):
        result = self.model.predict(img, show=True, save=False)[0].boxes.xyxy.tolist()
        print(result)
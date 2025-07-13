from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path, confidence):
        self.model = YOLO(model_path)
        self.classList = ["player"]
        self.confidence = confidence

    def detect(self, image):
        results = self.model.predict(image, conf=self.confidence)
        result = results[0]
        return self.make_detections(result)

    def make_detections(self, result):
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            class_id = int(box.cls[0])
            conf = box.conf[0]

            if result.names[class_id] == "player":
                detections.append(([x1, y1, w, h], class_id, conf))

        return detections

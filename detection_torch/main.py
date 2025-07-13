import cv2
from yolo_detector import YoloDetector
from tracker import PlayerTracker

yolo_model_path = 'models/best.pt'
reid_model_path = 'models/osnet_x1_0_imagenet.pth'
video_path = '15sec_input_720p.mp4'
output_path = 'output_annotated.mp4'

detector = YoloDetector(yolo_model_path, confidence=0.5)
tracker = PlayerTracker(reid_model_path)

cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)
    ids, boxes = tracker.track(detections, frame)

    for box, id in zip(boxes, ids):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    writer.write(frame)

cap.release()
writer.release()

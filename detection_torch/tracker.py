import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from reid_extractor import ReIDExtractor

class PlayerTracker:
    def __init__(self, reid_model_path):
        self.reid = ReIDExtractor(model_path=reid_model_path)
        self.deepsort = DeepSort(
            max_age=60,
            n_init=3,
            nn_budget=100,
            max_cosine_distance=0.4,
            embedder="torchreid",
            embedder_fn=self.reid.extract,
            bgr=True
        )

    def track(self, detections, frame):
        tracks = self.deepsort.update_tracks(detections, frame=frame)
        ids = []
        boxes = []

        for track in tracks:
            if not track.is_confirmed():
                continue
            ids.append(track.track_id)
            boxes.append(track.to_ltrb())

        return ids, boxes

import torch
from torchreid.utils import FeatureExtractor

class ReIDExtractor:
    def __init__(self, model_path, device='cuda'):
        self.extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path=model_path,
            device=device
        )

    def extract(self, crops):
        features = self.extractor(crops)
        return features.cpu().numpy()

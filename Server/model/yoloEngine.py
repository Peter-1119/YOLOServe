from ultralytics import YOLO
import torch
import numpy as np
class YOLOEngine:
    def __init__(self, weights_path: str):
        # Set model parameters for better performance
        self.model = YOLO(weights_path)
        self.model.conf = 0.25  # Confidence threshold
        self.model.iou = 0.45   # NMS IoU threshold
        self.model.max_det = 100  # Maximum number of detections

        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Model warmup
        self.warmup()
        
    def detect(self, image):
        results = self.model(image)
        return results
    
    def warmup(self):
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.model(dummy)
        print(f"Model warmup complete")

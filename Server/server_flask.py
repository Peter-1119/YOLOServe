import time
import json
import numpy as np

import cv2
from flask import Flask, request
from model.yoloEngine import YOLOEngine

# Initialize Flask app
app = Flask(__name__)

# Initialize YOLO model
model = YOLOEngine('model/yolov10n.pt')

# Define route for detecting objects in an image
@app.route('/detect', methods=['POST'])
def detect():
    # Start timing server processing
    serverProcessingStart = time.time()
    
    # Get image data and shape information from request
    image_data = request.files['file']
    shape_info = json.loads(request.files['shape'].read())
    
    # Decode image data
    nparr = np.frombuffer(image_data.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # img = np.frombuffer(image_data.read(), dtype = np.uint8).reshape(shape_info['shape'])
    
    # Run inference
    results = model.detect(image)
    
    # Process results
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            class_name = model.model.names[class_id]
            
            detections.append({
                "class": class_name,
                "confidence": confidence,
                "bbox": [float(x1), float(y1), float(x2), float(y2)]
            })
    
    # Return results
    return{
        'filename': image_data.filename,
        "detections": detections,
        'results': 'success',
        'serverProcessingTime': time.time() - serverProcessingStart,
        "device": model.device.type,
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

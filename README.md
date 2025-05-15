# YOLOv10 Object Detection RESTful API

## Description
This project implements a YOLOv10 object detection service using Flask and a corresponding client application for testing the detection service. The server processes incoming images using a pre-trained YOLOv10 model and returns detected objects with their bounding boxes, confidence scores, and class labels. The client application sends images to the server and processes the returned detection results.


## Project Structure
```
YOLOServe
│  install.bat              # Installation script
│  README.md                # Project documentation
│  requirements.txt         # Required Python packages
├─Client
│  │  client.py             # Client inference script
│  │  ResizeImage.py        # Image resizing script
│  │  utils.py              # Utility functions module
│  ├─Raw_images             # Original images folder (partial list shown)
│  │      street1.jpg
│  │      street2.png
│  │      street3.png
│  ├─Resize_images          # Resized images folder (partial list shown)
│  │      street1.jpg
│  │      street2.png
│  │      street3.png
│  └─output_images          # Output folder for detection results (created during runtime)
└─Server
    │  server_flask.py      # Server inference script
    └─model
        │  yoloEngine.py    # YOLO model wrapper module
        │  yolov10n.pt      # YOLOv10n model weights
```


## Installation Steps
1. Clone the repository:
```
git clone https://github.com/Peter-1119/YOLOServe .git
cd YOLOServe 
```

2. Create a virtual environment and activate it:
```
python -m venv env
source env/bin/activate  # On Windows, use: .\env\Scripts\activate
```

3. Install dependencies:
```
run install.bat
```
or: 
```
pip install -r requirements.txt
pip install git+https://github.com/THU-MIG/yolov10.git
```


# ⚡ Important: GPU Installation Notice
If you intend to use GPU acceleration, you do not need to install the CPU versions of OpenCV and PyTorch listed in requirements.txt.
For GPU-based installation:

1. Uninstall the CPU versions of OpenCV and PyTorch:
```
pip uninstall opencv-python torch
```

2. Install the GPU versions:
* CUDA 11.8:
```
pip install torch==2.1.0+cu118 torchvision==0.12.0+cu118 torchaudio==0.12.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

* CUDA 12.1:
```
pip install torch==2.1.0+cu121 torchvision==0.12.0+cu121 torchaudio==0.12.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

* OpenCV with CUDA:
    *   It is recommended to build OpenCV with CUDA support manually or install a precompiled version such as:
```
pip install opencv-python-headless==4.8.1.78
```
    *   Or use a third-party precompiled OpenCV with CUDA.



## Model and Data Preparation
* The pretrained YOLOv10 model (yolov10n.pt) should be placed in the model/ directory.
* Test images should be placed in the Resize_images/ directory. All images should have a resolution of 1280x720 to avoid additional processing overhead.


## Running the Server
To start the server, run:
```
python server.py
```
* The server will be hosted at http://127.0.0.1:8000.
* It will load the YOLOv10 model and wait for image requests.


## Client Usage
To test the server using the client application, run:
```
python client.py
```
* The client will send each image in the Resize_images/ folder to the server for detection.
* Processed images with bounding boxes will be saved in the output_images/ folder.
* The detection results and timing information will be printed to the console.


## Example API Request
* Endpoint: http://127.0.0.1:8000/detect
* Method: POST
* Payload:
** file: The image file to be processed.
** shape: JSON object containing the image dimensions.

Example Payload:
```
curl -X POST -F "file=@Resize_images/street9.png" -F "shape={\"shape\": [720, 1280, 3]}" http://127.0.0.1:8000/detect
```


## Output and Logs
* The processed images with bounding boxes will be saved in the output_images/ folder.
* The client will print detailed timing information for each step, including:
    *   Image loading time
    *   Optimization time
    *   Encoding time
    *   Network time
    *   Bounding box drawing time
    *   Image saving time


## Performance Evaluation
* Ensure that the system is equipped with a GPU for optimal inference performance.
* The target FPS is 10 FPS or higher, including network response time.


## Notes:
* Ensure that the GPU drivers and CUDA toolkit are correctly configured to enable GPU acceleration.
* The YOLOv10 model is optimized to prevent memory leaks and excessive memory usage during inference.
* Adjustments to the image size or encoding settings can further optimize the processing speed.
import time
import json
import os

import cv2
import requests
from utils import optimize_image

os.makedirs("output_images", exist_ok=True)

# Send image to server
def send_image(image_path: str, server_url: str):
    timing = {
        'load': 0,
        'optimize': 0,
        'encode': 0,
        'network': 0,
        'draw': 0,
        'save': 0,
        'total': 0
    }
    
    clientProcessingImageStart = time.time()
    
    # Read image
    image = cv2.imread(image_path)
    timing['load'] = time.time() - clientProcessingImageStart
    
    # Optimize image
    if image.shape[0] !=720 or image.shape[1] != 1280:
        resized_image = optimize_image(image, 1280, 720)
    else:
        resized_image = image
    timing['optimize'] = time.time() - clientProcessingImageStart
    
    # Compress image to bytes
    success, encoded_image = cv2.imencode('.jpg', resized_image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    image_data = encoded_image.tobytes()
    timing['encode'] = time.time() - clientProcessingImageStart
    
    # Set format for files
    files = {
        'file': (image_path, image_data, 'image/jpeg'),
        'shape': ('shape.json', json.dumps({'shape': resized_image.shape}), 'application/json'),
    }
    
    # Send image and shape to server
    clientTransferStart = time.time()
    response = requests.post(server_url, files=files)
    timing['network'] = time.time() - clientTransferStart
    result = response.json()
           
    # Convert back to BGR for OpenCV operations
    img = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
    
    # Draw bounding boxes on image
    draw_start = time.time()
    for detection in result['detections']:
        x1, y1, x2, y2 = map(int, detection['bbox'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{detection['class']} {detection['confidence']:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    timing['draw'] = time.time() - draw_start
    
    # Save annotated image
    save_start = time.time()
    image_name = image_path.split('/')[-1]
    output_path = f"output_images/{image_name[:-3]}.jpg"
    cv2.imwrite(output_path, img)
    timing['save'] = time.time() - save_start
    
    # Calculate total time
    timing['total'] = time.time() - clientProcessingImageStart

    # Print timing
    print(f"\nPicture {image_path} Timing Breakdown:")
    print(f"Client processing time: {timing['total']} seconds")
    print(f"Load image time: {timing['load']} seconds")
    print(f"Optimize image time: {timing['optimize']} seconds")
    print(f"Encode image time: {timing['encode']} seconds")
    print(f"Network time: {timing['network']} seconds")
    print(f"Draw bounding boxes time: {timing['draw']} seconds")
    print(f"Save annotated image time: {timing['save']} seconds")
    print(f"Total time: {timing['total']} seconds")
    
    return True, result['serverProcessingTime'], timing['network'], timing


# Main function
def main():
    # Set test image folder
    test_image_folder = 'Resize_images'
    
    # Set server URL
    server_url = 'http://127.0.0.1:8000/detect'
    
    # Warmup client
    _, _, _, _ = send_image("Resize_images/street9.png", server_url)
    
    # Initialize timing variables
    total_server_time = 0
    total_network_time = 0
    total_timing = {
        'load': 0,
        'optimize': 0,
        'encode': 0,
        'network': 0,
        'draw': 0,
        'save': 0,
        'total': 0
    }
    
    # Process all images in the test_images folder
    total_start_time = time.time()
    for image_file in os.listdir(test_image_folder):
        image_path = f'{test_image_folder}/{image_file}'
        success, server_time, network_time, timing = send_image(image_path, server_url)
        
        if success:
            total_server_time += server_time
            total_network_time += network_time
            for key in total_timing:
                total_timing[key] += timing[key]
    
    # Calculate total time
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    # Print timing summary
    print("\n" + "="*50)
    print("TIMING SUMMARY")
    print("="*50)
    print(f"Average processing time: {total_time / len(os.listdir(test_image_folder))} seconds")
    print(f"Average network time: {total_network_time / len(os.listdir(test_image_folder))} seconds")
    print(f"Average server processing time: {total_server_time / len(os.listdir(test_image_folder))} seconds")
    print(f"fps: {len(os.listdir(test_image_folder)) / total_time} fps")
    print(f"Total images: {len(os.listdir(test_image_folder))}")

if __name__ == '__main__':
    main()


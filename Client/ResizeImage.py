import cv2
import os
from utils import LetterBox

output_folder = "Resize_images"
os.makedirs(output_folder, exist_ok=True)

if __name__ == "__main__":
    image_folder = "Raw_images"
    image_path = [f"{image_folder}/{image_file}" for image_file in os.listdir(image_folder)]
    
    for image_path in image_path:
        image = cv2.imread(image_path)
        image = LetterBox(image, 1280, 720)
        cv2.imwrite(f"{output_folder}/{image_path.split('/')[-1]}", image)

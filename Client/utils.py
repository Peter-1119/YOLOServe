import cv2
import numpy as np

# LetterBox function
def LetterBox(Image, width: int, height: int):
    '''
    Args:
        Image (H * W * C): Numpy formated image
        ContainerSize (List): (Width, Height)

    Returns:
        Resized numpy formated image

    '''
    ImageContainer = np.zeros((height, width, 3), dtype = np.uint8)
    H, W, C = Image.shape

    scaled_ratio_W = width / W
    scaled_ratio_H = height / H
    scaled_ratio = scaled_ratio_H if scaled_ratio_H < scaled_ratio_W else scaled_ratio_W
    
    New_H, New_W = int(H * scaled_ratio), int(W * scaled_ratio)

    dh, dw = int((height - New_H) / 2), int((width - New_W) / 2)

    # im0 = cv2.resize(im0, (New_W, New_H), interpolation = cv2.INTER_LINEAR)  # yolov5 origin zoom in code
    # im0 = cv2.resize(im0, (New_W, New_H), interpolation = cv2.INTER_AREA)  # Slow but Performance best
    ResizeImage = cv2.resize(Image, (New_W, New_H), interpolation = cv2.INTER_NEAREST)  # fast but performance worst
    
    ImageContainer[dh: New_H + dh, dw: New_W + dw] = ResizeImage
    return ImageContainer

# Optimize image
def optimize_image(image: np.ndarray, width: int, height: int):
    resized_image = LetterBox(image, width, height)
    return cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
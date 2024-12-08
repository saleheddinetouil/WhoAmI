
import cv2
import numpy as np
from PIL import Image
import io

def load_image(uploaded_file):
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    return image

def convert_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def convert_to_pil(cv2_image):
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

def resize_image(image, max_size=800):
    height, width = image.shape[:2]
    if height > max_size or width > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height))
    return image
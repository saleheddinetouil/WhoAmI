
import cv2
import numpy as np
from pathlib import Path

class FaceDetector:
    def __init__(self, config):
        self.config = config
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.config['face_detection']['scale_factor'],
            minNeighbors=self.config['face_detection']['min_neighbors'],
            minSize=tuple(self.config['face_detection']['min_size'])
        )
        return faces
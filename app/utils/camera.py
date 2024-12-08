
import cv2
import streamlit as st
from threading import Thread
import time

class Camera:
    def __init__(self, config):
        self.config = config
        self.is_running = False
        self.camera = None

    def start(self):
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 
                          self.config['camera']['width'])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 
                          self.config['camera']['height'])
        self.is_running = True

    def stop(self):
        self.is_running = False
        if self.camera:
            self.camera.release()
            self.camera = None

    def get_frame(self):
        if self.camera and self.is_running:
            ret, frame = self.camera.read()
            if ret:
                return frame
        return None

    def __del__(self):
        self.stop()
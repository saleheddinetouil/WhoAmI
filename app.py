import streamlit as st
import cv2
import yaml
import numpy as np
from pathlib import Path
from PIL import Image
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.face_detector import FaceDetector
from app.utils.image_utils import load_image, convert_to_cv2
from app.core.face_recognizer import FaceRecognizer
from app.utils.camera import Camera

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def initialize():
    if 'face_detector' not in st.session_state:
        config = load_config()
        st.session_state.face_detector = FaceDetector(config)
        st.session_state.face_recognizer = FaceRecognizer(config)
        st.session_state.camera = Camera(config)
        st.session_state.config = config

def process_upload(uploaded_file):
    if uploaded_file is not None:
        image = load_image(uploaded_file)
        cv_image = convert_to_cv2(image)
        faces = st.session_state.face_detector.detect_faces(cv_image)
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        st.image(cv_image, channels="BGR", caption='Detected Faces')
        return len(faces)
    return 0

def process_camera():
    st.session_state.camera.start()
    stframe = st.empty()
    stop_button = st.button("Stop Camera")
    
    while not stop_button:
        frame = st.session_state.camera.get_frame()
        if frame is not None:
            faces = st.session_state.face_detector.detect_faces(frame)
            predictions = st.session_state.face_recognizer.recognize_faces(frame, faces)
            
            for name, (x, y, w, h) in predictions:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            stframe.image(frame, channels="BGR")
        
        if not st.session_state.camera.is_running:
            break
    
    st.session_state.camera.stop()

def train_face(name, image):
    faces = st.session_state.face_detector.detect_faces(image)
    if len(faces) == 1:
        x, y, w, h = faces[0]
        face_roi = image[y:y+h, x:x+w]
        return st.session_state.face_recognizer.add_face(face_roi, name)
    return False

def show_face_management():
    st.sidebar.markdown("---")
    st.sidebar.subheader("Known Faces")
    
    faces = st.session_state.face_recognizer.get_known_faces()
    if not faces:
        st.sidebar.info("No faces trained yet")
        return
    
    st.sidebar.markdown(f"**Total faces:** {len(faces)}")
    st.sidebar.markdown("---")
    
    for label, name in faces.items():
        col1, col2 = st.sidebar.columns([3, 1])
        col1.text(f"{name} (ID: {label})")
        if col2.button("Remove", key=f"remove_{label}"):
            if st.session_state.face_recognizer.remove_face(label):
                st.sidebar.success(f"Removed {name}")
                st.experimental_rerun()
            else:
                st.sidebar.error("Failed to remove face")

def main():
    try:
        initialize()
        
        st.title("WhoAmI - Facial Recognition")
        st.sidebar.title("Options")
        
        # Add face management section to sidebar
        show_face_management()
        
        mode = st.sidebar.selectbox("Choose Mode", ["Detect", "Train"])
        source = st.sidebar.selectbox("Choose Source", ["Upload", "Camera"])
        
        if mode == "Detect":
            if source == "Upload":
                uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
                if uploaded_file:
                    faces_count = process_upload(uploaded_file)
                    st.success(f"Found {faces_count} faces in the image!")
            
            elif source == "Camera":
                if st.button("Start Camera"):
                    process_camera()
                if st.button("Stop Camera"):
                    st.session_state.camera.stop()
                
        elif mode == "Train":
            name = st.text_input("Enter name for the face:")
            if source == "Upload":
                uploaded_file = st.file_uploader("Choose an image...", 
                                               type=["jpg", "jpeg", "png"])
                if uploaded_file and name:
                    image = load_image(uploaded_file)
                    cv_image = convert_to_cv2(image)
                    if train_face(name, cv_image):
                        st.success(f"Successfully trained face for {name}!")
                    else:
                        st.error("Please provide an image with exactly one face.")
            
            elif source == "Camera":
                if st.button("Capture Face"):
                    st.session_state.camera.start()
                    frame = st.session_state.camera.get_frame()
                    if frame is not None and name:
                        if train_face(name, frame):
                            st.success(f"Successfully trained face for {name}!")
                        else:
                            st.error("Please ensure exactly one face is visible.")
                    st.session_state.camera.stop()
                    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        if 'camera' in st.session_state:
            st.session_state.camera.stop()

if __name__ == "__main__":
    main()
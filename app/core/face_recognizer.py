import cv2
import numpy as np
import pickle
from pathlib import Path

class FaceRecognizer:
    def __init__(self, config):
        self.config = config
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        except AttributeError:
            # Fallback for older OpenCV versions
            self.recognizer = cv2.face_LBPHFaceRecognizer.create()
        except Exception as e:
            raise Exception(f"Failed to create face recognizer. Please ensure opencv-contrib-python is installed. Error: {str(e)}")
        self.known_faces = []
        self.labels = []
        self.label_names = {}
        self.current_label = 0
        self.model_path = Path(config['paths']['models'])
        self.model_file = self.model_path / 'model.yml'
        self.labels_file = self.model_path / 'labels.pkl'
        self.load_models()

    def load_models(self):
        if self.model_file.exists() and self.labels_file.exists():
            self.recognizer.read(str(self.model_file))
            with open(self.labels_file, 'rb') as f:
                data = pickle.load(f)
                self.label_names = data['names']
                self.current_label = data['next_label']

    def save_models(self):
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.recognizer.write(str(self.model_file))
        with open(self.labels_file, 'wb') as f:
            pickle.dump({
                'names': self.label_names,
                'next_label': self.current_label
            }, f)

    def add_face(self, face_image, name):
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        self.known_faces.append(gray)
        self.label_names[self.current_label] = name
        self.labels.append(self.current_label)
        self.current_label += 1
        
        if len(self.known_faces) > 0:
            self.recognizer.train(self.known_faces, np.array(self.labels))
            self.save_models()
            return True
        return False

    def recognize_faces(self, image, faces):
        predictions = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            try:
                label, confidence = self.recognizer.predict(roi_gray)
                name = self.label_names.get(label, "Unknown")
                if confidence > 100:  # Adjust threshold as needed
                    name = "Unknown"
                predictions.append((name, (x, y, w, h)))
            except:
                predictions.append(("Unknown", (x, y, w, h)))
                
        return predictions

    def get_known_faces(self):
        """Return a dictionary of label: name pairs"""
        return {label: name for label, name in self.label_names.items()}

    def remove_face(self, label):
        """Remove a face from the recognizer"""
        if label in self.label_names:
            # Remove from label names
            name = self.label_names.pop(label)
            
            # Remove from training data
            indices = [i for i, l in enumerate(self.labels) if l == label]
            for index in sorted(indices, reverse=True):
                self.known_faces.pop(index)
                self.labels.pop(index)
            
            # Retrain the model if we have remaining faces
            if self.known_faces:
                self.recognizer.train(self.known_faces, np.array(self.labels))
            self.save_models()
            return True
        return False

    def get_face_count(self):
        """Return the number of known faces"""
        return len(self.label_names)
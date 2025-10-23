#!/usr/bin/env python3
"""
Lightweight Gaze Tracker for Raspberry Pi
Detects eye gaze direction and prints to terminal
"""

import cv2
import dlib
import numpy as np
import math
import time

class GazeTracker:
    def __init__(self):
        # Initialize dlib's face detector and landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        
        # Download the shape predictor model (68 face landmarks)
        # This is a lightweight model suitable for Raspberry Pi
        try:
            self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        except:
            print("Error: shape_predictor_68_face_landmarks.dat not found!")
            print("Download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            exit(1)
        
        # Eye landmark indices (dlib 68-point model)
        self.LEFT_EYE_POINTS = list(range(36, 42))
        self.RIGHT_EYE_POINTS = list(range(42, 48))
        
        # Gaze direction thresholds
        self.HORIZONTAL_THRESHOLD = 0.1
        self.VERTICAL_THRESHOLD = 0.1
        
    def get_eye_center(self, landmarks, eye_points):
        """Calculate the center point of an eye"""
        eye_region = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in eye_points])
        center = np.mean(eye_region, axis=0)
        return center.astype(int)
    
    def get_eye_aspect_ratio(self, landmarks, eye_points):
        """Calculate Eye Aspect Ratio (EAR) for blink detection"""
        eye_region = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in eye_points])
        
        # Calculate distances
        A = np.linalg.norm(eye_region[1] - eye_region[5])
        B = np.linalg.norm(eye_region[2] - eye_region[4])
        C = np.linalg.norm(eye_region[0] - eye_region[3])
        
        # Calculate EAR
        ear = (A + B) / (2.0 * C)
        return ear
    
    def calculate_gaze_ratio(self, landmarks, eye_points):
        """Calculate gaze direction ratio for an eye"""
        eye_region = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in eye_points])
        
        # Get eye corners
        left_corner = eye_region[0]
        right_corner = eye_region[3]
        
        # Calculate horizontal gaze ratio
        eye_width = right_corner[0] - left_corner[0]
        pupil_x = (eye_region[1][0] + eye_region[2][0]) / 2  # Average of top points
        
        if eye_width > 0:
            gaze_ratio = (pupil_x - left_corner[0]) / eye_width
        else:
            gaze_ratio = 0.5
            
        return gaze_ratio
    
    def determine_gaze_direction(self, left_ratio, right_ratio, left_ear, right_ear):
        """Determine gaze direction based on eye ratios and blink detection"""
        # Check if eyes are closed (blinking)
        if left_ear < 0.2 or right_ear < 0.2:
            return "BLINKING"
        
        # Average the gaze ratios from both eyes
        avg_ratio = (left_ratio + right_ratio) / 2
        
        # Determine horizontal direction
        if avg_ratio < 0.4:
            horizontal = "LEFT"
        elif avg_ratio > 0.6:
            horizontal = "RIGHT"
        else:
            horizontal = "CENTER"
        
        return horizontal
    
    def process_frame(self, frame):
        """Process a single frame and return gaze direction"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        if len(faces) == 0:
            return "NO_FACE"
        
        # Use the first detected face
        face = faces[0]
        landmarks = self.predictor(gray, face)
        
        # Calculate eye centers
        left_eye_center = self.get_eye_center(landmarks, self.LEFT_EYE_POINTS)
        right_eye_center = self.get_eye_center(landmarks, self.RIGHT_EYE_POINTS)
        
        # Calculate eye aspect ratios for blink detection
        left_ear = self.get_eye_aspect_ratio(landmarks, self.LEFT_EYE_POINTS)
        right_ear = self.get_eye_aspect_ratio(landmarks, self.RIGHT_EYE_POINTS)
        
        # Calculate gaze ratios
        left_gaze_ratio = self.calculate_gaze_ratio(landmarks, self.LEFT_EYE_POINTS)
        right_gaze_ratio = self.calculate_gaze_ratio(landmarks, self.RIGHT_EYE_POINTS)
        
        # Determine gaze direction
        gaze_direction = self.determine_gaze_direction(
            left_gaze_ratio, right_gaze_ratio, left_ear, right_ear
        )
        
        return gaze_direction
    
    def run(self, camera_index=0):
        """Main loop to capture video and track gaze"""
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return
        
        print("Gaze Tracker Started!")
        print("Press 'q' to quit")
        print("Gaze directions: LEFT, CENTER, RIGHT, BLINKING, NO_FACE")
        print("-" * 50)
        
        last_direction = None
        direction_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Process frame
                gaze_direction = self.process_frame(frame)
                
                # Only print when direction changes to reduce spam
                if gaze_direction != last_direction:
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"[{timestamp}] Gaze: {gaze_direction}")
                    last_direction = gaze_direction
                
                # Optional: Display frame (comment out for headless operation)
                # cv2.imshow('Gaze Tracker', frame)
                
                # Break on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nStopping gaze tracker...")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    """Main function"""
    print("Initializing Lightweight Gaze Tracker...")
    
    # Create gaze tracker instance
    tracker = GazeTracker()
    
    # Start tracking (use camera index 0 for default camera)
    tracker.run(camera_index=0)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Ultra-Lightweight Gaze Tracker for Raspberry Pi
Uses only OpenCV for face and eye detection - no dlib required
"""

import cv2
import numpy as np
import time

class SimpleGazeTracker:
    def __init__(self):
        # Load OpenCV's Haar Cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Gaze direction thresholds
        self.HORIZONTAL_THRESHOLD = 0.3
        
    def detect_eyes_in_face(self, face_roi, gray_face):
        """Detect eyes within a face region"""
        eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 4)
        
        if len(eyes) >= 2:
            # Sort eyes by x-coordinate (left to right)
            eyes = sorted(eyes, key=lambda x: x[0])
            return eyes[:2]  # Return only the first two eyes
        return []
    
    def calculate_gaze_direction(self, left_eye, right_eye, face_roi):
        """Calculate gaze direction based on eye positions and face region"""
        # Get eye centers
        left_center_x = left_eye[0] + left_eye[2] // 2
        left_center_y = left_eye[1] + left_eye[3] // 2
        right_center_x = right_eye[0] + right_eye[2] // 2
        right_center_y = right_eye[1] + right_eye[3] // 2
        
        # Calculate the center point between eyes
        eye_center_x = (left_center_x + right_center_x) // 2
        eye_center_y = (left_center_y + right_center_y) // 2
        
        # Get face dimensions
        face_height, face_width = face_roi.shape[:2]
        
        # Calculate eye distance for horizontal normalization
        eye_distance = right_center_x - left_center_x
        
        # Horizontal gaze detection (fixed logic)
        horizontal_direction = "CENTER"
        if eye_distance > 0:
            # Calculate relative position of eye center within face
            # Face center is at face_width/2, so we compare eye_center_x to this
            face_center_x = face_width // 2
            relative_x = (eye_center_x - face_center_x) / (face_width // 2)
            
            if relative_x < -0.25:  # Eyes are significantly to the left of face center
                horizontal_direction = "LEFT"
            elif relative_x > 0.25:  # Eyes are significantly to the right of face center
                horizontal_direction = "RIGHT"
            else:
                horizontal_direction = "CENTER"
        
        # Vertical gaze detection
        vertical_direction = "CENTER"
        face_center_y = face_height // 2
        relative_y = (eye_center_y - face_center_y) / (face_height // 2)
        
        if relative_y < -0.2:  # Eyes are significantly above face center
            vertical_direction = "UP"
        elif relative_y > 0.2:  # Eyes are significantly below face center
            vertical_direction = "DOWN"
        else:
            vertical_direction = "CENTER"
        
        # Combine horizontal and vertical directions
        if horizontal_direction == "CENTER" and vertical_direction == "CENTER":
            return "CENTER"
        elif horizontal_direction == "CENTER":
            return vertical_direction
        elif vertical_direction == "CENTER":
            return horizontal_direction
        else:
            return f"{vertical_direction}_{horizontal_direction}"
    
    def detect_blinking(self, eyes):
        """Simple blink detection based on eye detection"""
        if len(eyes) < 2:
            return True  # Consider it blinking if we can't detect both eyes
        return False
    
    def process_frame(self, frame):
        """Process a single frame and return gaze direction"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return "NO_FACE"
        
        # Use the first detected face
        face = faces[0]
        x, y, w, h = face
        
        # Extract face region
        face_roi = frame[y:y+h, x:x+w]
        gray_face = gray[y:y+h, x:x+w]
        
        # Detect eyes within the face
        eyes = self.detect_eyes_in_face(face_roi, gray_face)
        
        # Check for blinking
        if self.detect_blinking(eyes):
            return "BLINKING"
        
        if len(eyes) >= 2:
            # Calculate gaze direction
            left_eye = eyes[0]
            right_eye = eyes[1]
            gaze_direction = self.calculate_gaze_direction(left_eye, right_eye, face_roi)
            return gaze_direction
        
        return "NO_EYES"
    
    def run(self, camera_index=0):
        """Main loop to capture video and track gaze"""
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return
        
        print("Simple Gaze Tracker Started!")
        print("Press 'q' to quit")
        print("Gaze directions: LEFT, CENTER, RIGHT, UP, DOWN, UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT, BLINKING, NO_FACE, NO_EYES")
        print("-" * 80)
        
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
                
                # Display frame with annotations for testing
                display_frame = frame.copy()
                
                # Draw face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    face = faces[0]
                    x, y, w, h = face
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(display_frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Draw face center
                    face_center_x = x + w // 2
                    face_center_y = y + h // 2
                    cv2.circle(display_frame, (face_center_x, face_center_y), 5, (0, 255, 0), -1)
                    
                    # Detect and draw eyes
                    face_roi = frame[y:y+h, x:x+w]
                    gray_face = gray[y:y+h, x:x+w]
                    eyes = self.detect_eyes_in_face(face_roi, gray_face)
                    
                    if len(eyes) >= 2:
                        left_eye = eyes[0]
                        right_eye = eyes[1]
                        
                        # Draw left eye
                        left_eye_x = x + left_eye[0]
                        left_eye_y = y + left_eye[1]
                        left_eye_w = left_eye[2]
                        left_eye_h = left_eye[3]
                        cv2.rectangle(display_frame, (left_eye_x, left_eye_y), 
                                    (left_eye_x + left_eye_w, left_eye_y + left_eye_h), (255, 0, 0), 2)
                        cv2.putText(display_frame, "L", (left_eye_x, left_eye_y-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
                        # Draw right eye
                        right_eye_x = x + right_eye[0]
                        right_eye_y = y + right_eye[1]
                        right_eye_w = right_eye[2]
                        right_eye_h = right_eye[3]
                        cv2.rectangle(display_frame, (right_eye_x, right_eye_y), 
                                    (right_eye_x + right_eye_w, right_eye_y + right_eye_h), (255, 0, 0), 2)
                        cv2.putText(display_frame, "R", (right_eye_x, right_eye_y-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
                        # Draw eye centers
                        left_center_x = left_eye_x + left_eye_w // 2
                        left_center_y = left_eye_y + left_eye_h // 2
                        right_center_x = right_eye_x + right_eye_w // 2
                        right_center_y = right_eye_y + right_eye_h // 2
                        
                        cv2.circle(display_frame, (left_center_x, left_center_y), 3, (255, 0, 0), -1)
                        cv2.circle(display_frame, (right_center_x, right_center_y), 3, (255, 0, 0), -1)
                        
                        # Draw line between eye centers
                        cv2.line(display_frame, (left_center_x, left_center_y), 
                               (right_center_x, right_center_y), (255, 0, 0), 2)
                        
                        # Draw overall eye center
                        eye_center_x = (left_center_x + right_center_x) // 2
                        eye_center_y = (left_center_y + right_center_y) // 2
                        cv2.circle(display_frame, (eye_center_x, eye_center_y), 5, (0, 0, 255), -1)
                        
                        # Draw line from face center to eye center
                        cv2.line(display_frame, (face_center_x, face_center_y), 
                               (eye_center_x, eye_center_y), (0, 0, 255), 2)
                
                # Display gaze direction on frame
                cv2.putText(display_frame, f"Gaze: {gaze_direction}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                cv2.imshow('Gaze Tracker - Testing Mode', display_frame)
                
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
    print("Initializing Simple Gaze Tracker...")
    print("Note: This is a simplified version using only OpenCV")
    print("For more accurate gaze tracking, use the dlib version on Linux/Raspberry Pi")
    
    # Create gaze tracker instance
    tracker = SimpleGazeTracker()
    
    # Start tracking (use camera index 0 for default camera)
    tracker.run(camera_index=0)

if __name__ == "__main__":
    main()

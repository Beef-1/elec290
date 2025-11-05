#!/usr/bin/env python3
"""
Ultra-Lightweight Gaze Tracker for Raspberry Pi
Uses only OpenCV for face and eye detection - no dlib required
"""

import cv2
import numpy as np
import time
import os
import warnings

# Suppress OpenCV warnings
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
warnings.filterwarnings('ignore', category=UserWarning)

class SimpleGazeTracker:
    def __init__(self):
        # Load OpenCV's Haar Cascade classifiers
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        # Check if cascade loaded successfully
        if self.face_cascade.empty():
            print(f"ERROR: Could not load face cascade from {face_cascade_path}")
            raise RuntimeError("Face cascade classifier not loaded")
        
        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        
        if self.eye_cascade.empty():
            print(f"ERROR: Could not load eye cascade from {eye_cascade_path}")
            raise RuntimeError("Eye cascade classifier not loaded")
        
        print("Face and eye cascade classifiers loaded successfully.")
        
        # Gaze direction thresholds
        self.HORIZONTAL_THRESHOLD = 0.3
        # Calibration ranges for normalized gaze mapping (relative coords)
        # These represent the min/max of relative_x and relative_y observed during calibration
        self.calib_min = np.array([-0.5, -0.4], dtype=np.float32)  # (x_min, y_min)
        self.calib_max = np.array([0.5, 0.4], dtype=np.float32)   # (x_max, y_max)
        
        # Temporal smoothing - keep history of gaze positions
        self.gaze_history = []
        self.max_history = 5  # Number of frames to average
        
        # Frame skipping for lower framerate
        self.frame_skip = 2  # Process every 2nd frame (30fps -> 15fps)
        self.frame_counter = 0
        
        # Debug mode - show camera feed
        self.debug_mode = False
        
    def detect_eyes_in_face(self, face_roi, gray_face):
        """Detect eyes within a face region - more lenient for looking down"""
        # Use more lenient parameters to detect eyes when looking down
        eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 3)  # Reduced minNeighbors from 4 to 3
        
        if len(eyes) >= 2:
            # Sort eyes by x-coordinate (left to right)
            eyes = sorted(eyes, key=lambda x: x[0])
            return eyes[:2]  # Return only the first two eyes
        
        # If not enough eyes found, try with even more lenient settings
        if len(eyes) < 2:
            eyes = self.eye_cascade.detectMultiScale(gray_face, 1.05, 2)
            if len(eyes) >= 2:
                eyes = sorted(eyes, key=lambda x: x[0])
                return eyes[:2]
        
        return []
    
    def calculate_eye_aspect_ratio(self, eye_bbox):
        """Calculate Eye Aspect Ratio (EAR) for blink detection - works even when looking down"""
        x, y, w, h = eye_bbox
        # EAR is height/width ratio - when closed, height is small relative to width
        if w > 0:
            ear = float(h) / float(w)
        else:
            ear = 0.0
        return ear
    
    def detect_pupil(self, eye_roi_gray):
        """Detect pupil center within an eye region using image processing - improved for edge detection"""
        # Enhance contrast - increased clip limit for better pupil detection at edges
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(eye_roi_gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # Use adaptive thresholding to find dark regions (pupil)
        # Smaller block size for better detection of pupils near edges
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 9, 3)
        
        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Also try Otsu thresholding as fallback for better edge cases
        _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours in both thresholded images
        contours1, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(thresh_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours = contours1 + contours2
        
        if len(all_contours) == 0:
            return None
        
        # Find the largest contour (likely the pupil)
        largest_contour = max(all_contours, key=cv2.contourArea)
        
        # Check if contour is reasonable size (between 3% and 50% of eye area - more lenient)
        eye_area = eye_roi_gray.shape[0] * eye_roi_gray.shape[1]
        contour_area = cv2.contourArea(largest_contour)
        
        if contour_area < 0.03 * eye_area or contour_area > 0.5 * eye_area:
            return None
        
        # Calculate centroid
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Ensure pupil position is within eye bounds (safety check)
        h, w = eye_roi_gray.shape
        cx = max(0, min(w-1, cx))
        cy = max(0, min(h-1, cy))
        
        return (cx, cy)
    
    def get_eye_region_from_bbox(self, eye_bbox, face_roi):
        """Extract eye region from bounding box"""
        x, y, w, h = eye_bbox
        # Add some padding
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(face_roi.shape[1] - x, w + 2*padding)
        h = min(face_roi.shape[0] - y, h + 2*padding)
        
        eye_roi = face_roi[y:y+h, x:x+w]
        return eye_roi, (x, y)
    
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

    def calculate_gaze_ratio_in_eye(self, pupil_pos, eye_roi_shape):
        """Calculate horizontal gaze ratio within an eye (0.0 = left edge, 0.5 = center, 1.0 = right edge)
        
        Args:
            pupil_pos: (x, y) tuple relative to eye ROI, or None
            eye_roi_shape: (height, width) of the eye ROI
        """
        if pupil_pos is None:
            return 0.5  # Default to center if no pupil detected
        
        pupil_x, pupil_y = pupil_pos
        eye_h, eye_w = eye_roi_shape
        
        # Calculate relative position within eye (0.0 to 1.0)
        if eye_w > 0:
            ratio = float(pupil_x) / float(eye_w)
        else:
            ratio = 0.5
        
        return np.clip(ratio, 0.0, 1.0)
    
    def compute_relative_eye_offsets(self, left_eye, right_eye, face_roi, gray_face):
        """Return continuous relative (x, y) offsets using pupil detection for accuracy."""
        face_height, face_width = face_roi.shape[:2]
        face_center_x = face_width // 2
        face_center_y = face_height // 2
        
        # Extract eye regions
        left_eye_roi, left_offset = self.get_eye_region_from_bbox(left_eye, face_roi)
        right_eye_roi, right_offset = self.get_eye_region_from_bbox(right_eye, face_roi)
        
        # Convert to grayscale if needed
        if len(left_eye_roi.shape) == 3:
            left_gray = cv2.cvtColor(left_eye_roi, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_eye_roi
            
        if len(right_eye_roi.shape) == 3:
            right_gray = cv2.cvtColor(right_eye_roi, cv2.COLOR_BGR2GRAY)
        else:
            right_gray = right_eye_roi
        
        # Detect pupils
        left_pupil = self.detect_pupil(left_gray)
        right_pupil = self.detect_pupil(right_gray)
        
        # Calculate gaze ratios within each eye (0.0 = left edge, 1.0 = right edge)
        # This is more accurate for left/right detection than just eye position
        left_gaze_ratio = self.calculate_gaze_ratio_in_eye(left_pupil, left_gray.shape)
        right_gaze_ratio = self.calculate_gaze_ratio_in_eye(right_pupil, right_gray.shape)
        avg_gaze_ratio = (left_gaze_ratio + right_gaze_ratio) / 2.0
        
        # Use non-linear mapping for better edge detection
        # Apply a power curve to exaggerate edge positions
        # Center (0.5) stays 0, edges (0.0 and 1.0) map to -1.0 and 1.0 with better sensitivity
        if avg_gaze_ratio < 0.5:
            # Left side: map [0.0, 0.5] to [-1.0, 0.0] with power curve
            normalized = (0.5 - avg_gaze_ratio) / 0.5  # [0.0, 1.0] for left
            horizontal_offset = -np.power(normalized, 0.7)  # Power curve for better edge sensitivity
        else:
            # Right side: map [0.5, 1.0] to [0.0, 1.0] with power curve
            normalized = (avg_gaze_ratio - 0.5) / 0.5  # [0.0, 1.0] for right
            horizontal_offset = np.power(normalized, 0.7)  # Power curve for better edge sensitivity
        
        # Additional scaling for maximum sensitivity (allows values beyond 1.0 for calibration)
        horizontal_offset *= 2.5  # Scale up significantly - this will be clipped but allows better edge detection
        
        # For vertical: use pupil vertical ratios within each eye for sensitivity
        # 0.0 = top of eye (UP), 1.0 = bottom of eye (DOWN)
        if left_pupil is not None:
            left_vert_ratio = float(left_pupil[1]) / max(1.0, float(left_gray.shape[0]))
        else:
            left_vert_ratio = 0.5
        if right_pupil is not None:
            right_vert_ratio = float(right_pupil[1]) / max(1.0, float(right_gray.shape[0]))
        else:
            right_vert_ratio = 0.5
        avg_vert_ratio = (left_vert_ratio + right_vert_ratio) / 2.0

        # Non-linear mapping for vertical like horizontal (more edge sensitivity)
        if avg_vert_ratio < 0.5:
            # Upper half -> negative offset (UP)
            vnorm = (0.5 - avg_vert_ratio) / 0.5
            vertical_offset_from_eye = -np.power(vnorm, 0.7)
        else:
            # Lower half -> positive offset (DOWN)
            vnorm = (avg_vert_ratio - 0.5) / 0.5
            vertical_offset_from_eye = np.power(vnorm, 0.7)
        vertical_offset_from_eye *= 2.0  # scale vertical sensitivity

        # Also compute face-relative vertical for stability and blend
        if left_pupil and right_pupil:
            left_pupil_abs_y = left_offset[1] + left_pupil[1]
            right_pupil_abs_y = right_offset[1] + right_pupil[1]
            gaze_center_y = (left_pupil_abs_y + right_pupil_abs_y) // 2
        else:
            left_center_y = left_eye[1] + left_eye[3] // 2
            right_center_y = right_eye[1] + right_eye[3] // 2
            gaze_center_y = (left_center_y + right_center_y) // 2
        vertical_offset_from_face = (gaze_center_y - face_center_y) / max(1, (face_height // 2))

        # Blend: favor eye-based ratio but keep some face-based stability
        blended_vertical = 0.7 * vertical_offset_from_eye + 0.3 * vertical_offset_from_face

        # Combine horizontal (from gaze ratio) and vertical (blended)
        # Allow horizontal to go beyond ±1.0 for better edge calibration (will be clipped later)
        relative_x = float(horizontal_offset)
        relative_y = float(np.clip(blended_vertical, -1.5, 1.5))

        return relative_x, relative_y
    
    def detect_blinking(self, left_eye, right_eye):
        """Blink detection using Eye Aspect Ratio (EAR) - works when looking down"""
        if left_eye is None or right_eye is None:
            return True
        
        left_ear = self.calculate_eye_aspect_ratio(left_eye)
        right_ear = self.calculate_eye_aspect_ratio(right_eye)
        
        # Average EAR
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Eyes are closed if EAR is below threshold (eyes become narrower when closed)
        # Typical open eye: EAR ~ 0.3-0.4, closed: EAR < 0.2
        EAR_THRESHOLD = 0.22
        
        return avg_ear < EAR_THRESHOLD
    
    def process_frame(self, frame):
        """Process a single frame and return gaze direction"""
        # Resize frame if too large for better detection performance
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640.0 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame_resized = cv2.resize(frame, (new_width, new_height))
        else:
            frame_resized = frame
            scale = 1.0
        
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        
        # Equalize histogram for better contrast (helps with face detection)
        gray = cv2.equalizeHist(gray)
        
        # Detect faces with very lenient parameters
        # Try multiple scale factors and minNeighbors for better detection
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,  # Smaller scale factor = more sensitive
            minNeighbors=2,    # Very low = more sensitive (was 3)
            minSize=(30, 30),  # Minimum face size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # If no faces found, try even more lenient settings
        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.02,  # Even smaller
                minNeighbors=1,    # Minimum required
                minSize=(20, 20),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
        
        # If still no faces, try without minSize constraint
        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.01,
                minNeighbors=1,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
        
        if len(faces) == 0:
            debug_frame = None
            if self.debug_mode:
                debug_frame = frame_resized.copy()
                cv2.putText(debug_frame, "NO FACE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(debug_frame, "Check lighting and positioning", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return "NO_FACE", (0.0, 0.0), debug_frame
        
        # Scale face coordinates back if frame was resized
        if scale != 1.0:
            faces = [(int(x/scale), int(y/scale), int(w/scale), int(h/scale)) for (x, y, w, h) in faces]
            # Extract face from original frame
            face = faces[0]
            x, y, w, h = face
            if x + w > width or y + h > height:
                # Adjust if out of bounds
                x = max(0, min(x, width - w))
                y = max(0, min(y, height - h))
                w = min(w, width - x)
                h = min(h, height - y)
            face_roi = frame[y:y+h, x:x+w]
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            # Use the first detected face from resized frame
            face = faces[0]
            x, y, w, h = face
            face_roi = frame_resized[y:y+h, x:x+w]
            gray_face = gray[y:y+h, x:x+w]
        
        # Detect eyes within the face
        eyes = self.detect_eyes_in_face(face_roi, gray_face)
        
        if len(eyes) >= 2:
            left_eye = eyes[0]
            right_eye = eyes[1]
            
            # Check for blinking using EAR (works even when looking down)
            if self.detect_blinking(left_eye, right_eye):
                debug_frame = None
                if self.debug_mode:
                    debug_frame = frame_resized.copy() if scale != 1.0 else frame.copy()
                    face = faces[0]
                    if scale != 1.0:
                        x, y, w, h = (int(f/scale) for f in face)
                    else:
                        x, y, w, h = face
                    cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(debug_frame, "BLINKING", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                return "BLINKING", (0.0, 0.0), debug_frame
            
            gaze_direction = self.calculate_gaze_direction(left_eye, right_eye, face_roi)
            rel_x, rel_y = self.compute_relative_eye_offsets(left_eye, right_eye, face_roi, gray_face)
            
            # Apply temporal smoothing
            self.gaze_history.append((rel_x, rel_y))
            if len(self.gaze_history) > self.max_history:
                self.gaze_history.pop(0)
            
            # Average the history
            smoothed_x = np.mean([g[0] for g in self.gaze_history])
            smoothed_y = np.mean([g[1] for g in self.gaze_history])
            
            return gaze_direction, (smoothed_x, smoothed_y)
        
        # If no eyes, still add to history but with zero (maintains smoothing)
        self.gaze_history.append((0.0, 0.0))
        if len(self.gaze_history) > self.max_history:
            self.gaze_history.pop(0)
        
        return "NO_EYES", (0.0, 0.0)
    
    def list_cameras(self):
        """List available cameras with better error handling"""
        import sys
        available = []
        print("Scanning for available cameras...")
        
        # Suppress OpenCV errors during scanning
        import os
        # Save current stderr
        old_stderr = sys.stderr
        
        # Try different backends for Windows compatibility
        backends = [
            cv2.CAP_DSHOW,  # DirectShow (Windows)
            cv2.CAP_MSMF,   # Microsoft Media Foundation (Windows)
            cv2.CAP_ANY     # Auto-detect
        ]
        
        for i in range(10):  # Check first 10 indices
            for backend in backends:
                try:
                    # Temporarily suppress stderr
                    sys.stderr = open(os.devnull, 'w')
                    cap = cv2.VideoCapture(i, backend)
                    sys.stderr = old_stderr
                    
                    if cap.isOpened():
                        # Try to read a frame to verify it works
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            if i not in available:  # Avoid duplicates
                                available.append(i)
                                print(f"  Camera {i}: Available")
                            cap.release()
                            break  # Found working camera, try next index
                        cap.release()
                except:
                    sys.stderr = old_stderr
                    pass
        
        sys.stderr = old_stderr
        return available
    
    def run(self, camera_index=None):
        """Main loop to capture video and track gaze and display gaze point UI"""
        # If no camera specified, try to find one
        if camera_index is None:
            available = self.list_cameras()
            if len(available) == 0:
                print("Error: No cameras found!")
                print("\nTroubleshooting:")
                print("  1. Make sure your webcam is connected")
                print("  2. Check if another application is using the camera")
                print("  3. Try specifying a camera manually: --camera 0")
                return
            camera_index = available[0]
            if len(available) > 1:
                print(f"Using camera {camera_index} (first available). Use --camera <index> to specify another.")
        
        # Try different backends for Windows compatibility
        # On Windows, prefer DirectShow over Media Foundation to avoid index issues
        import platform, os
        if platform.system() == 'Windows':
            os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
        cap = None
        backends = [
            (cv2.CAP_ANY, "Auto-detect"),
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
        ]
        
        for backend_id, backend_name in backends:
            try:
                cap = cv2.VideoCapture(camera_index, backend_id)
                if cap.isOpened():
                    # Test if we can actually read a frame
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        print(f"Camera opened successfully using {backend_name}")
                        break
                    else:
                        cap.release()
                        cap = None
            except Exception as e:
                if cap:
                    cap.release()
                cap = None
                continue
        
        # Final fallback: try without specifying backend
        if (cap is None or not cap.isOpened()):
            tmp = cv2.VideoCapture(camera_index)
            if tmp.isOpened():
                cap = tmp
                print("Camera opened successfully using default backend")

        if cap is None or not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            print("\nTroubleshooting:")
            print("  1. Make sure your webcam is connected and not in use by another app")
            print("  2. Try running as administrator")
            print("  3. Check camera permissions in Windows settings")
            print("  4. Try specifying a different camera: --list to see available cameras")
            return
        
        # Set camera resolution for better performance
        # Suppress warnings during property setting
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("Simple Gaze Tracker Started!")
        print("Press 'q' to quit | 'h' for help | 'w/a/s/r' to calibrate edges (up/left/down/right) | 'c' center | 'd' toggle debug (camera feed)")
        print("Gaze directions: LEFT, CENTER, RIGHT, UP, DOWN, UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT, BLINKING, NO_FACE, NO_EYES")
        print("-" * 80)
        
        last_direction = None
        direction_count = 0

        # Gaze point visualizer: a simple 800x600 canvas where we draw a dot
        canvas_w, canvas_h = 800, 600
        bg_color = (20, 20, 20)
        dot_color = (0, 255, 255)
        help_color = (200, 200, 200)

        def map_relative_to_canvas(rel_xy):
            # Clamp using calibration ranges, then scale to canvas
            rel = np.array(rel_xy, dtype=np.float32)
            # Avoid zero range by ensuring max>min
            rng = np.maximum(self.calib_max - self.calib_min, 1e-3)
            norm = (rel - self.calib_min) / rng  # [0,1]
            norm = np.clip(norm, 0.0, 1.0)
            x_px = int(norm[0] * (canvas_w - 1))
            y_px = int(norm[1] * (canvas_h - 1))
            return x_px, y_px
        
        consecutive_failures = 0
        max_failures = 10
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        print(f"\nError: Could not read frame after {max_failures} attempts")
                        print("Camera may have been disconnected or is in use by another application.")
                        break
                    # Small delay before retry
                    time.sleep(0.1)
                    continue
                
                consecutive_failures = 0  # Reset on successful read
                
                # Frame skipping - only process every Nth frame
                self.frame_counter += 1
                if self.frame_counter % self.frame_skip != 0:
                    # Still update display but use last processed gaze values
                    if not hasattr(self, 'last_gaze_result'):
                        continue  # Skip display if we haven't processed anything yet
                    result = self.last_gaze_result
                else:
                    # Process frame
                    result = self.process_frame(frame)
                    
                    # Store for frame skipping
                    self.last_gaze_result = result
                
                # Unpack result (may include debug frame)
                if isinstance(result, tuple) and len(result) == 3:
                    gaze_direction, rel, debug_frame = result
                elif isinstance(result, tuple) and len(result) == 2:
                    gaze_direction, rel = result
                    debug_frame = None
                else:
                    gaze_direction, rel = result, (0.0, 0.0)
                    debug_frame = None
                
                # Only print when direction changes to reduce spam
                if gaze_direction != last_direction:
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"[{timestamp}] Gaze: {gaze_direction}")
                    last_direction = gaze_direction
                # Build gaze point canvas
                canvas = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)
                
                # Show face detection status
                status_color = (0, 255, 0) if gaze_direction != "NO_FACE" else (0, 0, 255)
                status_text = f"Status: {gaze_direction}"
                if gaze_direction == "NO_FACE":
                    status_text += " - Make sure face is clearly visible with good lighting"
                
                x_px, y_px = map_relative_to_canvas(rel)
                if gaze_direction != "NO_FACE":
                    cv2.circle(canvas, (x_px, y_px), 10, dot_color, -1)
                
                cv2.putText(canvas, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                cv2.putText(canvas, f"Gaze: {gaze_direction}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, help_color, 2)
                cv2.putText(canvas, "Controls: q quit | h help | w/a/s/r edges | c center | d debug", (10, canvas_h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, help_color, 1)

                cv2.imshow('Gaze Point', canvas)
                
                # Show debug camera feed if enabled
                if self.debug_mode and debug_frame is not None:
                    cv2.imshow('Debug Camera Feed', debug_frame)
                elif self.debug_mode:
                    # Show regular frame if no debug frame
                    cv2.imshow('Debug Camera Feed', frame)
                
                # Break on 'q' key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('h'):
                    print("Help: Use w(up), s(down), a(left), r(right) to set calibration edges to the CURRENT gaze; c sets the center (averages); d toggles debug camera feed.")
                elif key == ord('d'):
                    # Toggle debug mode (press 'd' key)
                    self.debug_mode = not self.debug_mode
                    if self.debug_mode:
                        print("Debug mode ON - Camera feed window will show face detection")
                    else:
                        print("Debug mode OFF")
                        cv2.destroyWindow('Debug Camera Feed')
                elif key == ord('a'):
                    # set left edge (only if not in debug toggle)
                    if not (cv2.waitKey(1) & 0xFF == ord('d')):  # Check if d was just pressed
                        self.calib_min[0] = rel[0]
                        print(f"Calib left set to {self.calib_min[0]:.3f}")
                elif key == ord('r'):  # Changed from 'd' to 'r' for right edge to avoid conflict
                    # set right edge
                    self.calib_max[0] = rel[0]
                    print(f"Calib right set to {self.calib_max[0]:.3f}")
                elif key == ord('w'):
                    # set top edge
                    self.calib_min[1] = rel[1]
                    print(f"Calib top set to {self.calib_min[1]:.3f}")
                elif key == ord('s'):
                    # set bottom edge
                    self.calib_max[1] = rel[1]
                    print(f"Calib bottom set to {self.calib_max[1]:.3f}")
                elif key == ord('c'):
                    # set center as midpoint of current min/max based on current rel
                    # We adjust min/max symmetrically around current rel
                    span_x = max(0.2, (self.calib_max[0] - self.calib_min[0]))
                    span_y = max(0.2, (self.calib_max[1] - self.calib_min[1]))
                    self.calib_min[0] = rel[0] - span_x/2
                    self.calib_max[0] = rel[0] + span_x/2
                    self.calib_min[1] = rel[1] - span_y/2
                    self.calib_max[1] = rel[1] + span_y/2
                    print(f"Center calibration set around ({rel[0]:.3f},{rel[1]:.3f})")
                    
        except KeyboardInterrupt:
            print("\nStopping gaze tracker...")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    """Main function"""
    import sys
    
    print("Initializing Improved Gaze Tracker...")
    print("Features: Pupil detection, temporal smoothing, frame skipping")
    print("-" * 80)
    
    # Create gaze tracker instance
    tracker = SimpleGazeTracker()
    
    # Parse command line arguments for camera selection
    camera_index = None
    if len(sys.argv) > 1:
        if sys.argv[1] == '--camera' and len(sys.argv) > 2:
            try:
                camera_index = int(sys.argv[2])
            except ValueError:
                print(f"Error: Invalid camera index '{sys.argv[2]}'. Must be a number.")
                sys.exit(1)
        elif sys.argv[1] == '--list':
            tracker.list_cameras()
            sys.exit(0)
        elif sys.argv[1] == '--help':
            print("Usage:")
            print("  python gaze_tracker_simple.py              # Auto-detect camera")
            print("  python gaze_tracker_simple.py --camera N    # Use camera index N")
            print("  python gaze_tracker_simple.py --list       # List available cameras")
            sys.exit(0)
    
    # Start tracking
    tracker.run(camera_index=camera_index)

if __name__ == "__main__":
    main()

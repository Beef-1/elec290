#!/usr/bin/env python3
"""
Ultra-Lightweight Gaze Tracker for Raspberry Pi
Uses only OpenCV for face and eye detection - no dlib required
"""

import cv2  # type: ignore
import numpy as np
import time
import os
import warnings
import sys

try:
    import mediapipe as mp  # type: ignore
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None
    _MEDIAPIPE_AVAILABLE = False

try:
    import psutil  # type: ignore
    _PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    _PSUTIL_AVAILABLE = False

try:
    import winsound  # type: ignore
    _WINSOUND_AVAILABLE = True
except ImportError:
    winsound = None
    _WINSOUND_AVAILABLE = False

try:
    import msvcrt  # type: ignore
    _MSVCRT_AVAILABLE = True
except ImportError:
    msvcrt = None
    _MSVCRT_AVAILABLE = False

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
        
        # Load eye cascades (include eyeglasses-friendly model automatically)
        self.eye_cascades = []
        eye_cascade_specs = [
            ('haarcascade_eye.xml', 'standard'),
            ('haarcascade_eye_tree_eyeglasses.xml', 'eyeglasses')
        ]
        for filename, label in eye_cascade_specs:
            cascade_path = cv2.data.haarcascades + filename
            cascade = cv2.CascadeClassifier(cascade_path)
            if cascade.empty():
                print(f"WARNING: Could not load eye cascade {filename} from {cascade_path}")
                continue
            self.eye_cascades.append((label, cascade))
        
        if not self.eye_cascades:
            raise RuntimeError("Eye cascade classifiers not loaded")
        
        # Keep the first cascade for backward compatibility
        self.eye_cascade = self.eye_cascades[0][1]
        print("Face and eye cascade classifiers loaded successfully.")
        
        # MediaPipe setup for robust landmark-based eye detection (handles glasses better)
        self.use_mediapipe = False
        self.mediapipe_face_mesh = None
        if _MEDIAPIPE_AVAILABLE:
            try:
                self.mediapipe_face_mesh = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.use_mediapipe = True
                print("MediaPipe FaceMesh enabled for eye detection (glasses support).")
            except Exception as mp_err:
                print(f"WARNING: Failed to initialize MediaPipe FaceMesh: {mp_err}")
                self.use_mediapipe = False
        else:
            print("MediaPipe not available. Install 'mediapipe' for improved glasses support.")
        
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
        
        # Eye detection state
        self.last_eye_detector = None
        self.last_valid_eyes = None
        self.last_valid_eyes_frame = -100
        self.eye_retention_frames = 5
        self.last_relative_offsets = (0.0, 0.0)
        self.console_line_length = 0
        self.last_head_down = False
        
        # Memory usage limit (3 GB default) to avoid uncontrolled growth
        self.memory_limit_bytes = 3 * 1024 * 1024 * 1024  # 3 GiB
        
        # Downward gaze monitoring
        self.down_threshold_seconds = 1.0
        self.down_beep_cooldown = 0.0  # seconds between beeps (repeat)
        self.down_offset_threshold = 0.15  # relative Y threshold (more sensitive)
        self.down_accumulator = 0.0
        self.last_down_beep_time = None
        self.last_frame_time = time.time()
        self.head_eye_ratio_baseline = 0.42
        self.head_face_ratio_baseline = 1.0
        self.head_baseline_alpha = 0.15

    def _check_memory_usage(self):
        """Return tuple (exceeded, current_bytes)."""
        if not _PSUTIL_AVAILABLE or psutil is None:
            return False, 0
        try:
            process = psutil.Process(os.getpid())
            rss = process.memory_info().rss
            return rss > self.memory_limit_bytes, rss
        except Exception:
            return False, 0

    def _mediapipe_detect_face_and_eyes(self, frame):
        """Use MediaPipe FaceMesh to obtain face and eye bounding boxes."""
        if not self.use_mediapipe or self.mediapipe_face_mesh is None:
            return None
        if frame is None or frame.size == 0:
            return None
        frame_h, frame_w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mediapipe_face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            return None
        face_landmarks = results.multi_face_landmarks[0]
        xs = [lm.x * frame_w for lm in face_landmarks.landmark]
        ys = [lm.y * frame_h for lm in face_landmarks.landmark]
        face_x_min = max(0, int(min(xs)))
        face_y_min = max(0, int(min(ys)))
        face_x_max = min(frame_w - 1, int(max(xs)))
        face_y_max = min(frame_h - 1, int(max(ys)))
        if face_x_max <= face_x_min or face_y_max <= face_y_min:
            return None
        face_w = face_x_max - face_x_min
        face_h = face_y_max - face_y_min
        # Eye landmark indices for MediaPipe FaceMesh
        left_eye_indices = [33, 133, 160, 159, 158, 157, 173, 155, 154, 153, 145, 144]
        right_eye_indices = [362, 263, 387, 386, 385, 384, 398, 382, 381, 380, 374, 373]
        def compute_bbox(indexes):
            x_vals = [face_landmarks.landmark[i].x * frame_w for i in indexes]
            y_vals = [face_landmarks.landmark[i].y * frame_h for i in indexes]
            x_min = max(face_x_min, int(min(x_vals)) - 2)
            y_min = max(face_y_min, int(min(y_vals)) - 2)
            x_max = min(face_x_max, int(max(x_vals)) + 2)
            y_max = min(face_y_max, int(max(y_vals)) + 2)
            if x_max <= x_min or y_max <= y_min:
                return None
            return (x_min - face_x_min,
                    y_min - face_y_min,
                    x_max - x_min,
                    y_max - y_min)
        left_eye_bbox = compute_bbox(left_eye_indices)
        right_eye_bbox = compute_bbox(right_eye_indices)
        if left_eye_bbox is None or right_eye_bbox is None:
            return None
        face_bbox = (face_x_min, face_y_min, face_w, face_h)
        return face_bbox, left_eye_bbox, right_eye_bbox

    def detect_eyes_in_face(self, face_roi, gray_face):
        """Detect eyes within a face region with automatic glasses handling."""
        detections = []
        # Parameter sets tuned for standard and glasses scenarios
        param_sets = [
            (1.10, 4, (28, 20)),
            (1.06, 3, (24, 18)),
            (1.03, 2, (18, 14))
        ]
        for label, cascade in self.eye_cascades:
            for scale, neighbors, min_size in param_sets:
                raw_eyes = cascade.detectMultiScale(
                    gray_face,
                    scaleFactor=scale,
                    minNeighbors=neighbors,
                    minSize=min_size,
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                if len(raw_eyes) >= 2:
                    dedup_eyes = self._deduplicate_rects(raw_eyes)
                    if len(dedup_eyes) < 2:
                        continue
                    dedup_eyes = [eye for eye in dedup_eyes if self._is_valid_eye_box(eye, face_roi.shape)]
                    if len(dedup_eyes) < 2:
                        continue
                    best_score = -np.inf
                    best_pair = None
                    for i in range(len(dedup_eyes)):
                        for j in range(i + 1, len(dedup_eyes)):
                            eye_pair = [tuple(map(int, dedup_eyes[i])), tuple(map(int, dedup_eyes[j]))]
                            eye_pair.sort(key=lambda x: x[0])
                            if not self._is_valid_eye_box(eye_pair[0], face_roi.shape):
                                continue
                            if not self._is_valid_eye_box(eye_pair[1], face_roi.shape):
                                continue
                            if not self._is_valid_eye_pair(eye_pair, face_roi.shape):
                                continue
                            score = self._score_eye_pair(eye_pair)
                            if score > best_score:
                                best_score = score
                                best_pair = eye_pair
                    if best_pair is not None:
                        detections.append((best_score, best_pair, label))
                        break
        
        if detections:
            detections.sort(key=lambda x: x[0], reverse=True)
            best_score, best_eyes, label = detections[0]
            self.last_eye_detector = label
            return best_eyes, label
        
        # Fallback: return the two best eyes ignoring strict validation to avoid total failure
        for label, cascade in self.eye_cascades:
            raw_eyes = cascade.detectMultiScale(
                gray_face,
                scaleFactor=1.05,
                minNeighbors=2,
                minSize=(20, 16),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(raw_eyes) >= 2:
                dedup_eyes = self._deduplicate_rects(raw_eyes)
                dedup_eyes = [eye for eye in dedup_eyes if self._is_valid_eye_box(eye, face_roi.shape, relaxed=True)]
                if len(dedup_eyes) < 2:
                    continue
                pairs = []
                for i in range(len(dedup_eyes)):
                    for j in range(i + 1, len(dedup_eyes)):
                        eye_pair = [tuple(map(int, dedup_eyes[i])), tuple(map(int, dedup_eyes[j]))]
                        eye_pair.sort(key=lambda x: x[0])
                        if not self._is_valid_eye_box(eye_pair[0], face_roi.shape, relaxed=True):
                            continue
                        if not self._is_valid_eye_box(eye_pair[1], face_roi.shape, relaxed=True):
                            continue
                        if not self._is_valid_eye_pair(eye_pair, face_roi.shape, relaxed=True):
                            continue
                        pairs.append(eye_pair)
                if pairs:
                    pairs.sort(key=lambda p: abs((p[1][0] + p[1][2]/2) - (p[0][0] + p[0][2]/2)), reverse=True)
                    eyes = pairs[0]
                    self.last_eye_detector = f"{label}-fallback"
                    return eyes, label
        return [], None
        
        self.last_eye_detector = None
        return [], None

    def _score_eye_pair(self, eyes):
        """Score a pair of eyes to decide which detection is best."""
        if len(eyes) < 2:
            return -np.inf
        left, right = eyes[0], eyes[1]
        # Horizontal separation
        left_center = (left[0] + left[2] / 2.0, left[1] + left[3] / 2.0)
        right_center = (right[0] + right[2] / 2.0, right[1] + right[3] / 2.0)
        separation = abs(right_center[0] - left_center[0])
        # Alignment and size similarity
        size_similarity = 1.0 / (1.0 + abs(left[2] - right[2]) + abs(left[3] - right[3]))
        vertical_alignment = 1.0 / (1.0 + abs(right_center[1] - left_center[1]))
        avg_area = (left[2] * left[3] + right[2] * right[3]) / 2.0
        vertical_offset = abs(right_center[1] - left_center[1])
        vertical_penalty = max(0.0, vertical_offset - 15.0) * 3.0
        return separation + 0.05 * avg_area + 50 * size_similarity + 50 * vertical_alignment - vertical_penalty

    def _is_valid_eye_box(self, eye, face_shape, relaxed=False):
        """Validate individual eye candidate bounding box."""
        if eye is None:
            return False
        face_h, face_w = face_shape[:2]
        x, y, w, h = eye
        if w <= 0 or h <= 0:
            return False
        aspect = float(w) / float(h)
        if aspect < (0.85 if relaxed else 0.95):
            return False
        if aspect > 4.0:
            return False
        area = w * h
        face_area = max(1, face_h * face_w)
        min_area_ratio = 0.005 if relaxed else 0.008
        max_area_ratio = 0.12 if relaxed else 0.08
        ratio = float(area) / float(face_area)
        if ratio < min_area_ratio or ratio > max_area_ratio:
            return False
        center_y = y + h / 2.0
        vertical_ratio = center_y / max(1.0, float(face_h))
        min_vertical = 0.16 if relaxed else 0.20
        max_vertical = 0.65 if relaxed else 0.55
        if vertical_ratio < min_vertical or vertical_ratio > max_vertical:
            return False
        return True

    def _is_valid_eye_pair(self, eyes, face_shape, relaxed=False):
        """Filter out implausible eye pair combinations."""
        if len(eyes) != 2:
            return False
        face_h, face_w = face_shape[:2]
        left, right = eyes[0], eyes[1]
        left_center = (left[0] + left[2] / 2.0, left[1] + left[3] / 2.0)
        right_center = (right[0] + right[2] / 2.0, right[1] + right[3] / 2.0)
        horizontal_sep = abs(right_center[0] - left_center[0])
        vertical_sep = abs(right_center[1] - left_center[1])

        avg_eye_width = (left[2] + right[2]) / 2.0
        avg_eye_height = (left[3] + right[3]) / 2.0

        min_horizontal_ratio = 0.18 if relaxed else 0.2
        min_horizontal = max(12.0, min_horizontal_ratio * face_w, 0.6 * avg_eye_width)
        max_vertical_ratio = 0.45 if relaxed else 0.3
        max_vertical = max(12.0, max_vertical_ratio * face_h, 1.8 * avg_eye_height)
        if horizontal_sep < min_horizontal:
            return False
        if vertical_sep > max_vertical:
            return False
        # Reject overlapping boxes heavily
        overlap_x = max(0, min(left[0] + left[2], right[0] + right[2]) - max(left[0], right[0]))
        overlap_y = max(0, min(left[1] + left[3], right[1] + right[3]) - max(left[1], right[1]))
        overlap_area = overlap_x * overlap_y
        left_area = left[2] * left[3]
        right_area = right[2] * right[3]
        overlap_threshold = 0.6 if relaxed else 0.4
        if overlap_area > overlap_threshold * min(left_area, right_area):
            return False
        return True

    def _intersection_over_union(self, rect_a, rect_b):
        """Compute Intersection over Union of two rectangles."""
        ax, ay, aw, ah = rect_a
        bx, by, bw, bh = rect_b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh

        inter_x1 = max(ax, bx)
        inter_y1 = max(ay, by)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = aw * ah
        area_b = bw * bh
        denom = float(area_a + area_b - inter_area)
        if denom <= 0:
            return 0.0
        return inter_area / denom

    def _deduplicate_rects(self, rects, overlap_threshold=0.6):
        """Remove duplicate detections that heavily overlap."""
        if rects is None or len(rects) == 0:
            return []
        rect_list = [tuple(map(int, r)) for r in rects]
        rect_list.sort(key=lambda r: r[2] * r[3], reverse=True)
        filtered = []
        for rect in rect_list:
            if all(self._intersection_over_union(rect, kept) <= overlap_threshold for kept in filtered):
                filtered.append(rect)
        return filtered
    
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
        
        # Guard against empty ROIs (can happen due to partial detections)
        if (left_eye_roi is None or left_eye_roi.size == 0 or left_eye_roi.shape[0] == 0 or left_eye_roi.shape[1] == 0 or
                right_eye_roi is None or right_eye_roi.size == 0 or right_eye_roi.shape[0] == 0 or right_eye_roi.shape[1] == 0):
            # Fallback to most recent offsets if available, otherwise neutral
            if hasattr(self, 'last_relative_offsets'):
                return self.last_relative_offsets
            return 0.0, 0.0
        
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
        
        # Store for fallback usage
        self.last_relative_offsets = (relative_x, relative_y)
        
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
        height, width = frame.shape[:2]
        eyes = []
        label = None
        face = None
        face_roi = None
        gray_face = None
        used_mediapipe = False

        # First try MediaPipe landmark-based detection for more robust eye tracking (glasses support)
        if self.use_mediapipe:
            mp_result = self._mediapipe_detect_face_and_eyes(frame)
            if mp_result:
                face_bbox, left_eye_bbox, right_eye_bbox = mp_result
                fx, fy, fw, fh = face_bbox
                fx = max(0, min(fx, width - 1))
                fy = max(0, min(fy, height - 1))
                fw = min(fw, width - fx)
                fh = min(fh, height - fy)
                if fw > 10 and fh > 10:
                    candidate_face_roi = frame[fy:fy+fh, fx:fx+fw]
                    if candidate_face_roi.size > 0:
                        face_roi = candidate_face_roi
                        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                        eyes = [left_eye_bbox, right_eye_bbox]
                        label = "mediapipe"
                        face = (fx, fy, fw, fh)
                        used_mediapipe = True
                        self.last_eye_detector = label

        # Fallback to Haar cascades if MediaPipe failed or not available
        if not used_mediapipe:
            scale = 1.0
            frame_resized = frame
            if width > 640:
                scale = 640.0 / width
                frame_resized = cv2.resize(frame, (int(width * scale), int(height * scale)))
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=2,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            if len(faces) == 0:
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.02,
                    minNeighbors=1,
                    minSize=(20, 20),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

            if len(faces) == 0:
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.01,
                    minNeighbors=1,
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

            if len(faces) == 0:
                debug_frame = frame.copy()
                cv2.putText(debug_frame, "NO FACE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(debug_frame, "Check lighting and positioning", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.last_head_down = False
                return "NO_FACE", (0.0, 0.0), debug_frame, False

            if scale != 1.0:
                faces = [(int(x / scale), int(y / scale), int(w / scale), int(h / scale)) for (x, y, w, h) in faces]

            face = faces[0]
            fx, fy, fw, fh = face
            fx = max(0, min(fx, width - 1))
            fy = max(0, min(fy, height - 1))
            fw = min(fw, width - fx)
            fh = min(fh, height - fy)
            if fw <= 0 or fh <= 0:
                debug_frame = frame.copy()
                cv2.putText(debug_frame, "INVALID FACE REGION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                self.last_head_down = False
                return "NO_FACE", (0.0, 0.0), debug_frame, False
            face_roi = frame[fy:fy+fh, fx:fx+fw]
            if face_roi.size == 0:
                debug_frame = frame.copy()
                cv2.putText(debug_frame, "EMPTY FACE ROI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                self.last_head_down = False
                return "NO_FACE", (0.0, 0.0), debug_frame, False
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            eyes, label = self.detect_eyes_in_face(face_roi, gray_face)

        # Reuse recent detections if current detection failed (helps with glasses glare)
        if len(eyes) < 2 and self.last_valid_eyes is not None:
            if (self.frame_counter - self.last_valid_eyes_frame) <= self.eye_retention_frames:
                eyes = [tuple(eye) for eye in self.last_valid_eyes]
                label = label or "cached"
                self.last_eye_detector = label if label == "cached" else f"{label}+cached"
            else:
                self.last_eye_detector = label
        else:
            if label:
                self.last_eye_detector = label

        if len(eyes) >= 2 and face_roi is not None and gray_face is not None:
            left_eye = eyes[0]
            right_eye = eyes[1]

            # Keep track of the most recent reliable detection
            self.last_valid_eyes = [tuple(eye) for eye in eyes]
            self.last_valid_eyes_frame = self.frame_counter

            # Check for blinking using EAR (works even when looking down)
            if self.detect_blinking(left_eye, right_eye):
                debug_frame = frame.copy()
                if face is not None:
                    fx, fy, fw, fh = face
                    cv2.rectangle(debug_frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
                    for eye in [left_eye, right_eye]:
                        eye_x = fx + eye[0]
                        eye_y = fy + eye[1]
                        eye_w = eye[2]
                        eye_h = eye[3]
                        cv2.rectangle(debug_frame, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (255, 0, 0), 2)
                cv2.putText(debug_frame, "BLINKING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                self.last_head_down = False
                return "BLINKING", (0.0, 0.0), debug_frame, False

            gaze_direction = self.calculate_gaze_direction(left_eye, right_eye, face_roi)
            rel_x, rel_y = self.compute_relative_eye_offsets(left_eye, right_eye, face_roi, gray_face)

            # Apply temporal smoothing
            self.gaze_history.append((rel_x, rel_y))
            if len(self.gaze_history) > self.max_history:
                self.gaze_history.pop(0)

            smoothed_x = np.mean([g[0] for g in self.gaze_history])
            smoothed_y = np.mean([g[1] for g in self.gaze_history])

            debug_frame = frame.copy()
            if face is not None:
                fx, fy, fw, fh = face
                cv2.rectangle(debug_frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
                for eye in [left_eye, right_eye]:
                    eye_x = fx + eye[0]
                    eye_y = fy + eye[1]
                    eye_w = eye[2]
                    eye_h = eye[3]
                    cv2.rectangle(debug_frame, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (255, 0, 0), 2)
                cv2.putText(debug_frame, f"Gaze: {gaze_direction}", (fx, max(20, fy - 15)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                detector_label = self.last_eye_detector or "unknown"
                cv2.putText(debug_frame, f"Eye detector: {detector_label}", (fx, fy + fh + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            head_down = self._estimate_head_down(face_roi, left_eye, right_eye, smoothed_y)
            self.last_head_down = head_down

            return gaze_direction, (smoothed_x, smoothed_y), debug_frame, head_down

        # If no eyes, still add to history but with zero (maintains smoothing)
        self.gaze_history.append((0.0, 0.0))
        if len(self.gaze_history) > self.max_history:
            self.gaze_history.pop(0)

        debug_frame = frame.copy()
        if self.debug_mode:
            cv2.putText(debug_frame, "NO EYES DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        self.last_head_down = False
        return "NO_EYES", (0.0, 0.0), debug_frame, False
    
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
    
    def run(self, camera_index=None, headless=False):
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
        
        # Simple camera opening - same as test script that works
        print(f"Opening camera {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            print("\nTroubleshooting:")
            print("  1. Make sure your webcam is connected and not in use by another app")
            print("  2. Try running as administrator")
            print("  3. Check camera permissions in Windows settings")
            print("  4. Try specifying a different camera: --list to see available cameras")
            return
        
        # Read a few frames to initialize (like test script)
        print("Initializing camera...")
        for i in range(5):
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                brightness = np.mean(test_frame)
                print(f"Frame {i+1}: brightness = {brightness:.2f}")
                if brightness > 5.0:
                    print("Camera initialized successfully!")
                    break
            time.sleep(0.1)
        
        # Try to set resolution (but don't fail if it doesn't work)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            except:
                pass
        
        print("Simple Gaze Tracker Started!")
        print("Press 'q' to quit | 'h' for help | 'w/a/s/r' to calibrate edges (up/left/down/right) | 'c' center")
        print("Gaze directions: LEFT, CENTER, RIGHT, UP, DOWN, UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT, BLINKING, NO_FACE, NO_EYES")
        print("-" * 80)

        if headless:
            print("Console mode active: gaze status will update on this line.")
            print("Controls: q quit | h help | w/a/s/r edges | c center")
            self.console_line_length = 0
            original_frame_skip = self.frame_skip
            self.frame_skip = 1
        else:
            cv2.namedWindow('Eye Tracker - Camera Feed', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Eye Tracker - Camera Feed', 640, 480)
        
        last_direction = None
        
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
                
                # Debug: Print frame info on first successful read and check if frame has content
                if not hasattr(self, 'frame_info_printed'):
                    print(f"Frame read successfully! Shape: {frame.shape}, Size: {frame.size}, Dtype: {frame.dtype}")
                    # Check if frame is all black
                    frame_mean = np.mean(frame)
                    print(f"Frame mean brightness: {frame_mean:.2f} (0=black, 255=white)")
                    if frame_mean < 1.0:
                        print("WARNING: Frame appears to be completely black!")
                    self.frame_info_printed = True
                
                display_frame = frame.copy() if not headless else None
                
                # Frame skipping - only process every Nth frame
                self.frame_counter += 1
                if self.frame_counter % self.frame_skip != 0:
                    # Still update display but use last processed gaze values
                    if not hasattr(self, 'last_gaze_result'):
                        if headless:
                            self._print_console_status("INITIALIZING", (0.0, 0.0))
                            time.sleep(0.01)
                        else:
                            cv2.putText(display_frame, "Initializing...", (10, 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            cv2.imshow('Eye Tracker - Camera Feed', display_frame)
                            cv2.waitKey(1)
                        continue  # Skip processing if no previous result
                    result = self.last_gaze_result
                else:
                    # Process frame
                    result = self.process_frame(frame)
                    # Store for frame skipping
                    self.last_gaze_result = result
                
                head_down = getattr(self, "last_head_down", False)

                # Unpack result (may include debug frame and head pose)
                if isinstance(result, tuple):
                    if len(result) == 4:
                        gaze_direction, rel, debug_frame, head_down = result
                        if not headless and debug_frame is not None and debug_frame.size > 0:
                            display_frame = debug_frame.copy()
                    elif len(result) == 3:
                        gaze_direction, rel, debug_frame = result
                        if not headless and debug_frame is not None and debug_frame.size > 0:
                            display_frame = debug_frame.copy()
                        head_down = getattr(self, "last_head_down", False)
                    elif len(result) == 2:
                        gaze_direction, rel = result
                        head_down = getattr(self, "last_head_down", False)
                    else:
                        gaze_direction = result[0]
                        rel = result[1] if len(result) > 1 else (0.0, 0.0)
                else:
                    gaze_direction, rel = result, (0.0, 0.0)
                    head_down = getattr(self, "last_head_down", False)
                
                self.last_head_down = head_down
                
                # Only print when direction changes to reduce spam (GUI mode)
                if gaze_direction != last_direction:
                    if not headless:
                        timestamp = time.strftime("%H:%M:%S")
                        print(f"[{timestamp}] Gaze: {gaze_direction}")
                    last_direction = gaze_direction
                
                # Update down-gaze timer
                if isinstance(rel, (tuple, list)) and len(rel) >= 2:
                    self._update_down_timer(gaze_direction, rel[1])
                else:
                    self._update_down_timer(gaze_direction, 0.0)
                
                if headless:
                    self.last_head_down = head_down
                    self._print_console_status(gaze_direction, rel)
                else:
                    # Add status text overlay to display frame
                    if display_frame is not None and display_frame.size > 0:
                        status_color = (0, 255, 0) if gaze_direction != "NO_FACE" else (0, 0, 255)
                        cv2.putText(display_frame, f"Status: {gaze_direction}", (10, display_frame.shape[0] - 40), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                        detector_label = self.last_eye_detector or "unknown"
                        cv2.putText(display_frame, f"Eye detector: {detector_label}", (10, display_frame.shape[0] - 65), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                        head_msg = "Head: DOWN" if head_down else "Head: UP"
                        cv2.putText(display_frame, head_msg, (10, display_frame.shape[0] - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                        cv2.putText(display_frame, "Controls: q quit | h help | w/a/s/r edges | c center", 
                                    (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                        cv2.imshow('Eye Tracker - Camera Feed', display_frame)
                    else:
                        # Fallback: create a test pattern if frame is invalid
                        print("Warning: Invalid frame, creating test pattern")
                        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(test_frame, "No valid frame", (50, 240), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.imshow('Eye Tracker - Camera Feed', test_frame)
                
                # Monitor memory usage and enforce limit
                exceeded, current_mem = self._check_memory_usage()
                if exceeded:
                    print(f"Memory limit exceeded: {current_mem / (1024**3):.2f} GiB > {self.memory_limit_bytes / (1024**3):.2f} GiB")
                    print("Stopping gaze tracker to avoid excessive RAM usage.")
                    break
                
                # Handle keyboard input
                if headless:
                    key_input = self._poll_console_key()
                else:
                    key_code = cv2.waitKey(1)
                    key_input = key_code if key_code != -1 else None
                key_char = self._translate_key_value(key_input)
                can_calibrate = isinstance(rel, (tuple, list)) and len(rel) >= 2
                
                if key_char == 'q':
                    break
                elif key_char == 'h':
                    print("Help: Use w(up), s(down), a(left), r(right) to set calibration edges to the CURRENT gaze; c sets the center (averages).")
                elif key_char == 'a' and can_calibrate:
                    self.calib_min[0] = rel[0]
                    print(f"Calib left set to {self.calib_min[0]:.3f}")
                elif key_char == 'r' and can_calibrate:
                    self.calib_max[0] = rel[0]
                    print(f"Calib right set to {self.calib_max[0]:.3f}")
                elif key_char == 'w' and can_calibrate:
                    self.calib_min[1] = rel[1]
                    print(f"Calib top set to {self.calib_min[1]:.3f}")
                elif key_char == 's' and can_calibrate:
                    self.calib_max[1] = rel[1]
                    print(f"Calib bottom set to {self.calib_max[1]:.3f}")
                elif key_char == 'c' and can_calibrate:
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
            if cap:
                cap.release()
            if headless:
                sys.stdout.write("\n")
                sys.stdout.flush()
                self.frame_skip = original_frame_skip
            cv2.destroyAllWindows()
            if self.use_mediapipe and self.mediapipe_face_mesh is not None:
                try:
                    self.mediapipe_face_mesh.close()
                except Exception:
                    pass

    def _trigger_down_beep(self):
        """Play an audible alert if available when user looks down too long."""
        if _WINSOUND_AVAILABLE and winsound is not None:
            try:
                winsound.Beep(880, 300)  # frequency in Hz, duration in ms
            except RuntimeError:
                pass
        else:
            # Fallback: console bell
            print('\a', end='', flush=True)

    def _update_down_timer(self, gaze_direction, rel_y):
        """Accumulate time spent looking down and trigger alert when threshold exceeded."""
        now = time.time()
        dt = max(0.0, now - self.last_frame_time)
        self.last_frame_time = now
        down_directions = {"DOWN", "DOWN_LEFT", "DOWN_RIGHT"}
        is_down = gaze_direction in down_directions or rel_y > self.down_offset_threshold
        if is_down:
            self.down_accumulator += dt
            if self.down_accumulator >= self.down_threshold_seconds:
                if (self.last_down_beep_time is None or
                        now - self.last_down_beep_time >= self.down_beep_cooldown):
                    self._trigger_down_beep()
                    self.last_down_beep_time = now
                    # leave accumulator as-is so we only beep again after cooldown
        else:
            # Allow accumulator to decay toward zero to avoid immediate trigger on brief glances
            self.down_accumulator = max(0.0, self.down_accumulator - dt)

    def _translate_key_value(self, key_value):
        """Normalize key inputs from OpenCV or console polling into lowercase chars."""
        if key_value is None:
            return None
        if isinstance(key_value, str):
            return key_value.lower()
        if isinstance(key_value, int):
            if key_value < 0:
                return None
            key_value &= 0xFF
            if key_value in (0xFF, 0x00):
                return None
            try:
                ch = chr(key_value)
            except (ValueError, OverflowError):
                return None
            return ch.lower()
        return None

    def _poll_console_key(self):
        """Non-blocking console key poll for headless mode."""
        if _MSVCRT_AVAILABLE and msvcrt is not None:
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ('\000', '\xe0'):
                    # Special key prefix, consume second char and ignore
                    if msvcrt.kbhit():
                        msvcrt.getwch()
                    return None
                return ch.lower()
            return None
        try:
            import select  # type: ignore
        except ImportError:
            return None
        try:
            if select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                return ch.lower()
        except Exception:
            return None
        return None

    def _print_console_status(self, gaze_direction, rel):
        """Display gaze information on a single console line."""
        if isinstance(rel, (tuple, list)) and len(rel) >= 2:
            rel_x, rel_y = float(rel[0]), float(rel[1])
        else:
            rel_x = rel_y = 0.0
        detector_label = self.last_eye_detector or "unknown"
        head_status = "HEAD_DOWN" if getattr(self, "last_head_down", False) else "HEAD_UP "
        status = (
            f"Gaze: {gaze_direction:<12} "
            f"rel=({rel_x:+0.3f},{rel_y:+0.3f}) "
            f"detector={detector_label} "
            f"{head_status}"
        )
        padding = max(0, self.console_line_length - len(status))
        sys.stdout.write("\r" + status + " " * padding)
        sys.stdout.flush()
        self.console_line_length = max(self.console_line_length, len(status))

    def _estimate_head_down(self, face_roi, left_eye, right_eye, smoothed_y):
        """Estimate if the head is pitched downward using facial geometry."""
        if face_roi is None or face_roi.size == 0:
            return smoothed_y > 0.35

        face_height, face_width = face_roi.shape[:2]
        if face_height <= 0 or face_width <= 0:
            return smoothed_y > 0.35

        left_center_y = left_eye[1] + left_eye[3] / 2.0
        right_center_y = right_eye[1] + right_eye[3] / 2.0
        eye_center_y = (left_center_y + right_center_y) / 2.0
        eye_position_ratio = eye_center_y / max(1.0, float(face_height))

        face_aspect_ratio = face_height / max(1.0, float(face_width))
        prev_baseline_eye = getattr(self, "head_eye_ratio_baseline", 0.45)
        prev_baseline_face = getattr(self, "head_face_ratio_baseline", 1.05)
        alpha = getattr(self, "head_baseline_alpha", 0.12)

        was_head_down = getattr(self, "last_head_down", False)
        if not was_head_down:
            if 0.22 <= eye_position_ratio <= 0.6:
                new_eye_baseline = (1.0 - alpha) * prev_baseline_eye + alpha * eye_position_ratio
                self.head_eye_ratio_baseline = float(np.clip(new_eye_baseline, 0.3, 0.6))
            if 0.8 <= face_aspect_ratio <= 1.6:
                new_face_baseline = (1.0 - alpha) * prev_baseline_face + alpha * face_aspect_ratio
                self.head_face_ratio_baseline = float(np.clip(new_face_baseline, 0.9, 1.5))

        eye_delta = eye_position_ratio - self.head_eye_ratio_baseline
        face_delta = face_aspect_ratio - self.head_face_ratio_baseline

        downward_eye = eye_delta > 0.04 or smoothed_y > 0.35
        elongated_face = face_delta > 0.08

        return downward_eye or elongated_face

def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ultra-Lightweight Gaze Tracker for Raspberry Pi and desktop systems."
    )
    parser.add_argument(
        '--camera',
        type=int,
        help="Camera index to use (default: auto-detect first available camera)",
        default=None
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help="List available cameras and exit"
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        help="Run in console mode without opening the camera preview window"
    )
    args = parser.parse_args()

    if args.list:
        tracker = SimpleGazeTracker()
        tracker.list_cameras()
        return

    print("Initializing Improved Gaze Tracker...")
    print("Features: Pupil detection, temporal smoothing, frame skipping")
    print("-" * 80)

    tracker = SimpleGazeTracker()
    tracker.run(camera_index=args.camera, headless=args.headless)

if __name__ == "__main__":
    main()

# Lightweight Gaze Tracker for Raspberry Pi

A very lightweight gaze tracking system that detects eye movements and prints gaze direction to the terminal. Designed specifically for Raspberry Pi with USB webcam.

## Features

- **Lightweight**: Optimized for Raspberry Pi performance
- **Real-time**: Continuous gaze direction detection
- **Terminal Output**: Prints gaze direction (LEFT, CENTER, RIGHT, BLINKING, NO_FACE)
- **Blink Detection**: Distinguishes between gaze changes and blinking
- **No GUI Required**: Runs headlessly for terminal-only operation

## Gaze Directions

- `LEFT`: Looking to the left
- `CENTER`: Looking straight ahead
- `RIGHT`: Looking to the right
- `BLINKING`: Eyes are closed/blinking
- `NO_FACE`: No face detected in frame

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Face Landmark Model

Download the dlib face landmark predictor model:

```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

Or manually download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

### 3. Raspberry Pi Specific Setup

For Raspberry Pi, you may need to install additional dependencies:

```bash
sudo apt-get update
sudo apt-get install python3-opencv python3-pip
sudo apt-get install libopenblas-dev liblapack-dev
sudo apt-get install libx11-dev libgtk-3-dev libboost-python-dev
```

## Usage

### Basic Usage

```bash
python3 gaze_tracker.py
```

### With Different Camera

If you have multiple cameras, specify the camera index:

```bash
python3 gaze_tracker.py
```

The script will automatically use camera index 0 (default camera).

### Headless Operation

The script is designed to run without a display. The video window is commented out by default. If you want to see the video feed for debugging, uncomment the `cv2.imshow` line in the code.

## Performance Optimization

### For Raspberry Pi

1. **Reduce Resolution**: Modify the camera resolution in the code if needed
2. **Skip Frames**: Add frame skipping logic for better performance
3. **Lower FPS**: Reduce processing frequency if CPU usage is too high

### Example Optimizations

```python
# In the run() method, add frame skipping:
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process every 3rd frame for better performance
    if frame_count % 3 == 0:
        gaze_direction = self.process_frame(frame)
        # ... rest of processing
    
    frame_count += 1
```

## Troubleshooting

### Common Issues

1. **"shape_predictor_68_face_landmarks.dat not found"**
   - Download the model file as described in installation step 2

2. **"Could not open camera"**
   - Check if camera is connected and recognized
   - Try different camera indices (0, 1, 2, etc.)
   - On Linux: `ls /dev/video*` to list available cameras

3. **Poor Performance on Raspberry Pi**
   - Reduce camera resolution
   - Add frame skipping
   - Close other applications

4. **No Face Detection**
   - Ensure good lighting
   - Face should be clearly visible
   - Try adjusting camera angle

### Camera Testing

Test your camera separately:

```python
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
print(f"Camera working: {ret}")
cap.release()
```

## Technical Details

- **Face Detection**: Uses dlib's HOG-based face detector
- **Landmark Detection**: 68-point facial landmark model
- **Gaze Calculation**: Based on eye corner and pupil position ratios
- **Blink Detection**: Eye Aspect Ratio (EAR) thresholding

## License

This project is open source. The dlib library and its models have their own licensing terms.

## Contributing

Feel free to submit issues and enhancement requests!

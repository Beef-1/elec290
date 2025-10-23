# ELEC 290 - Electronics and Computer Engineering Projects

This repository contains projects developed for ELEC 290, focusing on computer vision and sensor technologies.

## Projects Overview

### 1. Gaze Tracking System
A lightweight gaze tracking system designed for real-time eye movement detection and direction analysis.

**Features:**
- Real-time gaze direction detection (LEFT, CENTER, RIGHT, UP, DOWN, and combinations)
- Blink detection and face detection
- Two implementation approaches:
  - **Advanced**: Uses dlib with 68-point facial landmark detection (higher accuracy)
  - **Simple**: Uses OpenCV Haar cascades (lighter weight, cross-platform)
- Optimized for Raspberry Pi deployment
- Terminal-based output for headless operation

**Use Cases:**
- Human-computer interaction research
- Accessibility applications
- Eye movement analysis
- Hands-free device control

### 2. Capacitive Touch Sensor
A capacitive touch sensing system for detecting touch interactions.

**Status:** In development

## Quick Start

### Gaze Tracking

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd elec290
   ```

2. **Choose your implementation:**
   - For **Raspberry Pi/Linux** (recommended): Use `gaze tracking/gaze_tracker.py`
   - For **Cross-platform/Windows**: Use `gaze tracking/gaze_tracker_simple.py`

3. **Install dependencies:**
   ```bash
   cd "gaze tracking"
   pip install -r requirements.txt
   ```

4. **For dlib version only** - Download the face landmark model:
   ```bash
   wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   bunzip2 shape_predictor_68_face_landmarks.dat.bz2
   ```

5. **Run the tracker:**
   ```bash
   # For dlib version (Linux/Raspberry Pi)
   python3 gaze_tracker.py
   
   # For simple version (cross-platform)
   python3 gaze_tracker_simple.py
   ```

## Technical Details

### Gaze Tracking Algorithms

**Advanced Implementation (dlib):**
- Uses dlib's HOG-based face detector
- 68-point facial landmark detection
- Eye Aspect Ratio (EAR) for blink detection
- Gaze ratio calculation based on eye corner and pupil positions

**Simple Implementation (OpenCV):**
- Haar cascade classifiers for face and eye detection
- Geometric analysis of eye positions relative to face center
- Simplified blink detection based on eye visibility

### Performance Considerations

- **Raspberry Pi Optimization**: Frame skipping, reduced resolution
- **Real-time Processing**: ~10-30 FPS depending on hardware
- **Memory Usage**: Minimal footprint for embedded applications

## Project Structure

```
elec290/
├── README.md                           # This file
├── gaze tracking/                      # Gaze tracking system
│   ├── README.md                      # Detailed gaze tracking documentation
│   ├── gaze_tracker.py                # Advanced dlib-based implementation
│   ├── gaze_tracker_simple.py         # Simple OpenCV-based implementation
│   └── requirements.txt               # Python dependencies
└── capacitive touch sensor/           # Touch sensor project (in development)
```

## Dependencies

### Gaze Tracking
- **Core**: OpenCV, NumPy
- **Advanced**: dlib (for facial landmark detection)
- **Platform**: Python 3.7+

### System Requirements
- **Camera**: USB webcam or built-in camera
- **OS**: Linux (recommended), Windows, macOS
- **Hardware**: Raspberry Pi 3B+ or better (for embedded deployment)

## Applications

### Research & Development
- Human-computer interaction studies
- Eye movement pattern analysis
- Accessibility technology development

### Practical Applications
- Hands-free device control
- Driver attention monitoring
- Gaming interfaces
- Medical diagnostics

## Contributing

This is an academic project repository. For contributions or questions:
1. Check existing issues
2. Create detailed bug reports or feature requests
3. Follow the existing code style and documentation standards

## License

This project is developed for educational purposes. Please respect the licensing terms of included libraries:
- OpenCV: Apache 2.0 License
- dlib: Boost Software License
- NumPy: BSD License

## Course Information

**Course**: ELEC 290 - Electronics and Computer Engineering  
**Focus**: Computer vision, sensor technologies, embedded systems  
**Platform**: Cross-platform with Raspberry Pi optimization

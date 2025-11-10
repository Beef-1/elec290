# Lightweight Gaze Tracker

Ultra-lightweight gaze tracking utility built around OpenCV Haar cascades (and optional MediaPipe landmarks). It can show a window with live annotations or run headlessly and stream text updates to the console—perfect for laptops, desktops, or a Raspberry Pi with a USB camera.

## Installation

```bash
python -m venv .venv
. .venv/bin/activate   # Windows: .\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows you may need the precompiled OpenCV wheels (already covered by `requirements.txt`). On Raspberry Pi see the dedicated notes below.

## Running the Tracker

From the `gaze tracking` directory:

- GUI with annotated camera feed  
  ```bash
  python gaze_tracker_simple.py
  ```

- Headless console output (no OpenCV window)  
  ```bash
  python gaze_tracker_simple.py --headless
  ```

- Explicit camera index  
  ```bash
  python gaze_tracker_simple.py --camera 1
  ```

- List available cameras  
  ```bash
  python gaze_tracker_simple.py --list
  ```

### Keyboard Controls (GUI or headless)

`q` quit · `h` help · `w / a / s / r` capture top/left/bottom/right edges · `c` recenter. Run the calibration keys while looking at the corresponding screen edges to fine-tune left/right/up/down detection.

## Adjusting Sensitivity & Thresholds

Most tuning knobs live near the top of `SimpleGazeTracker.__init__` in `gaze_tracker_simple.py`.

| Setting | Purpose | Typical tweak |
| --- | --- | --- |
| `self.down_offset_threshold` | How far the gaze `rel_y` must drop before counting as “eyes down” | Lower for more sensitivity, raise to avoid false positives |
| `self.down_threshold_seconds` | Dwell time before the monitoring beep fires | Lower for faster alerts (default 1.0 s) |
| `self.head_eye_ratio_baseline`, `self.head_face_ratio_baseline` | Initial head-pitch baselines used by `_estimate_head_down` | Set closer to your neutral posture if you always sit higher/lower |
| `self.head_baseline_alpha` | Learning rate when updating the baseline while the head is up | Smaller = slower adaptation |
| `_is_valid_eye_box()` constants (`min_area_ratio`, `min_vertical`, aspect thresholds) | Filters out bogus eye detections (ears, nose bridge) | Relax very cautiously if eyes are frequently missed; tightening reduces false positives |

Workflow:
1. Open `gaze_tracker_simple.py`.
2. Locate the attributes above in `__init__` or in `_is_valid_eye_box`.
3. Adjust values, save, and rerun the script (`--headless` if you prefer console mode).

Tip: keep notes of the original defaults so you can revert quickly if a change makes detection worse.

## Running on Raspberry Pi

The tracker has been exercised on Raspberry Pi 4/5 using the 64-bit OS. Steps:

1. **Prepare the OS**
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-opencv python3-numpy libatlas-base-dev
   ```
   If you plan to use MediaPipe for improved landmark quality: `pip install mediapipe-rpi4` (or the appropriate whl for your Pi).

2. **Enable the camera** (if using the CSI ribbon connector) via `raspi-config` → Interface Options → Camera. Reboot if prompted.

3. **Clone / copy the project** to the Pi and install Python dependencies  
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Launch the tracker**
   ```bash
   python3 gaze_tracker_simple.py --headless
   ```
   Headless mode avoids driving the X11 display and runs every frame for smoother console output. Add `--camera N` if you need to target `/dev/videoN`.

5. **Performance tips**
   - Reduce resolution via `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)` / height (already done for 640×480; feel free to lower further).
   - Increase `self.frame_skip` in `__init__` if you need to process fewer frames per second (headless mode temporarily overrides it to 1 for responsiveness).
   - Use good lighting—Haar cascades and pupils are more reliable with even illumination.

## Troubleshooting

- **No camera found**: run `python gaze_tracker_simple.py --list` and confirm the expected index appears; on Linux check `/dev/video*`.
- **Immediate `NO_FACE` status**: adjust lighting or distance; confirm the camera feed is not black (`Frame mean brightness` print).
- **Eyes not detected**: tweak `_is_valid_eye_box` thresholds or temporarily lower `min_area_ratio` to confirm the boxes are in range.
- **Frequent false head-down alerts**: raise `self.down_offset_threshold` slightly or increase `self.head_baseline_alpha` so the baseline adapts to your posture faster.

## License & Contributions

Open source—feel free to file issues or PRs. Remember that OpenCV, MediaPipe, and any optional models have their own licenses; review them before redistribution.

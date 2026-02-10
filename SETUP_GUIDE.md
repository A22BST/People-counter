# Complete Setup Guide: People Counter System

This guide will get you from zero to a working people counter.

---

## Step 1: Prerequisites

### Required Software

1. **Python 3.9 or higher**
   ```bash
   # Check your version
   python --version
   
   # If not installed, download from python.org
   ```

2. **pip** (Python package manager - usually comes with Python)
   ```bash
   pip --version
   ```

### Hardware

- **Camera**: Any of these work:
  - Webcam (built-in or USB)
  - iPhone with DroidCam app
  - Android with DroidCam app
  - IP camera with RTSP/HTTP stream
  - Video file (for testing)

---

## Step 2: Create Project Folder

```bash
# Create folder
mkdir people_counter
cd people_counter

# Create virtual environment (recommended)
python -m venv venv

# Activate it:
# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

---

## Step 3: Download the Files

Save these files in your `people_counter` folder:

```
people_counter/
├── direction_detector.py    # Core detection algorithm
├── people_counter.py        # Main application
├── requirements.txt         # Dependencies
├── bytetrack.yaml          # Tracker config
└── test_direction_detector.py  # Tests (optional)
```

---

## Step 4: Install Dependencies

```bash
# Make sure virtual environment is activated, then:
pip install -r requirements.txt
```

This installs:
- `ultralytics` - YOLOv8 detection and ByteTrack tracking
- `opencv-python` - Camera and video handling
- `numpy` - Math operations
- `matplotlib` - Diagnostic plots

First run will also download the YOLOv8 model (~25MB).

---

## Step 5: Test with Webcam

```bash
python people_counter.py --source 0
```

You should see:
- A window with your webcam feed
- Yellow line showing the "door threshold"
- Boxes around detected people
- Stats overlay in top-left corner

**Controls:**
- `q` - Quit and generate diagnostic report
- `r` - Reset counts
- `s` - Print stats to console

---

## Step 6: Setup with DroidCam (iPhone/Android)

### On Your Phone:

1. Install **DroidCam** from App Store / Play Store
2. Open the app
3. Note the **IP address** shown (e.g., `192.168.1.100`)
4. Keep the app open

### On Your Computer:

```bash
# Replace with your phone's IP
python people_counter.py --source "http://192.168.1.100:4747/video"
```

### Tips:
- Phone and computer must be on **same WiFi network**
- Position phone on door frame, camera pointing **outside**
- Use a phone mount or tape to secure it

---

## Step 7: Camera Positioning

**Critical for accuracy!**

```
        OUTSIDE (where people come from)
              │
              │  People approach from here
              │
    ┌─────────┴─────────┐
    │    📷 CAMERA      │  ← Mount here, pointing outside
    │    (on door frame)│
    │                   │
    │   ┌───────────┐   │
    │   │           │   │
    │   │   Door    │   │
    │   │  Opening  │   │
    │   │           │   │
    │   └───────────┘   │
    │                   │
    └───────────────────┘
              │
        INSIDE (mosque)
```

### Best Position:
- **Top of door frame**, angled slightly down
- Camera should see people's **front/side** (not top of head)
- Bottom of video frame = door threshold

---

## Step 8: Run and Calibrate

### First Run - Gather Data

```bash
python people_counter.py --source "http://YOUR_IP:4747/video"
```

Let it run for a few minutes with real traffic. Press `q` to quit.

### Read the Diagnostic Report

The report shows:
```
⚙️ RECOMMENDED THRESHOLDS
   near_edge_threshold: 0.87      ← Use this value!
   spawn_far_threshold: 0.35
   area_growth_for_approach: 1.45
```

### Apply the Recommendations

```bash
python people_counter.py --source "http://YOUR_IP:4747/video" --near-threshold 0.87
```

---

## Step 9: Verify Accuracy

Watch the live view and verify:

| What You See | What Should Happen |
|--------------|-------------------|
| Person walks toward door, enters | Green "IN" flash, entry count +1 |
| Person walks out from door | Red "OUT" flash, exit count +1 |
| Person approaches then turns around | No count (correctly ignored) |
| Two people enter together | Should count 2 entries |

---

## Step 10: Production Use

### Run Without Display (headless)

```bash
python people_counter.py --source "http://YOUR_IP:4747/video" --no-display
```

### Save Events to File

```bash
python people_counter.py --source "http://YOUR_IP:4747/video" --output events.json
```

### Full Command with All Options

```bash
python people_counter.py \
  --source "http://192.168.1.100:4747/video" \
  --model yolov8s.pt \
  --confidence 0.3 \
  --near-threshold 0.85 \
  --diagnostics-dir ./reports \
  --output events.json
```

---

## Troubleshooting

### "No module named 'ultralytics'"
```bash
pip install ultralytics
```

### "Cannot open video source"
- Check IP address is correct
- Check phone and computer on same WiFi
- Check DroidCam app is running
- Try opening URL in browser: `http://192.168.1.100:4747/video`

### Low Detection Rate
- Increase sensitivity: `--confidence 0.25`
- Check lighting (needs adequate light)
- Adjust camera angle (avoid pure overhead view)

### Wrong Direction Detection
- Check camera is pointing **outside** (not inside)
- Adjust `--near-threshold` based on diagnostic report
- Make sure bottom of frame is at door threshold

### Too Many False Positives
- Decrease sensitivity: `--confidence 0.4`
- Check for reflections or moving objects in view

---

## File Structure Explained

```
people_counter/
│
├── direction_detector.py     # THE BRAIN
│   │                         # - TrackState machine
│   │                         # - Area/position analysis
│   │                         # - Crossing classification
│   │                         # - Confidence scoring
│   │
├── people_counter.py         # THE BODY
│   │                         # - Camera input
│   │                         # - YOLO detection
│   │                         # - ByteTrack tracking
│   │                         # - Visualization
│   │                         # - Diagnostic reporting
│   │                         # - Standby Slot Method
│   │
├── requirements.txt          # Dependencies list
│
├── bytetrack.yaml           # Tracker settings
│
└── test_direction_detector.py  # Unit tests
```

### How They Connect

```python
# people_counter.py does this:
from direction_detector import DirectionDetector

# Creates instance:
self.detector = DirectionDetector(config)

# Every frame:
events = self.detector.update(detections)
# Returns: [{"direction": "IN", "confidence": "HIGH"}, ...]
```

---

## Command Line Options Reference

| Option | Default | Description |
|--------|---------|-------------|
| `--source` | `0` | Camera source (0=webcam, URL, or file) |
| `--model` | `yolov8s.pt` | YOLO model to use |
| `--confidence` | `0.3` | Detection threshold (0.0-1.0) |
| `--near-threshold` | `0.85` | Door threshold position (0.0-1.0) |
| `--no-display` | off | Run without video window |
| `--no-diagnostics` | off | Skip diagnostic report |
| `--no-plots` | off | Skip diagnostic plots |
| `--diagnostics-dir` | `.` | Where to save reports |
| `--output` | none | JSON file for events |

---

## Next Steps

1. **Test with your setup** - Run for 10+ minutes with real traffic
2. **Review diagnostics** - Check confidence rates and patterns
3. **Tune thresholds** - Apply recommended values
4. **Deploy** - Mount camera permanently, run headless

For questions or issues, check the diagnostic report first - it usually shows what's wrong!

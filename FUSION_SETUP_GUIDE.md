# Complete Setup Guide: People Counter with Sensor Fusion

This guide covers the full integrated system: **Camera (YOLOv8) + ESP32 Sensors (Radar/ToF)**

**Version 2.0 Features:**
- Direction flip (swap IN↔OUT for different mounting orientations)
- Calibration mode (see raw sensor coordinates for mapping)
- Negative occupancy fix (handles uncounted entries)
- Infinite standby slots (optional timeout)
- Real ToF support via SparkFun VL53L5CX library

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        YOUR COMPUTER                            │
│                                                                 │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │   Camera     │      │   Sensor     │      │   People     │  │
│  │  (iPhone/    │─────▶│   Fusion     │─────▶│   Counter    │  │
│  │   Webcam)    │      │   Engine     │      │   Output     │  │
│  └──────────────┘      └──────┬───────┘      └──────────────┘  │
│                               │                                 │
│                               │ USB Serial                      │
└───────────────────────────────┼─────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │        ESP32          │
                    │  ┌───────┐ ┌───────┐  │
                    │  │ Radar │ │  ToF  │  │
                    │  └───────┘ └───────┘  │
                    └───────────────────────┘
                                │
                    ════════════╪════════════  ← Door Threshold
                                │
```

**How it works:**
1. **Camera** detects and tracks people, determines direction (IN/OUT)
2. **Radar/ToF sensors** detect threshold crossings
3. **Fusion engine** combines both signals for higher accuracy
4. **Standby Slot Method** counts unique visitors

---

## Part 1: Hardware Setup

### Components Needed

| Component | Purpose | Approx. Cost |
|-----------|---------|--------------|
| ESP32 DevKit V4 | Microcontroller | $8 |
| HLK-LD6001A | 60GHz mmWave Radar | $15 |
| VL53L5CX (optional) | 8x8 ToF Sensor | $25 |
| Breadboard + Jumper Wires | Connections | $5 |
| USB Cable | ESP32 to PC | $3 |
| Phone/Webcam | Camera input | (existing) |

### Wiring Diagram

```
                    ESP32 DevKit V4
                   ┌───────────────┐
                   │           3V3 │──────┬──────────── Radar VCC
                   │           GND │──────┼──────┬───── Radar GND
                   │               │      │      │
                   │       GPIO 16 │──────│──────│───── Radar TX (→ ESP RX)
                   │       GPIO 17 │──────│──────│───── Radar RX (← ESP TX)
                   │        GPIO 4 │──────│──────│───── Radar RST (optional)
                   │               │      │      │
                   │       GPIO 21 │──────│──────│───── ToF SDA ─┐
                   │       GPIO 22 │──────│──────│───── ToF SCL  │ I2C
                   │       GPIO 19 │──────│──────│───── ToF INT  │
                   │       GPIO 18 │──────│──────│───── ToF LPN ─┘
                   │               │      │      │
                   │           3V3 │──────┘      └───── ToF VCC
                   │           GND │─────────────────── ToF GND
                   │               │
                   │           USB │ ← To Computer
                   └───────────────┘
```

### Physical Mounting

```
        OUTSIDE (where people come from)
              │
              ▼
    ┌─────────────────────┐
    │    ┌─────────┐      │
    │    │  📷     │      │  ← Camera on door frame, pointing outside
    │    │ Camera  │      │
    │    └─────────┘      │
    │                     │
    │   ══════════════    │  ← Door threshold line
    │   │ ESP32+Radar│    │  ← Sensors at threshold height (~1m)
    │   │    ToF     │    │
    │   ══════════════    │
    │                     │
    │      [Door]         │
    │                     │
    └─────────────────────┘
              │
        INSIDE (mosque)
```

**Sensor Placement Tips:**
- Mount radar/ToF at **chest height** (~1m) pointing across doorway
- Camera should see people's **front/side** (not top of head)
- Radar detects motion in a ~60° cone
- ToF measures distance in 8x8 grid

---

## Part 2: Software Setup

### Step 1: Create Project Folder

```bash
mkdir people_counter
cd people_counter
```

### Step 2: Download Required Files

Save these files in your `people_counter` folder:

```
people_counter/
├── people_counter.py         # Main application (modified with fusion)
├── direction_detector.py     # Direction detection algorithm
├── sensor_fusion.py          # NEW: Fusion engine
├── direction_sensor_modular.ino  # ESP32 firmware
├── bytetrack.yaml            # Tracker configuration
├── requirements.txt          # Python dependencies
├── test_direction_detector.py    # Tests (optional)
└── test_sensor_fusion.py     # Fusion tests (optional)
```

### Step 3: Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install pyserial  # For ESP32 communication
```

### Step 4: Flash ESP32 Firmware

1. **Install Arduino IDE** (if not already installed)
   - Download from https://www.arduino.cc/en/software

2. **Add ESP32 Board Support**
   - File → Preferences → Additional Board Manager URLs:
   - Add: `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
   - Tools → Board → Boards Manager → Search "esp32" → Install

3. **Open and Configure Firmware**
   - Open `direction_sensor_modular.ino` in Arduino IDE
   - Select board: Tools → Board → ESP32 Dev Module
   - Select port: Tools → Port → (your ESP32 port)

4. **Upload**
   - Click Upload button (→)
   - Wait for "Done uploading"

5. **Verify**
   - Open Serial Monitor (Tools → Serial Monitor)
   - Set baud rate to 115200
   - You should see:
   ```
   ==========================================
     Modular Direction Sensor - All Methods  
   ==========================================
   Type 'help' for commands
   ```

---

## Part 3: Find Your ESP32 Port

### Windows
1. Open Device Manager
2. Expand "Ports (COM & LPT)"
3. Look for "USB-SERIAL CH340" or "CP210x"
4. Note the COM port (e.g., `COM3`)

### Mac/Linux
```bash
# List serial ports
ls /dev/tty*

# Look for /dev/ttyUSB0 or /dev/tty.usbserial-*
```

---

## Part 4: Running the System

### Basic Usage (Camera Only - No Fusion)

```bash
# With webcam
python people_counter.py --source 0

# With DroidCam (iPhone/Android)
python people_counter.py --source "http://192.168.1.100:4747/video"
```

### With Sensor Fusion Enabled

```bash
# Windows
python people_counter.py --source 0 --fusion --fusion-port COM3

# Mac/Linux
python people_counter.py --source 0 --fusion --fusion-port /dev/ttyUSB0

# With DroidCam + Fusion
python people_counter.py \
    --source "http://192.168.1.100:4747/video" \
    --fusion \
    --fusion-port COM3
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--source` | `0` | Camera source (0=webcam, URL, or file) |
| `--fusion` | off | Enable sensor fusion with ESP32 |
| `--fusion-port` | `COM3` | ESP32 serial port |
| `--fusion-strategy` | `confirmation` | Fusion strategy (see below) |
| `--flip` | off | Flip direction: IN↔OUT swapped |
| `--standby-timeout` | `0` | Standby timeout (0=infinite, >0=seconds) |
| `--model` | `yolov8s.pt` | YOLO model to use |
| `--confidence` | `0.3` | Detection threshold |
| `--near-threshold` | `0.85` | Door threshold position |
| `--no-display` | off | Run without video window |

### Direction Flip

Use `--flip` when camera/sensors are mounted the "opposite" way:
- **Normal**: Camera sees people approaching = IN, people leaving = OUT
- **Flipped**: Camera sees people approaching = OUT, people leaving = IN

```bash
# Camera mounted looking inside (reverse of normal)
python people_counter.py --source 0 --flip

# With fusion - also tells ESP32 to flip
python people_counter.py --source 0 --fusion --flip
```

ESP32 command: `flip on` / `flip off`

### Standby Timeout

By default, standby slots **never expire** (infinite). A person who exits can return any time and be recognized as the same visitor.

To add a timeout:
```bash
# Slots expire after 5 minutes (300 seconds)
python people_counter.py --source 0 --standby-timeout 300

# Slots expire after 30 seconds
python people_counter.py --source 0 --standby-timeout 30
```

### Negative Occupancy Fix

If someone exits when occupancy is already 0 (they were never counted on entry):
1. They're added to unique visitors
2. A standby slot is created (in case they return)

This handles the case where the system started after people were already inside.

### Fusion Strategies

| Strategy | Best For | Behavior |
|----------|----------|----------|
| `confirmation` | General use (default) | Camera decides, sensors boost/reduce confidence |
| `tiebreaker` | Camera has hesitation issues | Uses sensors only when camera signals conflict |
| `crossing` | High accuracy needed | Waits for sensor to confirm threshold crossing |

```bash
# Use tiebreaker strategy
python people_counter.py --source 0 --fusion --fusion-strategy tiebreaker

# Use crossing confirmation strategy
python people_counter.py --source 0 --fusion --fusion-strategy crossing
```

---

## Part 5: Calibration Mode

Before running the full system, use calibration mode to understand how sensors map to camera coordinates.

### ESP32 Calibration

```bash
# Open Serial Monitor (115200 baud) and type:
calibrate on
```

Walk through the sensor field and observe:

**Radar output:**
```
CAL,RADAR,density=45.2,ratio=1.35,dir=approaching,first=180,second=134
```
- `ratio > 1` = approaching (first half has more activity)
- `ratio < 1` = receding (second half has more activity)

**ToF output (single mode):**
```
CAL,TOF,123,456,789,...,t=1,z=12
```
- Full 8x8 grid of distances in mm
- `t=1` = triggered, `z=12` = 12 zones active

**ToF output (zones mode):**
```
CAL,TOF_Z,A=1(8),B=0(0)
```
- Zone A (cols 0-3) vs Zone B (cols 4-7)
- `A=1(8)` = Zone A active with 8 zones triggered

### Mapping Coordinates

Walk from **outside to inside** (entry) and note:
1. Does radar show `approaching` or `receding`?
2. Does ToF show `A→B` or `B→A` sequence?

If they're backwards from expected, use `--flip` on the Python side or `flip on` on ESP32.

```bash
# Turn off calibration
calibrate off
```

---

## Part 6: Testing the Sensors

### Test ESP32 Standalone

1. Open Arduino Serial Monitor (115200 baud)
2. Type these commands:

```
status          # Show current sensor modes
raw on          # Enable raw data output
radar tripwire  # Set radar to tripwire mode
tof single      # Set ToF to single tripwire mode
```

3. Walk through the sensor field and watch for:
```
EVT,1,0.85,radar_tripwire,approaching   # Entry detected
EVT,2,0.78,radar_tripwire,receding      # Exit detected
EVT,4,1.00,tof_tripwire,12              # ToF triggered (12 zones)
```

### Test Fusion Without Camera

```bash
python sensor_fusion.py --port COM3
```

Then type commands:
```
> in       # Simulate camera IN event
> out      # Simulate camera OUT event
> status   # Show counts
> esp raw on   # Enable ESP32 raw output
```

### Run Unit Tests

```bash
# Test direction detector
python test_direction_detector.py

# Test sensor fusion
python test_sensor_fusion.py
```

---

## Part 7: Calibration

### Step 1: Run Without Fusion First

```bash
python people_counter.py --source 0
```

Let it run for 5-10 minutes with real traffic. Press `q` to quit.

### Step 2: Check Diagnostic Report

The report shows recommended thresholds:
```
⚙️ RECOMMENDED THRESHOLDS
   near_edge_threshold: 0.87
   spawn_far_threshold: 0.35
   area_growth_for_approach: 1.45
```

### Step 3: Apply Thresholds

Edit `people_counter.py` QUICK CONFIG section (top of file):
```python
NEAR_EDGE_THRESHOLD = 0.87
SPAWN_FAR_THRESHOLD = 0.35
AREA_GROWTH_FOR_APPROACH = 1.45
```

### Step 4: Enable Fusion and Test

```bash
python people_counter.py --source 0 --fusion --fusion-port COM3
```

Watch the console for fusion decisions:
```
INFO - Fusion decision: IN (conf=0.95, src=sensor_confirmed)
INFO - New unique visitor #1
```

---

## Part 8: Understanding the Output

### Live Display

```
┌─────────────────────────────┐
│ FPS: 28.5                   │
│ Entries: 12                 │
│ Exits: 8                    │
│ Occupancy: 4                │
│ Unique: 10                  │
│ Standby: 2                  │
└─────────────────────────────┘
```

| Metric | Meaning |
|--------|---------|
| Entries | Total IN crossings detected |
| Exits | Total OUT crossings detected |
| Occupancy | Current people inside (Entries - Exits) |
| Unique | Unique visitors (excludes re-entries) |
| Standby | Slots waiting for returning people |

### Console Messages

```
# Camera-only detection
INFO - Crossing: IN (confidence: HIGH)

# With fusion - sensor confirmed
INFO - Fusion decision: IN (conf=0.95, src=sensor_confirmed)

# With fusion - conflict detected
WARNING - Fusion decision: IN (conf=0.55, src=sensor_conflict)

# With fusion - waiting for crossing (crossing strategy)
DEBUG - Track 42: Waiting for sensor confirmation
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit and generate diagnostic report |
| `r` | Reset all counts |
| `s` | Print current stats to console |

---

## Part 9: Troubleshooting

### Camera Issues

| Problem | Solution |
|---------|----------|
| "Cannot open video source" | Check IP address, ensure same WiFi |
| Low detection rate | Lower confidence: `--confidence 0.25` |
| Too many false detections | Raise confidence: `--confidence 0.4` |
| Wrong direction detection | Check camera orientation, adjust thresholds |

### ESP32 Issues

| Problem | Solution |
|---------|----------|
| "Failed to connect" | Check port name, ensure ESP32 plugged in |
| No sensor events | Type `status` in Serial Monitor to check |
| Radar shows no motion | Check wiring, try `AT+DEBUG=1` command |
| ToF not responding | Check I2C wiring (SDA/SCL) |

### Fusion Issues

| Problem | Solution |
|---------|----------|
| "sensor_fusion.py not found" | Ensure file is in same folder |
| Always "camera_only" source | Check ESP32 connection, verify events with `raw on` |
| Low fusion confidence | Adjust sensor mounting, check timing |
| Counts not matching | Try different fusion strategy |

### Check ESP32 Events

```bash
# In a separate terminal, monitor ESP32 directly
# Windows:
mode COM3 baud=115200
type COM3

# Or use PuTTY/Arduino Serial Monitor
```

---

## Part 10: Production Deployment

### Headless Mode (No Display)

```bash
python people_counter.py \
    --source "http://192.168.1.100:4747/video" \
    --fusion --fusion-port COM3 \
    --no-display \
    --output events.json
```

### Auto-Start on Boot (Linux)

Create `/etc/systemd/system/people-counter.service`:
```ini
[Unit]
Description=People Counter Service
After=network.target

[Service]
ExecStart=/path/to/venv/bin/python /path/to/people_counter.py --source 0 --fusion --fusion-port /dev/ttyUSB0 --no-display
WorkingDirectory=/path/to/people_counter
Restart=always
User=pi

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl enable people-counter
sudo systemctl start people-counter
```

### Logging to File

Events are saved to JSON when using `--output`:
```json
{
  "events": [
    {"track_id": 1, "direction": "IN", "confidence": "HIGH", "timestamp": 1706634521.5},
    {"track_id": 2, "direction": "OUT", "confidence": "MEDIUM", "timestamp": 1706634545.2}
  ],
  "summary": {
    "total_entries": 15,
    "total_exits": 12,
    "unique_visitors": 13,
    "current_occupancy": 3
  }
}
```

---

## Quick Reference Card

### Files Needed
```
people_counter.py      # Main app
direction_detector.py  # Direction logic
sensor_fusion.py       # Fusion engine
bytetrack.yaml         # Tracker config
```

### Minimum Command (Camera Only)
```bash
python people_counter.py --source 0
```

### Full Command (With Fusion)
```bash
python people_counter.py \
    --source "http://192.168.1.100:4747/video" \
    --fusion \
    --fusion-port COM3 \
    --fusion-strategy confirmation \
    --near-threshold 0.85
```

### ESP32 Commands
```
status         # Show current state
raw on/off     # Toggle raw data
radar tripwire # Radar mode: direction detection
radar correlation # Radar mode: camera-guided
tof single     # ToF mode: simple tripwire
tof zones      # ToF mode: dual-zone direction
help           # Show all commands
```

---

## Need Help?

1. **Check diagnostics** - Run without fusion first, review the report
2. **Test components separately** - Camera alone, then ESP32 alone
3. **Enable raw output** - `raw on` command shows what sensors see
4. **Run tests** - `python test_sensor_fusion.py` verifies logic
5. **Check wiring** - Most hardware issues are wiring problems

Good luck with your mosque attendance system! 🕌

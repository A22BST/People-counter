"""
Quick Video Recorder - Capture Camera Footage + ToF Sensor Data

Records camera video AND ToF 8x8 grid data from ESP32 simultaneously.
Record yourself walking in/out, then review with ground_truth_annotator.py
The synced ToF data can be used for offline analysis of tracking methods.

Controls:
    R - Start/Stop recording
    Q - Quit
    S - Take screenshot

Usage:
    # Record from webcam + ToF sensor on COM5
    python quick_record.py --serial-port COM5
    
    # Record from specific camera + ToF
    python quick_record.py --source 1 --serial-port COM5
    
    # Record from DroidCam + ToF
    python quick_record.py --source "http://192.168.1.100:4747/video" --serial-port COM5
    
    # Record video only (no ToF)
    python quick_record.py
    
    # Then analyze the recording:
    python ground_truth_annotator.py --source recording_20260211_123456.mp4
    # ToF data will be in: recording_20260211_123456_tof.csv

Author: Ahmad's Mosque Attendance System
"""

import cv2
import numpy as np
import argparse
import time
import threading
import csv
import json
from datetime import datetime
from pathlib import Path

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("WARNING: pyserial not installed. ToF recording disabled.")
    print("  Install with: pip install pyserial")


class QuickRecorder:
    def __init__(self, source='0', output_dir='.', auto_start=False,
                 serial_port=None, baud_rate=115200):
        self.source = source
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.auto_start = auto_start
        
        # Video state
        self.cap = None
        self.video_writer = None
        self.recording = False
        self.recording_path = None
        self.frame_count = 0
        self.fps = 30.0
        self.frame_width = 640
        self.frame_height = 480
        self.start_time = None
        
        # Serial / ToF state
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.ser = None
        self.serial_connected = False
        self.serial_thread = None
        self.serial_running = False
        self.tof_sensor_ok = False     # Whether ESP32 reports ToF initialized
        
        # ToF data buffer (thread-safe)
        self.tof_lock = threading.Lock()
        self.tof_buffer = []          # Accumulated during recording
        self.tof_csv_writer = None
        self.tof_csv_file = None
        self.tof_path = None
        self.tof_frame_count = 0
        self.tof_last_frame = None    # Latest frame for display
        self.tof_fps = 0.0
        self._tof_fps_counter = 0
        self._tof_fps_time = time.time()
        self._serial_lines_received = 0  # Total lines from serial
        self._serial_last_line = ''      # Last non-TOF line for debug
        
        # Also capture EVT lines from the sensor
        self.evt_buffer = []
        
    def open_camera(self):
        """Open video source."""
        source = self.source
        if source.isdigit():
            source = int(source)
        
        print(f"Opening camera: {source}")
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera: {self.source}")
        
        # Get camera properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        print(f"Camera opened: {self.frame_width}x{self.frame_height} @ {self.fps:.1f} FPS")
        print("\n" + "="*60)
        print("QUICK RECORDER - Ready")
        print("="*60)
        print("Controls:")
        print("  R - Start/Stop recording")
        print("  S - Screenshot")
        print("  Q - Quit")
        if self.serial_connected:
            print(f"  ToF: Connected on {self.serial_port}")
            if self.tof_sensor_ok:
                print(f"  ToF Sensor: OK (streaming)")
            else:
                print(f"  ToF Sensor: NOT INITIALIZED on ESP32!")
                print(f"    -> Check I2C wiring (SDA=21, SCL=22)")
                print(f"    -> Restart ESP32 and check Serial Monitor")
        else:
            print("  ToF: Not connected (video only)")
        print("="*60 + "\n")
    
    # ================================================================
    # SERIAL / ToF METHODS
    # ================================================================
    
    def open_serial(self):
        """Open serial connection to ESP32."""
        if not SERIAL_AVAILABLE:
            print("pyserial not installed - ToF recording disabled")
            return False
        
        if not self.serial_port:
            # Try to auto-detect ESP32
            self.serial_port = self._auto_detect_port()
            if not self.serial_port:
                print("No serial port specified and auto-detect failed.")
                print("  Use --serial-port COM5 (or your port)")
                print("  Available ports:")
                for p in serial.tools.list_ports.comports():
                    print(f"    {p.device} - {p.description}")
                return False
        
        try:
            self.ser = serial.Serial(
                port=self.serial_port,
                baudrate=self.baud_rate,
                timeout=0.1
            )
            self.serial_connected = True
            print(f"Serial connected: {self.serial_port} @ {self.baud_rate} baud")
            
            # Flush any boot messages
            time.sleep(1.0)
            self.ser.reset_input_buffer()
            
            # Query ToF status from ESP32
            self.ser.write(b'status\n')
            time.sleep(0.5)
            status_lines = []
            while self.ser.in_waiting:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    status_lines.append(line)
                    print(f"  ESP32> {line}")
            
            # Check if ToF sensor is initialized
            for sl in status_lines:
                if 'tof initialized: yes' in sl.lower():
                    self.tof_sensor_ok = True
                    break
            
            if not self.tof_sensor_ok:
                print("\n  WARNING: ToF sensor NOT initialized on ESP32!")
                print("  The ESP32 is connected but the VL53L5CX sensor")
                print("  failed to start. Check I2C wiring and restart.")
                print("  Recording will capture radar EVT events only.\n")
            
            # Send command to enable ToF streaming
            self.ser.write(b'tof stream on\n')
            time.sleep(0.1)
            # Read confirmation
            while self.ser.in_waiting:
                self.ser.readline()
            
            # Start reader thread
            self.serial_running = True
            self.serial_thread = threading.Thread(
                target=self._serial_reader_loop, daemon=True
            )
            self.serial_thread.start()
            return True
            
        except serial.SerialException as e:
            print(f"Serial error: {e}")
            print("  Available ports:")
            for p in serial.tools.list_ports.comports():
                print(f"    {p.device} - {p.description}")
            return False
    
    def _auto_detect_port(self):
        """Try to auto-detect ESP32 serial port."""
        if not SERIAL_AVAILABLE:
            return None
        for p in serial.tools.list_ports.comports():
            desc = (p.description or '').lower()
            mfg = (p.manufacturer or '').lower()
            # Common ESP32 identifiers
            if any(kw in desc for kw in ['cp210', 'ch340', 'ch910', 'esp32', 'silicon labs', 'usb-serial']):
                print(f"Auto-detected ESP32 on {p.device} ({p.description})")
                return p.device
            if any(kw in mfg for kw in ['silicon', 'wch', 'espressif']):
                print(f"Auto-detected ESP32 on {p.device} ({p.description})")
                return p.device
        return None
    
    def _serial_reader_loop(self):
        """Background thread: reads serial lines and buffers ToF data."""
        while self.serial_running:
            try:
                if self.ser and self.ser.in_waiting:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if not line:
                        continue
                    
                    self._serial_lines_received += 1
                    pc_timestamp = time.time()
                    
                    if line.startswith('TOF,'):
                        self._process_tof_line(line, pc_timestamp)
                    elif line.startswith('EVT,'):
                        self._process_evt_line(line, pc_timestamp)
                    elif line.startswith('RAW,') or line.startswith('CAL,'):
                        # Radar debug/calibration data - store if recording
                        if self.recording:
                            self._process_evt_line(line, pc_timestamp)
                    else:
                        # Debug/status line - keep for display
                        self._serial_last_line = line
                else:
                    time.sleep(0.001)  # Avoid busy-wait
                    
            except (serial.SerialException, OSError):
                print("\nSerial connection lost!")
                self.serial_connected = False
                break
            except Exception as e:
                # Don't crash the thread on parse errors
                pass
    
    def _process_tof_line(self, line, pc_timestamp):
        """Parse and store a TOF data line."""
        parts = line.split(',')
        if len(parts) < 66:  # TOF + timestamp + 64 distances
            return
        
        try:
            esp_timestamp = int(parts[1])
            distances = [int(x) for x in parts[2:66]]
        except ValueError:
            return
        
        # Update display data
        self.tof_last_frame = distances
        
        # Update FPS counter
        self._tof_fps_counter += 1
        now = time.time()
        if now - self._tof_fps_time >= 1.0:
            self.tof_fps = self._tof_fps_counter / (now - self._tof_fps_time)
            self._tof_fps_counter = 0
            self._tof_fps_time = now
        
        # If recording, write to CSV
        if self.recording:
            rec_elapsed = pc_timestamp - self.start_time
            with self.tof_lock:
                if self.tof_csv_writer:
                    row = [f"{rec_elapsed:.4f}", esp_timestamp] + distances
                    self.tof_csv_writer.writerow(row)
                    self.tof_frame_count += 1
    
    def _process_evt_line(self, line, pc_timestamp):
        """Parse and store an EVT (event) line."""
        if self.recording:
            rec_elapsed = pc_timestamp - self.start_time
            with self.tof_lock:
                self.evt_buffer.append({
                    'rec_time': round(rec_elapsed, 4),
                    'raw': line
                })
    
    def close_serial(self):
        """Close serial connection."""
        self.serial_running = False
        if self.serial_thread:
            self.serial_thread.join(timeout=2)
        if self.ser:
            try:
                self.ser.close()
            except:
                pass
            self.ser = None
        self.serial_connected = False
    
    def start_recording(self):
        """Start recording video + ToF data."""
        if self.recording:
            print("Already recording!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_path = str(self.output_dir / f"recording_{timestamp}.mp4")
        
        # Use mp4v codec for compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.recording_path,
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height)
        )
        
        if not self.video_writer.isOpened():
            print("ERROR: Could not initialize video writer!")
            self.video_writer = None
            return
        
        # Start ToF CSV recording
        if self.serial_connected:
            self.tof_path = str(self.output_dir / f"recording_{timestamp}_tof.csv")
            self.tof_csv_file = open(self.tof_path, 'w', newline='')
            self.tof_csv_writer = csv.writer(self.tof_csv_file)
            # Header row
            zone_headers = [f"z{i}" for i in range(64)]
            self.tof_csv_writer.writerow(['rec_time_s', 'esp_timestamp_ms'] + zone_headers)
            self.tof_frame_count = 0
            self.evt_buffer = []
        
        self.recording = True
        self.frame_count = 0
        self.start_time = time.time()
        
        print(f"\n** RECORDING STARTED")
        print(f"   Video: {self.recording_path}")
        if self.serial_connected:
            print(f"   ToF:   {self.tof_path}")
        print(f"   Press R to stop, Q to quit\n")
    
    def stop_recording(self):
        """Stop recording video + ToF data."""
        if not self.recording:
            print("Not recording!")
            return
        
        self.recording = False
        duration = time.time() - self.start_time
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        # Close ToF CSV
        tof_frames = self.tof_frame_count
        with self.tof_lock:
            if self.tof_csv_file:
                self.tof_csv_file.flush()
                self.tof_csv_file.close()
                self.tof_csv_file = None
                self.tof_csv_writer = None
        
        # Save events to JSON alongside
        if self.evt_buffer:
            evt_path = self.recording_path.replace('.mp4', '_events.json')
            with open(evt_path, 'w') as f:
                json.dump(self.evt_buffer, f, indent=2)
            print(f"   Events: {evt_path} ({len(self.evt_buffer)} events)")
        
        print(f"\n== RECORDING STOPPED")
        print(f"   Video: {self.recording_path}")
        print(f"   Duration: {duration:.1f}s ({self.frame_count} video frames)")
        if self.tof_path:
            print(f"   ToF:   {self.tof_path} ({tof_frames} ToF frames)")
        print(f"\n   Next step:")
        print(f"   python ground_truth_annotator.py --source {Path(self.recording_path).name}\n")
    
    def draw_overlay(self, frame):
        """Draw recording indicator, ToF status, and info."""
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Top bar
        bar_height = 50
        cv2.rectangle(overlay, (0, 0), (w, bar_height), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        if self.recording:
            # Recording indicator (blinking red dot)
            if int(time.time() * 2) % 2 == 0:
                cv2.circle(frame, (20, 25), 12, (0, 0, 255), -1)
            
            # Recording time
            duration = time.time() - self.start_time
            mins = int(duration // 60)
            secs = int(duration % 60)
            time_text = f"REC {mins:02d}:{secs:02d}"
            cv2.putText(frame, time_text, (45, 32),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Frame count
            cv2.putText(frame, f"V:{self.frame_count}", (w - 250, 32),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            if self.serial_connected:
                cv2.putText(frame, f"T:{self.tof_frame_count}", (w - 160, 32),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        else:
            # Ready to record
            cv2.putText(frame, "READY - Press R to record", (20, 32),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # ToF status indicator (top-right)
        if self.serial_connected:
            if self.tof_sensor_ok and self.tof_fps > 0:
                tof_color = (0, 200, 255)  # Orange - active
                tof_text = f"ToF {self.tof_fps:.0f}Hz"
            elif self.tof_sensor_ok:
                tof_color = (0, 255, 255)  # Yellow - waiting
                tof_text = "ToF ..."
            else:
                tof_color = (0, 0, 255)  # Red - sensor not init
                tof_text = "ToF ERR"
            cv2.putText(frame, tof_text, (w - 100, 32),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, tof_color, 1)
        else:
            cv2.putText(frame, "No ToF", (w - 70, 32),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        # Mini ToF heatmap (bottom-right corner)
        if self.tof_last_frame and self.serial_connected:
            self._draw_tof_mini(frame, w - 110, h - 145, 100, 100)
        
        # Bottom controls
        controls_text = "R=Record/Stop  |  S=Screenshot  |  Q=Quit"
        text_size = cv2.getTextSize(controls_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = (w - text_size[0]) // 2
        
        cv2.rectangle(frame, (0, h - 30), (w, h), (30, 30, 30), -1)
        cv2.putText(frame, controls_text, (text_x, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        
        return frame
    
    def _draw_tof_mini(self, frame, x, y, w, h):
        """Draw a mini 8x8 ToF heatmap on the frame."""
        distances = self.tof_last_frame
        if not distances or len(distances) < 64:
            return
        
        cell_w = w // 8
        cell_h = h // 8
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 2, y - 2), (x + w + 2, y + h + 2), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        for row in range(8):
            for col in range(8):
                idx = row * 8 + col
                dist = distances[idx]
                
                cx = x + col * cell_w
                cy = y + row * cell_h
                
                if dist < 0:
                    color = (50, 50, 50)  # Gray for invalid
                else:
                    # Map distance to color: close=red, far=blue
                    # Clamp to 0-3000mm range
                    normalized = max(0, min(1, dist / 3000))
                    # Blue (far) -> Green (mid) -> Red (close)
                    if normalized < 0.5:
                        r = int(255 * (1 - normalized * 2))
                        g = int(255 * normalized * 2)
                        b = 0
                    else:
                        r = 0
                        g = int(255 * (1 - (normalized - 0.5) * 2))
                        b = int(255 * (normalized - 0.5) * 2)
                    color = (b, g, r)  # BGR
                
                cv2.rectangle(frame, (cx, cy), (cx + cell_w - 1, cy + cell_h - 1), color, -1)
        
        # Label
        cv2.putText(frame, "ToF 8x8", (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    
    def take_screenshot(self, frame):
        """Save a screenshot."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = str(self.output_dir / f"screenshot_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot saved: {filename}")
    
    def run(self):
        """Main loop."""
        # Try to connect serial (non-fatal if fails)
        if self.serial_port != '__DISABLED__':
            self.open_serial()
        
        self.open_camera()
        
        if self.auto_start:
            self.start_recording()
        
        window_name = "Quick Recorder"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Lost camera feed, retrying...")
                    time.sleep(1)
                    continue
                
                # Record if active
                if self.recording and self.video_writer:
                    self.video_writer.write(frame)
                    self.frame_count += 1
                
                # Draw overlay
                display = self.draw_overlay(frame)
                cv2.imshow(window_name, display)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    if self.recording:
                        self.stop_recording()
                    break
                elif key == ord('r') or key == ord('R'):
                    if self.recording:
                        self.stop_recording()
                    else:
                        self.start_recording()
                elif key == ord('s') or key == ord('S'):
                    self.take_screenshot(frame)
        
        finally:
            if self.recording:
                self.stop_recording()
            self.close_serial()
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            
            print("\nRecorder closed.")


def main():
    parser = argparse.ArgumentParser(description='Quick Video + ToF Recorder')
    parser.add_argument('--source', type=str, default='0',
                       help='Camera source: index (0, 1, ...) or URL')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory for recordings')
    parser.add_argument('--auto-start', action='store_true',
                       help='Start recording immediately')
    parser.add_argument('--serial-port', type=str, default=None,
                       help='ESP32 serial port (e.g. COM5, /dev/ttyUSB0). Auto-detects if omitted.')
    parser.add_argument('--baud-rate', type=int, default=115200,
                       help='Serial baud rate (default: 115200)')
    parser.add_argument('--no-serial', action='store_true',
                       help='Disable serial/ToF recording entirely')
    
    args = parser.parse_args()
    
    recorder = QuickRecorder(
        source=args.source,
        output_dir=args.output_dir,
        auto_start=args.auto_start,
        serial_port=args.serial_port if not args.no_serial else '__DISABLED__',
        baud_rate=args.baud_rate
    )
    recorder.run()


if __name__ == "__main__":
    main()

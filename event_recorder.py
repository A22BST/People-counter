"""
Event Data Recorder - Capture All Sensor Data for Offline Analysis

Records:
- All camera detections (bbox, track_id, timestamp)
- All camera crossing events (direction, confidence, trends)
- All ESP32 events (radar, ToF)
- Timestamps for everything

Usage:
    python event_recorder.py --source 0 --fusion-port COM3 --output friday_prayer.json

After event:
    python event_replay.py --input friday_prayer.json --compare-all

Author: Ahmad's Mosque Attendance System
Version: 1.0.0
"""

import cv2
import json
import time
import threading
import argparse
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from collections import deque
import numpy as np

# Import detection components
from ultralytics import YOLO

# Try to import serial for ESP32
try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False
    print("Warning: pyserial not installed. ESP32 recording disabled.")


@dataclass
class RecordedDetection:
    """Single frame's detections"""
    timestamp: float
    frame_number: int
    detections: List[Dict]  # [{track_id, bbox, confidence}, ...]


@dataclass
class RecordedCrossing:
    """Camera crossing event"""
    timestamp: float
    track_id: int
    direction: str
    confidence: str
    trigger: str
    spawn_y: float
    crossing_y: Optional[float]
    area_change_ratio: float
    area_trend_samples: int
    y_trend_samples: int


@dataclass
class RecordedESPEvent:
    """ESP32 sensor event"""
    timestamp: float
    event_type: int
    confidence: float
    source: str
    extra: str
    raw_line: str


@dataclass
class EventRecording:
    """Complete recording of an event"""
    # Metadata
    event_name: str = ""
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0
    
    # Ground truth (add manually after recording)
    ground_truth_in: Optional[int] = None
    ground_truth_out: Optional[int] = None
    
    # Configuration used
    config: Dict = field(default_factory=dict)
    
    # Recorded data
    camera_detections: List[Dict] = field(default_factory=list)
    camera_crossings: List[Dict] = field(default_factory=list)
    esp32_events: List[Dict] = field(default_factory=list)
    
    # Summary (computed during recording)
    total_frames: int = 0
    total_detections: int = 0
    total_crossings: int = 0
    total_esp_events: int = 0


class ESP32Recorder:
    """Records all ESP32 serial events"""
    
    def __init__(self, port: str, baud: int = 115200):
        self.port = port
        self.baud = baud
        self.serial = None
        self.running = False
        self.thread = None
        self.events: List[RecordedESPEvent] = []
        self.lock = threading.Lock()
    
    def start(self):
        if not HAS_SERIAL:
            print("ESP32 recording disabled (pyserial not installed)")
            return False
        
        try:
            self.serial = serial.Serial(self.port, self.baud, timeout=0.1)
            self.running = True
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()
            print(f"ESP32 recorder started on {self.port}")
            return True
        except Exception as e:
            print(f"Failed to open ESP32 port {self.port}: {e}")
            return False
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.serial:
            self.serial.close()
    
    def _read_loop(self):
        while self.running:
            try:
                if self.serial and self.serial.in_waiting:
                    line = self.serial.readline().decode('utf-8', errors='ignore').strip()
                    if line.startswith('EVT,'):
                        event = self._parse_event(line)
                        if event:
                            with self.lock:
                                self.events.append(event)
            except Exception as e:
                pass  # Ignore read errors
    
    def _parse_event(self, line: str) -> Optional[RecordedESPEvent]:
        """Parse: EVT,<type>,<confidence>,<source>,<extra>"""
        try:
            parts = line.split(',')
            if len(parts) >= 4:
                return RecordedESPEvent(
                    timestamp=time.time(),
                    event_type=int(parts[1]),
                    confidence=float(parts[2]),
                    source=parts[3],
                    extra=parts[4] if len(parts) > 4 else "",
                    raw_line=line
                )
        except:
            pass
        return None
    
    def get_events(self) -> List[RecordedESPEvent]:
        with self.lock:
            events = self.events.copy()
        return events
    
    def send_command(self, cmd: str):
        if self.serial:
            self.serial.write(f"{cmd}\n".encode())


class EventRecorder:
    """Main recorder - captures camera and ESP32 data"""
    
    def __init__(self, source, esp_port: Optional[str] = None,
                 model_path: str = "yolov8s.pt",
                 detection_confidence: float = 0.3,
                 near_threshold: float = 0.7,
                 crossing_hysteresis: float = 0.05,
                 flip_direction: bool = False):
        
        self.source = source
        self.model_path = model_path
        self.detection_confidence = detection_confidence
        self.near_threshold = near_threshold
        self.crossing_hysteresis = crossing_hysteresis
        self.flip_direction = flip_direction
        
        # Recording storage
        self.recording = EventRecording()
        self.recording.config = {
            "source": str(source),
            "model": model_path,
            "detection_confidence": detection_confidence,
            "near_threshold": near_threshold,
            "crossing_hysteresis": crossing_hysteresis,
            "flip_direction": flip_direction
        }
        
        # ESP32 recorder
        self.esp_recorder = None
        if esp_port:
            self.esp_recorder = ESP32Recorder(esp_port)
        
        # Camera/detection components
        self.model = None
        self.cap = None
        self.frame_count = 0
        self.running = False
        
        # Track state for crossing detection
        self.tracks: Dict[int, Dict] = {}  # track_id -> state
        
        # FPS tracking
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()
    
    def start(self):
        """Start recording"""
        print("Loading YOLO model...")
        self.model = YOLO(self.model_path)
        
        # Open video source
        source = self.source
        if source.isdigit():
            source = int(source)
        
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")
        
        # Get frame dimensions
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video resolution: {self.frame_width}x{self.frame_height}")
        
        # Start ESP32 recorder
        if self.esp_recorder:
            self.esp_recorder.start()
            # Enable raw output for maximum data capture
            time.sleep(0.5)
            self.esp_recorder.send_command("raw on")
        
        self.recording.start_time = datetime.now().isoformat()
        self.running = True
        print("\n" + "="*50)
        print("  RECORDING STARTED")
        if self.flip_direction:
            print("  Direction: FLIPPED (IN↔OUT swapped)")
        print("  Controls:")
        print("    s = Show stats")
        print("    q = Stop and save")
        print("="*50 + "\n")
    
    def stop(self):
        """Stop recording and compile data"""
        self.running = False
        self.recording.end_time = datetime.now().isoformat()
        
        # Calculate duration
        start = datetime.fromisoformat(self.recording.start_time)
        end = datetime.fromisoformat(self.recording.end_time)
        self.recording.duration_seconds = (end - start).total_seconds()
        
        # Get ESP32 events
        if self.esp_recorder:
            esp_events = self.esp_recorder.get_events()
            self.recording.esp32_events = [asdict(e) for e in esp_events]
            self.recording.total_esp_events = len(esp_events)
            self.esp_recorder.stop()
        
        # Compile stats
        self.recording.total_frames = self.frame_count
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nRecording stopped after {self.recording.duration_seconds:.1f} seconds")
    
    def process_frame(self) -> Optional[np.ndarray]:
        """Process one frame and record data"""
        if not self.cap or not self.running:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        self.frame_count += 1
        current_time = time.time()
        
        # FPS calculation
        frame_time = current_time - self.last_frame_time
        self.fps_history.append(1.0 / max(frame_time, 0.001))
        self.last_frame_time = current_time
        
        # Run YOLO with tracking
        results = self.model.track(
            frame,
            persist=True,
            conf=self.detection_confidence,
            classes=[0],  # person only
            verbose=False
        )
        
        # Extract detections
        detections = []
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                track_id = int(boxes.id[i])
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i])
                
                detections.append({
                    "track_id": track_id,
                    "bbox": bbox,
                    "confidence": conf
                })
                
                # Check for crossing
                self._check_crossing(track_id, bbox, current_time)
        
        # Record detections
        if detections:
            self.recording.camera_detections.append({
                "timestamp": current_time,
                "frame_number": self.frame_count,
                "detections": detections
            })
            self.recording.total_detections += len(detections)
        
        # Draw visualization
        frame = self._draw_frame(frame, detections)
        
        return frame
    
    def _check_crossing(self, track_id: int, bbox: List[float], timestamp: float):
        """Check if track crossed threshold and record"""
        x1, y1, x2, y2 = bbox
        centroid_y = (y1 + y2) / 2 / self.frame_height
        area = (x2 - x1) * (y2 - y1)
        
        # Initialize track if new
        if track_id not in self.tracks:
            # Determine initial state
            if centroid_y < self.near_threshold - self.crossing_hysteresis:
                state = "above"
            elif centroid_y > self.near_threshold + self.crossing_hysteresis:
                state = "below"
            else:
                state = None
            
            self.tracks[track_id] = {
                "state": state,
                "spawn_y": centroid_y,
                "spawn_area": area,
                "crossed": False,
                "crossed_direction": None,
                "y_history": [centroid_y],
                "area_history": [area]
            }
            return
        
        track = self.tracks[track_id]
        track["y_history"].append(centroid_y)
        track["area_history"].append(area)
        
        # Keep history bounded
        if len(track["y_history"]) > 30:
            track["y_history"] = track["y_history"][-30:]
            track["area_history"] = track["area_history"][-30:]
        
        # Determine current state
        if centroid_y < self.near_threshold - self.crossing_hysteresis:
            current_state = "above"
        elif centroid_y > self.near_threshold + self.crossing_hysteresis:
            current_state = "below"
        else:
            current_state = None
        
        # Check for crossing
        if track["state"] and current_state and track["state"] != current_state:
            # Prevent same-direction double counting
            if track["state"] == "above" and current_state == "below":
                direction = "IN"
            else:
                direction = "OUT"
            
            # Apply flip if enabled
            if self.flip_direction:
                direction = "OUT" if direction == "IN" else "IN"
            
            if track["crossed_direction"] != direction:
                # Calculate trends
                y_hist = track["y_history"]
                area_hist = track["area_history"]
                
                y_changes = np.diff(y_hist) if len(y_hist) > 1 else [0]
                area_changes = np.diff(area_hist) if len(area_hist) > 1 else [0]
                
                y_trend = int(np.sum(y_changes > 0.01) - np.sum(y_changes < -0.01))
                area_trend = int(np.sum(area_changes > 0) - np.sum(area_changes < 0))
                
                # Determine confidence
                if direction == "IN":
                    signals = (area_trend > 0) + (y_trend > 0) + (track["spawn_y"] < 0.5)
                else:
                    signals = (area_trend < 0) + (y_trend < 0) + (track["spawn_y"] > 0.5)
                
                confidence = "HIGH" if signals >= 2 else "MEDIUM" if signals >= 1 else "LOW"
                
                # Record crossing
                crossing = RecordedCrossing(
                    timestamp=timestamp,
                    track_id=track_id,
                    direction=direction,
                    confidence=confidence,
                    trigger="threshold_crossing",
                    spawn_y=track["spawn_y"],
                    crossing_y=centroid_y,
                    area_change_ratio=area / track["spawn_area"] if track["spawn_area"] > 0 else 1.0,
                    area_trend_samples=area_trend,
                    y_trend_samples=y_trend
                )
                
                self.recording.camera_crossings.append(asdict(crossing))
                self.recording.total_crossings += 1
                
                track["crossed"] = True
                track["crossed_direction"] = direction
                
                print(f"  → CROSSING: {direction} (track {track_id}, conf={confidence})")
        
        # Update state
        if current_state:
            track["state"] = current_state
    
    def _draw_frame(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw visualization"""
        # Draw threshold line
        threshold_y = int(self.near_threshold * self.frame_height)
        cv2.line(frame, (0, threshold_y), (self.frame_width, threshold_y), (0, 255, 255), 2)
        cv2.putText(frame, "THRESHOLD", (10, threshold_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw hysteresis zone
        upper = int((self.near_threshold - self.crossing_hysteresis) * self.frame_height)
        lower = int((self.near_threshold + self.crossing_hysteresis) * self.frame_height)
        cv2.rectangle(frame, (0, upper), (self.frame_width, lower), (0, 255, 255), 1)
        
        # Draw detections
        for det in detections:
            bbox = det["bbox"]
            track_id = det["track_id"]
            x1, y1, x2, y2 = map(int, bbox)
            
            # Color based on position relative to threshold
            cy = (y1 + y2) / 2 / self.frame_height
            if cy < self.near_threshold:
                color = (0, 255, 0)  # Green = above
            else:
                color = (0, 0, 255)  # Red = below
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw stats
        fps = np.mean(self.fps_history) if self.fps_history else 0
        in_count = sum(1 for c in self.recording.camera_crossings if c["direction"] == "IN")
        out_count = sum(1 for c in self.recording.camera_crossings if c["direction"] == "OUT")
        
        stats = [
            f"FPS: {fps:.1f}  Frame: {self.frame_count}",
            f"IN: {in_count}   OUT: {out_count}",
            "[s]=stats [q]=quit"
        ]
        
        for i, stat in enumerate(stats):
            cv2.putText(frame, stat, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Recording indicator (blinking red dot)
        if self.frame_count % 30 < 15:
            cv2.circle(frame, (self.frame_width - 30, 30), 10, (0, 0, 255), -1)
        
        return frame
    
    def save(self, output_path: str):
        """Save recording to JSON"""
        with open(output_path, 'w') as f:
            json.dump(asdict(self.recording), f, indent=2)
        
        # Calculate system counts
        sys_in = sum(1 for c in self.recording.camera_crossings if c["direction"] == "IN")
        sys_out = sum(1 for c in self.recording.camera_crossings if c["direction"] == "OUT")
        
        print(f"\n{'='*50}")
        print(f"  RECORDING SAVED: {output_path}")
        print(f"{'='*50}")
        print(f"  Duration: {self.recording.duration_seconds:.1f}s")
        print(f"  Frames: {self.recording.total_frames}")
        print(f"  System count:  IN={sys_in}, OUT={sys_out}")
        print(f"  ESP events: {self.recording.total_esp_events}")
        print(f"{'='*50}")
        print(f"\nNext steps:")
        print(f"  1. Edit {output_path} and add:")
        print(f'     "ground_truth_in": <your count>,')
        print(f'     "ground_truth_out": <your count>,')
        print(f"  2. Run: python event_replay.py --input {output_path}")
    
    def run(self):
        """Main recording loop"""
        self.start()
        
        try:
            while self.running:
                frame = self.process_frame()
                
                if frame is None:
                    break
                
                cv2.imshow('Event Recorder', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    in_count = sum(1 for c in self.recording.camera_crossings if c["direction"] == "IN")
                    out_count = sum(1 for c in self.recording.camera_crossings if c["direction"] == "OUT")
                    print(f"\n=== Stats ===")
                    print(f"Frames: {self.frame_count}")
                    print(f"System:  IN={in_count}, OUT={out_count}")
                    print(f"ESP Events: {len(self.esp_recorder.get_events()) if self.esp_recorder else 0}")
                    print(f"=============\n")
        
        finally:
            self.stop()


def main():
    parser = argparse.ArgumentParser(description='Record event data for offline analysis')
    parser.add_argument('--source', type=str, default='0',
                        help='Video source (0 for webcam, URL for IP cam)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path (default: auto-generated timestamped filename)')
    parser.add_argument('--fusion-port', type=str, default=None,
                        help='ESP32 serial port (optional)')
    parser.add_argument('--model', type=str, default='yolov8s.pt',
                        help='YOLO model path')
    parser.add_argument('--confidence', type=float, default=0.3,
                        help='Detection confidence threshold')
    parser.add_argument('--near-threshold', type=float, default=0.7,
                        help='Near-edge threshold (0-1)')
    parser.add_argument('--flip', action='store_true',
                        help='Flip IN/OUT direction')
    parser.add_argument('--event-name', type=str, default='',
                        help='Name of the event being recorded')
    
    args = parser.parse_args()

    output_path = args.output or f"event_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    if args.output is None:
        print(f"No --output provided; using generated file: {output_path}")
    
    recorder = EventRecorder(
        source=args.source,
        esp_port=args.fusion_port,
        model_path=args.model,
        detection_confidence=args.confidence,
        near_threshold=args.near_threshold,
        flip_direction=args.flip
    )
    
    if args.event_name:
        recorder.recording.event_name = args.event_name
    
    recorder.run()
    recorder.save(output_path)


if __name__ == '__main__':
    main()

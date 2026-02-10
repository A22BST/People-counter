"""
People Counter: Full Detection Pipeline (v2.0 - Sensor Fusion v2)

Integrates:
- YOLOv8 for person detection
- ByteTrack for multi-object tracking
- DirectionDetector for entry/exit classification
- Standby Slot Method for unique visitor counting
- Sensor Fusion v2 with ToF dual-zone direction detection
- Live diagnostics and analysis

Supports:
- DroidCam/IP camera input
- USB/built-in camera
- Video file input (for testing)

Changes from v1.1:
- sensor_fusion_v2 replaces sensor_fusion (ToF-focused, no radar)
- FusionConfig object replaces positional args
- New strategies: confirmation, dual_zone, hybrid (replaces tiebreaker, crossing)
- Simplified CameraEvent (trigger field instead of area_trend/y_trend)
- create_camera_event() replaces create_camera_event_from_detector_event()
- ToF-only event detection (camera missed, ToF caught)
- FusedDecision now has track_id and tof_confidence

Author: Ahmad's Mosque Attendance System
Version: 2.0.0
"""

# ============================================================================
#                        QUICK CONFIGURATION
#                   (Edit these values after calibration)
# ============================================================================

# Camera source - change this to your camera
CAMERA_SOURCE = "0"  # "0" for webcam, or "http://192.168.1.100:4747/video" for DroidCam

# Detection thresholds - adjust if missing people or too many false positives
DETECTION_CONFIDENCE = 0.3      # Lower = detect more (but more false positives)

# Direction detection thresholds - UPDATE THESE FROM DIAGNOSTIC REPORT
NEAR_EDGE_THRESHOLD = 0.584      # Where the door threshold line is (0-1, bottom of frame)
SPAWN_FAR_THRESHOLD = 0.382       # Where entries appear (top of frame, 0-1)
AREA_GROWTH_FOR_APPROACH = 0.68  # Area ratio for "approaching" (>1 = growing)
SPAWN_NEAR_THRESHOLD = 0.57      # Where exits appear (bottom of frame, 0-1)
AREA_SHRINK_FOR_DEPART = 0.372    # Area ratio for "departing" (<1 = shrinking)
MIN_TRACK_DURATION_MS = 10     # Minimum track lifetime to count (milliseconds)

# ===== Threshold crossing settings =====
COUNT_ON_CROSSING = True         # True = count when crossing threshold (recommended)
                                 # False = count when track terminates (legacy)
CROSSING_HYSTERESIS = 0.03       # Buffer zone to prevent double-counting (3% of frame)

# Standby slot - for handling re-entry
STANDBY_TIMEOUT_SECONDS = 0  # 0 = infinite (never expires), >0 = timeout in seconds

# Direction flip - swap IN/OUT if camera/sensors mounted opposite way
FLIP_DIRECTION = False  # True = IN becomes OUT, OUT becomes IN

# ============================================================================
#                        END OF QUICK CONFIG
# ============================================================================

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Deque
import time
import json
import logging
from datetime import datetime
from pathlib import Path

from direction_detector import DirectionDetector, DirectionDetectorConfig, CrossingConfidence

# Try to import sensor fusion v2 (optional - for ESP32 ToF integration)
try:
    from sensor_fusion_v2 import (
        SensorFusionEngine,
        FusionConfig,
        CameraEvent,
        FusedDecision,
        create_camera_event,
    )
    HAS_FUSION = True
except ImportError:
    HAS_FUSION = False

# Try to import matplotlib for plots (optional)
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrackAnalyzer:
    """
    Analyzes track behavior patterns in real-time.
    Collects data during operation and generates diagnostic report on exit.
    """
    
    def __init__(self):
        self.tracks = {}  # track_id -> track data
        self.completed_tracks = []
        self.frame_count = 0
        self.detection_counts = []
        self.start_time = time.time()
    
    def record_frame(self, detections, frame_number, frame_height):
        """Record detections for this frame"""
        self.frame_count = frame_number
        self.detection_counts.append(len(detections))
        
        for track_id, bbox in detections:
            x1, y1, x2, y2 = bbox
            center_y = (y1 + y2) / 2.0 / frame_height
            area = (x2 - x1) * (y2 - y1)
            
            if track_id not in self.tracks:
                self.tracks[track_id] = {
                    'track_id': track_id,
                    'spawn_y': center_y,
                    'spawn_area': area,
                    'spawn_frame': frame_number,
                    'y_history': [],
                    'area_history': [],
                    'frame_count': 0
                }
            
            t = self.tracks[track_id]
            t['y_history'].append(center_y)
            t['area_history'].append(area)
            t['frame_count'] += 1
            t['last_y'] = center_y
            t['last_area'] = area
            t['last_frame'] = frame_number
    
    def finalize_track(self, track_id, direction=None, confidence=None, event=None):
        """Mark a track as completed"""
        if track_id in self.tracks:
            t = self.tracks[track_id]
            t['direction'] = direction
            t['confidence'] = confidence
            if event:
                t['trigger'] = event.get('trigger', 'unknown')
                t['duration_ms'] = event.get('duration_ms', 0)
            self.completed_tracks.append(t)
    
    def generate_report(self) -> dict:
        """Generate diagnostic report"""
        elapsed = time.time() - self.start_time
        
        # Separate by result
        entries = [t for t in self.completed_tracks if t.get('direction') == 'IN']
        exits = [t for t in self.completed_tracks if t.get('direction') == 'OUT']
        abandoned = [t for t in self.completed_tracks if t.get('direction') is None]
        
        report = {
            'runtime_seconds': elapsed,
            'total_frames': self.frame_count,
            'fps': self.frame_count / elapsed if elapsed > 0 else 0,
            'total_tracks': len(self.completed_tracks),
            'entries': len(entries),
            'exits': len(exits),
            'abandoned': len(abandoned),
            'avg_detection_count': np.mean(self.detection_counts) if self.detection_counts else 0,
            'max_detection_count': max(self.detection_counts) if self.detection_counts else 0,
        }
        
        # Analyze entry patterns
        if entries:
            report['entry_spawn_y_mean'] = np.mean([t['spawn_y'] for t in entries])
            report['entry_spawn_y_std'] = np.std([t['spawn_y'] for t in entries])
            report['entry_final_y_mean'] = np.mean([t.get('last_y', 0.5) for t in entries])
        
        if exits:
            report['exit_spawn_y_mean'] = np.mean([t['spawn_y'] for t in exits])
            report['exit_spawn_y_std'] = np.std([t['spawn_y'] for t in exits])
            report['exit_final_y_mean'] = np.mean([t.get('last_y', 0.5) for t in exits])
        
        # Recommend thresholds
        if entries and exits:
            all_entry_spawn = [t['spawn_y'] for t in entries]
            all_exit_spawn = [t['spawn_y'] for t in exits]
            all_final_y = [t.get('last_y', 0.5) for t in entries + exits]
            
            report['recommended'] = {
                'near_edge_threshold': float(np.percentile(all_final_y, 85)),
                'spawn_far_threshold': float(np.percentile(all_entry_spawn, 75)),
                'spawn_near_threshold': float(np.percentile(all_exit_spawn, 25)),
            }
        
        return report
    
    def print_report(self, report: dict):
        """Print formatted report"""
        print("\n" + "=" * 60)
        print("  DIAGNOSTIC REPORT")
        print("=" * 60)
        print(f"\n  Runtime: {report['runtime_seconds']:.1f}s")
        print(f"  Frames: {report['total_frames']} ({report['fps']:.1f} FPS)")
        print(f"  Total tracks: {report['total_tracks']}")
        print(f"  Entries: {report['entries']}")
        print(f"  Exits: {report['exits']}")
        print(f"  Abandoned: {report['abandoned']}")
        print(f"  Avg detections/frame: {report['avg_detection_count']:.1f}")
        print(f"  Max detections/frame: {report['max_detection_count']}")
        
        if 'entry_spawn_y_mean' in report:
            print(f"\n  Entry spawn Y: {report['entry_spawn_y_mean']:.3f} +/- {report['entry_spawn_y_std']:.3f}")
            print(f"  Entry final Y: {report['entry_final_y_mean']:.3f}")
        
        if 'exit_spawn_y_mean' in report:
            print(f"  Exit spawn Y: {report['exit_spawn_y_mean']:.3f} +/- {report['exit_spawn_y_std']:.3f}")
            print(f"  Exit final Y: {report['exit_final_y_mean']:.3f}")
        
        if 'recommended' in report:
            rec = report['recommended']
            print(f"\n  RECOMMENDED THRESHOLDS:")
            print(f"    near_edge_threshold: {rec['near_edge_threshold']:.3f}")
            print(f"    spawn_far_threshold: {rec['spawn_far_threshold']:.3f}")
            print(f"    spawn_near_threshold: {rec['spawn_near_threshold']:.3f}")
        
        print("=" * 60 + "\n")
    
    def generate_plots(self, output_path: str = "diagnostics.png") -> bool:
        """Generate diagnostic plots"""
        if not HAS_MATPLOTLIB:
            print("matplotlib not available, skipping plots")
            return False
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('People Counter Diagnostics', fontsize=14)
        
        entries = [t for t in self.completed_tracks if t.get('direction') == 'IN']
        exits = [t for t in self.completed_tracks if t.get('direction') == 'OUT']
        abandoned = [t for t in self.completed_tracks if t.get('direction') is None]
        
        # Plot 1: Spawn Y distribution
        ax = axes[0, 0]
        if entries:
            ax.hist([t['spawn_y'] for t in entries], bins=20, alpha=0.6, label='Entry', color='green')
        if exits:
            ax.hist([t['spawn_y'] for t in exits], bins=20, alpha=0.6, label='Exit', color='red')
        ax.set_xlabel('Spawn Y (normalized)')
        ax.set_title('Spawn Y Distribution')
        ax.legend()
        
        # Plot 2: Final Y distribution
        ax = axes[0, 1]
        if entries:
            ax.hist([t.get('last_y', 0.5) for t in entries], bins=20, alpha=0.6, label='Entry', color='green')
        if exits:
            ax.hist([t.get('last_y', 0.5) for t in exits], bins=20, alpha=0.6, label='Exit', color='red')
        ax.set_xlabel('Final Y (normalized)')
        ax.set_title('Final Y Distribution')
        ax.legend()
        
        # Plot 3: Track duration
        ax = axes[0, 2]
        durations = [t.get('duration_ms', t['frame_count'] * 33) for t in self.completed_tracks if t.get('direction')]
        if durations:
            ax.hist(durations, bins=30, alpha=0.6, color='blue')
        ax.set_xlabel('Duration (ms)')
        ax.set_title('Track Duration')
        
        # Plot 4: Area change
        ax = axes[1, 0]
        for t in entries[:20]:
            if t['area_history']:
                areas = np.array(t['area_history']) / t['spawn_area']
                ax.plot(areas, alpha=0.4, color='green')
        for t in exits[:20]:
            if t['area_history']:
                areas = np.array(t['area_history']) / t['spawn_area']
                ax.plot(areas, alpha=0.4, color='red')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Area Ratio')
        ax.set_title('Area Change (green=IN, red=OUT)')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        
        # Plot 5: Detections over time
        ax = axes[1, 1]
        if self.detection_counts:
            ax.fill_between(range(len(self.detection_counts)), self.detection_counts, alpha=0.3)
            ax.set_xlabel('Frame')
            ax.set_ylabel('People Detected')
            ax.set_title('Detection Count Over Time')
            ax.grid(True, alpha=0.3)
        
        # Plot 6: Confidence pie chart
        ax = axes[1, 2]
        conf_counts = {
            'HIGH': len([t for t in self.completed_tracks if t.get('confidence') == 'HIGH']),
            'MEDIUM': len([t for t in self.completed_tracks if t.get('confidence') == 'MEDIUM']),
            'LOW': len([t for t in self.completed_tracks if t.get('confidence') == 'LOW']),
            'ABANDONED': len(abandoned)
        }
        non_zero = {k: v for k, v in conf_counts.items() if v > 0}
        if non_zero:
            colors_pie = {'HIGH': 'green', 'MEDIUM': 'orange', 'LOW': 'red', 'ABANDONED': 'gray'}
            ax.pie(non_zero.values(), labels=non_zero.keys(), autopct='%1.0f%%',
                   colors=[colors_pie[k] for k in non_zero.keys()])
            ax.set_title('Detection Confidence')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Plots saved to {output_path}")
        return True


@dataclass
class StandbySlot:
    """Represents a standby slot for someone who exited"""
    created_time: float
    timeout_seconds: float = 0  # 0 = infinite (never expires)
    
    def is_expired(self, current_time: float) -> bool:
        if self.timeout_seconds <= 0:
            return False  # Never expires
        return (current_time - self.created_time) > self.timeout_seconds


@dataclass
class PeopleCounterConfig:
    """Master configuration for the people counter system"""
    
    # Input source - uses QUICK CONFIG value
    camera_source: str = CAMERA_SOURCE
    
    # YOLOv8 settings
    model_path: str = "yolov8s.pt"
    detection_confidence: float = DETECTION_CONFIDENCE
    nms_iou_threshold: float = 0.7
    person_class_id: int = 0
    
    # Frame processing
    frame_width: int = 640
    frame_height: int = 480
    process_every_n_frames: int = 1
    
    # ByteTrack settings
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    track_buffer: int = 30
    match_thresh: float = 0.8
    
    # Direction detector config (passed through)
    direction_config: DirectionDetectorConfig = field(default_factory=lambda: DirectionDetectorConfig(
        near_edge_threshold=NEAR_EDGE_THRESHOLD,
        spawn_far_threshold=SPAWN_FAR_THRESHOLD,
        spawn_near_threshold=SPAWN_NEAR_THRESHOLD,
        area_growth_for_approach=AREA_GROWTH_FOR_APPROACH,
        area_shrink_for_depart=AREA_SHRINK_FOR_DEPART,
        min_track_duration_ms=MIN_TRACK_DURATION_MS,
        count_on_crossing=COUNT_ON_CROSSING,
        crossing_hysteresis=CROSSING_HYSTERESIS
    ))
    
    # Standby slot settings
    standby_timeout_seconds: float = STANDBY_TIMEOUT_SECONDS
    
    # Visualization
    show_visualization: bool = True
    show_debug_overlay: bool = True
    visualization_scale: float = 1.0
    
    # Output
    log_events: bool = True
    output_file: Optional[str] = None
    
    # Diagnostics
    enable_diagnostics: bool = True
    diagnostics_output_dir: str = "."
    generate_plots: bool = True
    
    # Sensor Fusion v2 (ESP32 ToF integration)
    enable_fusion: bool = False
    fusion_port: str = "COM3"
    fusion_strategy: str = "dual_zone"   # 'confirmation', 'dual_zone', or 'hybrid'
    fusion_confidence_threshold: float = 0.5
    
    # Direction flip - swap IN/OUT
    flip_direction: bool = FLIP_DIRECTION
    
    # Calibration mode
    calibrate_mode: bool = False


class ByteTrackWrapper:
    """
    Wrapper for ByteTrack integration with YOLOv8.
    Uses the built-in tracker from ultralytics.
    """
    
    def __init__(self, config: PeopleCounterConfig):
        self.config = config
        self.model = YOLO(config.model_path)
    
    def track_frame(self, frame: np.ndarray) -> List[Tuple[int, Tuple[float, float, float, float]]]:
        """Track people in frame, returns list of (track_id, bbox)"""
        results = self.model.track(
            frame,
            persist=True,
            classes=[self.config.person_class_id],
            conf=self.config.detection_confidence,
            iou=self.config.nms_iou_threshold,
            tracker="bytetrack.yaml",
            verbose=False
        )
        
        detections = []
        if results and len(results) > 0 and results[0].boxes is not None \
           and results[0].boxes.id is not None:
            boxes = results[0].boxes
            for i, (box, track_id) in enumerate(zip(boxes.xyxy, boxes.id)):
                x1, y1, x2, y2 = box.cpu().numpy()
                tid = int(track_id.cpu().numpy())
                detections.append((tid, (float(x1), float(y1), float(x2), float(y2))))
        
        return detections


class PeopleCounter:
    """
    Main people counter integrating all components.
    """
    
    def __init__(self, config: PeopleCounterConfig = None):
        self.config = config or PeopleCounterConfig()
        
        # Core components
        self.tracker = ByteTrackWrapper(self.config)
        self.direction_detector = DirectionDetector(self.config.direction_config)
        
        # Standby Slot Method state
        self.unique_visitors = 0
        self.standby_slots: Deque[StandbySlot] = deque()
        
        # Frame processing state
        self.cap = None
        self.frame_count = 0
        self.events: List[dict] = []
        self.active_track_ids: set = set()
        self.running = False
        
        # Diagnostics
        self.analyzer = TrackAnalyzer() if self.config.enable_diagnostics else None
        
        # Sensor fusion v2 engine (optional - for ESP32 ToF integration)
        self.fusion_engine: Optional['SensorFusionEngine'] = None
        if HAS_FUSION and self.config.enable_fusion:
            fusion_config = FusionConfig(
                port=self.config.fusion_port,
                strategy=self.config.fusion_strategy,
            )
            self.fusion_engine = SensorFusionEngine(fusion_config)
            logger.info(f"Sensor fusion v2 initialized: {self.config.fusion_strategy} "
                       f"on {self.config.fusion_port}")
    
    def start(self):
        """Initialize video capture and start processing"""
        source = self.config.camera_source
        
        if source.isdigit():
            source = int(source)
        
        logger.info(f"Opening video source: {source}")
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Video resolution: {actual_width}x{actual_height}")
        
        self.config.frame_width = actual_width
        self.config.frame_height = actual_height
        self.config.direction_config.frame_width = actual_width
        self.config.direction_config.frame_height = actual_height
        
        self.running = True
        logger.info("People counter started")
        
        # Start sensor fusion if enabled
        if self.fusion_engine:
            if self.fusion_engine.start():
                logger.info("Sensor fusion v2 engine started")
                
                # Apply initial flip state
                if self.config.flip_direction:
                    self.fusion_engine.set_flip(True)
                
                # Start calibration mode if configured
                if self.config.calibrate_mode:
                    self.fusion_engine.start_calibration()
                    logger.info("Calibration mode started")
            else:
                logger.warning("Failed to start sensor fusion - running camera-only")
                self.fusion_engine = None
    
    def stop(self):
        """Stop processing and release resources"""
        self.running = False
        
        if self.fusion_engine:
            self.fusion_engine.stop()
            logger.info("Sensor fusion v2 engine stopped")
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Generate diagnostic report
        if self.analyzer:
            print("\n  Generating diagnostic report...")
            report = self.analyzer.generate_report()
            self.analyzer.print_report(report)
            
            output_dir = Path(self.config.diagnostics_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            report_path = output_dir / f"diagnostics_{timestamp}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"  Report saved to {report_path}")
            
            if self.config.generate_plots:
                plot_path = output_dir / f"diagnostics_{timestamp}.png"
                self.analyzer.generate_plots(str(plot_path))
        
        # Save events log
        if self.config.output_file and self.events:
            entries, exits, occupancy = self.direction_detector.get_counts()
            output = {
                'events': self.events,
                'summary': {
                    'total_entries': entries,
                    'total_exits': exits,
                    'unique_visitors': self.unique_visitors,
                    'current_occupancy': occupancy
                }
            }
            with open(self.config.output_file, 'w') as f:
                json.dump(output, f, indent=2, default=str)
            print(f"  Events saved to {self.config.output_file}")
        
        logger.info("People counter stopped")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """Process a single frame"""
        self.frame_count += 1
        current_time = time.time()
        
        # Run tracker
        detections = self.tracker.track_frame(frame)
        current_track_ids = set(tid for tid, _ in detections)
        
        # Record frame data for diagnostics
        if self.analyzer:
            self.analyzer.record_frame(detections, self.frame_count, self.config.frame_height)
        
        # Feed to direction detector
        crossing_events = self.direction_detector.update(detections)
        
        # Process crossing events with Standby Slot Method
        for event in crossing_events:
            self._handle_crossing_event(event, current_time)
            
            if self.analyzer:
                self.analyzer.finalize_track(
                    event['track_id'],
                    event['direction'],
                    event['confidence'],
                    event
                )
            
            if self.config.log_events:
                event['timestamp'] = current_time
                event['frame'] = self.frame_count
                self.events.append(event)
                logger.info(f"Crossing: {event['direction']} (confidence: {event['confidence']})")
        
        # Check for ToF-only events (camera missed, ToF detected)
        if self.fusion_engine and not crossing_events:
            self.fusion_engine.check_tof_only_events()
        
        # Track terminated tracks (for analyzer)
        if self.analyzer:
            terminated = self.active_track_ids - current_track_ids
            for tid in terminated:
                if tid not in [e['track_id'] for e in crossing_events]:
                    self.analyzer.finalize_track(tid, None, None)
        
        self.active_track_ids = current_track_ids
        
        # Clean expired standby slots
        self._clean_expired_slots(current_time)
        
        # Draw visualization if enabled
        if self.config.show_visualization:
            frame = self._draw_visualization(frame, detections, crossing_events)
        
        return frame, crossing_events
    
    def _handle_crossing_event(self, event: dict, current_time: float):
        """
        Apply Standby Slot Method to handle crossing events.
        Uses sensor fusion v2 if enabled for improved accuracy.
        
        Rules:
        - EXIT at occupancy > 0: Create a standby slot
        - EXIT at occupancy = 0: Person wasn't counted! Add to unique + create standby
        - ENTRY with standby slot: Consume slot (returning person)
        - ENTRY without standby: New unique visitor
        """
        direction = event['direction']
        trigger = event.get('trigger', 'unknown')
        
        logger.info(f"Crossing: {direction} (conf={event.get('confidence', '?')}, "
                   f"trigger={trigger}, track={event['track_id']})")
        
        # Apply direction flip if enabled
        if self.config.flip_direction:
            direction = 'OUT' if direction == 'IN' else 'IN'
        
        # Use sensor fusion v2 if enabled
        if self.fusion_engine:
            # Create camera event using v2 helper (simpler - no track_history needed)
            camera_event = create_camera_event(event)
            
            # Apply flip to camera event too
            if self.config.flip_direction:
                camera_event.direction = 'OUT' if camera_event.direction == 'IN' else 'IN'
            
            # Get fused decision
            decision = self.fusion_engine.process_camera_event(camera_event)
            
            if decision is None:
                logger.debug(f"Track {event['track_id']}: Fusion returned None (waiting/rejected)")
                return
            
            if decision.confidence < self.config.fusion_confidence_threshold:
                logger.warning(f"Track {event['track_id']}: Low fusion confidence "
                              f"({decision.confidence:.2f}), skipping")
                return
            
            # Use fused direction and log decision source
            direction = decision.direction
            logger.info(f"Fusion decision: {direction} (conf={decision.confidence:.2f}, "
                       f"src={decision.source}, tof_conf={decision.tof_confidence:.2f})")
        
        # Get current occupancy for negative occupancy check
        _, _, occupancy = self.direction_detector.get_counts()
        
        # Apply Standby Slot Method with negative occupancy fix
        if direction == 'OUT':
            if occupancy <= 0:
                # Person was inside but never counted on entry
                self.unique_visitors += 1
                logger.info(f"Uncounted exit detected! Added unique visitor #{self.unique_visitors}")
            
            slot = StandbySlot(created_time=current_time,
                              timeout_seconds=self.config.standby_timeout_seconds)
            self.standby_slots.append(slot)
            logger.debug(f"Created standby slot (total: {len(self.standby_slots)})")
            
        elif direction == 'IN':
            if self.standby_slots:
                self.standby_slots.popleft()
                logger.debug(f"Consumed standby slot (remaining: {len(self.standby_slots)})")
            else:
                self.unique_visitors += 1
                logger.info(f"New unique visitor #{self.unique_visitors}")
    
    def _clean_expired_slots(self, current_time: float):
        """Remove expired standby slots"""
        while self.standby_slots and self.standby_slots[0].is_expired(current_time):
            self.standby_slots.popleft()
    
    def _draw_visualization(self, frame: np.ndarray,
                           detections: List[Tuple[int, Tuple[float, float, float, float]]],
                           events: List[dict]) -> np.ndarray:
        """Draw debug visualization on frame"""
        
        # Draw detection zone indicator (near-edge threshold)
        near_edge_y = int(self.config.direction_config.near_edge_threshold * self.config.frame_height)
        cv2.line(frame, (0, near_edge_y), (self.config.frame_width, near_edge_y),
                 (0, 255, 255), 2)
        cv2.putText(frame, "DOOR THRESHOLD", (10, near_edge_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw bounding boxes with track IDs
        for track_id, (x1, y1, x2, y2) in detections:
            color = (0, 255, 0)
            
            # Check track state for color
            state = self.direction_detector.get_track_state(track_id)
            if state is not None:
                from direction_detector import TrackState
                if state == TrackState.APPROACHING:
                    color = (0, 200, 0)
                elif state == TrackState.DEPARTING:
                    color = (0, 0, 200)
                elif state in (TrackState.CROSSED_IN, TrackState.CROSSED_OUT):
                    color = (200, 200, 0)
            
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"ID:{track_id}"
            cv2.putText(frame, label, (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw events
        for event in events:
            direction = event['direction']
            confidence = event.get('confidence', '?')
            trigger = event.get('trigger', 'unknown')
            
            if direction == 'IN':
                color = (0, 255, 0)
                text = f">>> IN ({confidence})"
            else:
                color = (0, 0, 255)
                text = f"<<< OUT ({confidence})"
            
            cv2.putText(frame, text, (10, 30 + len(events) * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw stats overlay
        entries, exits, occupancy = self.direction_detector.get_counts()
        stats = [
            f"FPS: {self.frame_count / max(1, time.time() - self.analyzer.start_time):.1f}" if self.analyzer else "",
            f"Entries: {entries}",
            f"Exits: {exits}",
            f"Occupancy: {occupancy}",
            f"Unique: {self.unique_visitors}",
            f"Standby: {len(self.standby_slots)}",
        ]
        
        # Fusion stats
        if self.fusion_engine:
            fusion_stats = self.fusion_engine.get_stats()
            stats.append(f"Fusion: {fusion_stats.get('total', 0)} decisions")
            stats.append(f"Avg conf: {fusion_stats.get('avg_confidence', 0):.2f}")
        
        y_offset = self.config.frame_height - 20 * len(stats) - 10
        for i, stat in enumerate(stats):
            if stat:
                cv2.putText(frame, stat, (10, y_offset + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Main processing loop"""
        self.start()
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    break
                
                # Process frame
                frame, events = self.process_frame(frame)
                
                if self.config.show_visualization:
                    if self.config.visualization_scale != 1.0:
                        new_size = (
                            int(frame.shape[1] * self.config.visualization_scale),
                            int(frame.shape[0] * self.config.visualization_scale)
                        )
                        frame = cv2.resize(frame, new_size)
                    
                    cv2.imshow('People Counter', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.direction_detector.reset_counts()
                    self.unique_visitors = 0
                    self.standby_slots.clear()
                    if self.fusion_engine:
                        self.fusion_engine.reset_counts()
                    logger.info("Counts reset")
                elif key == ord('s'):
                    entries, exits, occupancy = self.direction_detector.get_counts()
                    print(f"\n=== Current Stats ===")
                    print(f"Entries: {entries}")
                    print(f"Exits: {exits}")
                    print(f"Occupancy: {occupancy}")
                    print(f"Unique Visitors: {self.unique_visitors}")
                    print(f"Standby Slots: {len(self.standby_slots)}")
                    print(f"Flip Direction: {self.config.flip_direction}")
                    if self.fusion_engine:
                        print(f"Fusion Stats: {self.fusion_engine.get_stats()}")
                    print(f"====================\n")
                elif key == ord('c'):
                    self.config.calibrate_mode = not self.config.calibrate_mode
                    if self.fusion_engine:
                        if self.config.calibrate_mode:
                            self.fusion_engine.start_calibration()
                        else:
                            self.fusion_engine.stop_calibration()
                    print(f"Calibration mode: {'ON' if self.config.calibrate_mode else 'OFF'}")
                elif key == ord('f'):
                    self.config.flip_direction = not self.config.flip_direction
                    if self.fusion_engine:
                        self.fusion_engine.set_flip(self.config.flip_direction)
                    print(f"Direction flip: {'ON (IN<->OUT swapped)' if self.config.flip_direction else 'OFF'}")
        
        finally:
            self.stop()


def main():
    """Entry point with CLI argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='People Counter with Direction Detection (v2)')
    parser.add_argument('--source', type=str, default=None,
                        help='Video source: 0 for webcam, URL for IP cam, or file path')
    parser.add_argument('--model', type=str, default='yolov8s.pt',
                        help='YOLO model path')
    parser.add_argument('--confidence', type=float, default=None,
                        help='Detection confidence threshold')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable visualization')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for events')
    parser.add_argument('--near-threshold', type=float, default=None,
                        help='Near-edge threshold (0-1, bottom of frame)')
    parser.add_argument('--no-diagnostics', action='store_true',
                        help='Disable diagnostic report generation')
    parser.add_argument('--diagnostics-dir', type=str, default='.',
                        help='Directory for diagnostic output files')
    parser.add_argument('--no-plots', action='store_true',
                        help='Disable diagnostic plot generation')
    
    # Sensor fusion v2 arguments
    parser.add_argument('--fusion', action='store_true',
                        help='Enable sensor fusion v2 with ESP32 ToF')
    parser.add_argument('--fusion-port', type=str, default='COM3',
                        help='ESP32 serial port (default: COM3)')
    parser.add_argument('--fusion-strategy', type=str, default='dual_zone',
                        choices=['confirmation', 'dual_zone', 'hybrid'],
                        help='Fusion strategy: confirmation, dual_zone (default), or hybrid')
    
    # Direction and standby arguments
    parser.add_argument('--flip', action='store_true',
                        help='Flip direction: IN becomes OUT, OUT becomes IN')
    parser.add_argument('--standby-timeout', type=float, default=0,
                        help='Standby slot timeout in seconds (0 = infinite, default)')
    parser.add_argument('--calibrate', action='store_true',
                        help='Start in calibration mode (shows raw sensor coordinates)')
    
    args = parser.parse_args()
    
    # Create config with QUICK CONFIG defaults
    config = PeopleCounterConfig(
        model_path=args.model,
        show_visualization=not args.no_display,
        output_file=args.output,
        enable_diagnostics=not args.no_diagnostics,
        diagnostics_output_dir=args.diagnostics_dir,
        generate_plots=not args.no_plots
    )
    
    # Override with command line args if provided
    if args.source is not None:
        config.camera_source = args.source
    if args.confidence is not None:
        config.detection_confidence = args.confidence
    if args.near_threshold is not None:
        config.direction_config.near_edge_threshold = args.near_threshold
    
    # Apply fusion settings
    if args.fusion:
        if not HAS_FUSION:
            print("  WARNING: sensor_fusion_v2.py not found. Fusion disabled.")
            print("   Make sure sensor_fusion_v2.py is in the same directory.")
        else:
            config.enable_fusion = True
            config.fusion_port = args.fusion_port
            config.fusion_strategy = args.fusion_strategy
    
    # Apply flip and standby settings
    if args.flip:
        config.flip_direction = True
    if args.standby_timeout > 0:
        config.standby_timeout_seconds = args.standby_timeout
    if args.calibrate:
        config.calibrate_mode = True
    
    print("=" * 60)
    print("    PEOPLE COUNTER v2.0 - Sensor Fusion v2 (ToF)")
    print("=" * 60)
    print(f"\n  CAMERA SOURCE: {config.camera_source}")
    print(f"\n  CURRENT THRESHOLDS:")
    print(f"    detection_confidence:    {config.detection_confidence}")
    print(f"    near_edge_threshold:     {config.direction_config.near_edge_threshold}")
    print(f"    spawn_far_threshold:     {config.direction_config.spawn_far_threshold}")
    print(f"    spawn_near_threshold:    {config.direction_config.spawn_near_threshold}")
    print(f"    area_growth_for_approach:{config.direction_config.area_growth_for_approach}")
    print(f"    area_shrink_for_depart:  {config.direction_config.area_shrink_for_depart}")
    print(f"    min_track_duration_ms:   {config.direction_config.min_track_duration_ms}")
    print(f"    count_on_crossing:       {config.direction_config.count_on_crossing} {'(recommended)' if config.direction_config.count_on_crossing else '(legacy)'}")
    print(f"    crossing_hysteresis:     {config.direction_config.crossing_hysteresis}")
    print(f"    standby_timeout_seconds: {config.standby_timeout_seconds} {'(infinite)' if config.standby_timeout_seconds == 0 else ''}")
    print(f"    flip_direction:          {config.flip_direction}")
    
    if config.enable_fusion:
        print(f"\n  SENSOR FUSION v2 ENABLED:")
        print(f"    port:     {config.fusion_port}")
        print(f"    strategy: {config.fusion_strategy}")
        print(f"    threshold:{config.fusion_confidence_threshold}")
    
    if config.calibrate_mode:
        print(f"\n  CALIBRATION MODE: Watch console for CAL,... messages")
    
    print(f"\n  Edit QUICK CONFIG at top of people_counter.py to change defaults")
    print("=" * 60)
    print("  Controls:")
    print("    q - Quit and generate report")
    print("    r - Reset counts")
    print("    s - Show current stats")
    print("    c - Toggle calibration mode (sensor raw data)")
    print("    f - Toggle direction flip")
    print("=" * 60 + "\n")
    
    # Run counter
    counter = PeopleCounter(config)
    counter.run()


if __name__ == '__main__':
    main()
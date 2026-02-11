"""
People Counter: Full Detection Pipeline

Integrates:
- YOLOv8 for person detection
- ByteTrack for multi-object tracking
- DirectionDetector for entry/exit classification
- Standby Slot Method for unique visitor counting
- Live diagnostics and analysis

Supports:
- DroidCam/IP camera input
- USB/built-in camera
- Video file input (for testing)

Author: Ahmad's Mosque Attendance System
Version: 1.1.0
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

# ===== NEW: Threshold crossing settings =====
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

# Try to import sensor fusion (optional - for ESP32 radar/ToF integration)
try:
    from sensor_fusion import (
        SensorFusionEngine,
        CameraEvent,
        create_camera_event_from_detector_event,
        get_track_history_for_event
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
        self.track_histories: Dict[int, List[dict]] = defaultdict(list)
        self.completed_tracks: List[dict] = []
        self.start_time: float = time.time()
        self.frame_count: int = 0
        self.detection_counts: List[int] = []  # Detections per frame
        
    def record_frame(self, detections: List[Tuple[int, Tuple[float, float, float, float]]], 
                     frame_num: int, frame_height: int):
        """Record detections for analysis"""
        self.frame_count = frame_num
        self.detection_counts.append(len(detections))
        
        for track_id, (x1, y1, x2, y2) in detections:
            area = (x2 - x1) * (y2 - y1)
            centroid_y = (y1 + y2) / 2
            y_normalized = centroid_y / frame_height
            
            self.track_histories[track_id].append({
                'frame': frame_num,
                'area': area,
                'y_normalized': y_normalized,
                'bbox': (x1, y1, x2, y2),
                'timestamp': time.time()
            })
    
    def finalize_track(self, track_id: int, direction: Optional[str], 
                       confidence: Optional[str], event_data: Optional[dict] = None):
        """Mark track as complete with final classification"""
        if track_id in self.track_histories:
            history = self.track_histories[track_id]
            if len(history) > 0:
                track_data = {
                    'track_id': track_id,
                    'direction': direction,
                    'confidence': confidence,
                    'duration_frames': len(history),
                    'duration_seconds': history[-1]['timestamp'] - history[0]['timestamp'],
                    'spawn_y': history[0]['y_normalized'],
                    'termination_y': history[-1]['y_normalized'],
                    'spawn_area': history[0]['area'],
                    'termination_area': history[-1]['area'],
                    'area_ratio': history[-1]['area'] / history[0]['area'] if history[0]['area'] > 0 else 1.0,
                    'y_travel': abs(history[-1]['y_normalized'] - history[0]['y_normalized']),
                }
                if event_data:
                    track_data.update(event_data)
                self.completed_tracks.append(track_data)
            # Clean up history to save memory
            del self.track_histories[track_id]
    
    def _analyze_pattern(self, tracks: List[dict], label: str) -> dict:
        """Analyze patterns in a set of tracks"""
        if not tracks:
            return {'count': 0}
        
        spawn_ys = [t['spawn_y'] for t in tracks]
        term_ys = [t['termination_y'] for t in tracks]
        area_ratios = [t['area_ratio'] for t in tracks]
        durations = [t['duration_frames'] for t in tracks]
        y_travels = [t['y_travel'] for t in tracks]
        
        return {
            'count': len(tracks),
            'spawn_y': {
                'mean': float(np.mean(spawn_ys)),
                'std': float(np.std(spawn_ys)),
                'min': float(np.min(spawn_ys)),
                'max': float(np.max(spawn_ys))
            },
            'termination_y': {
                'mean': float(np.mean(term_ys)),
                'std': float(np.std(term_ys)),
                'min': float(np.min(term_ys)),
                'max': float(np.max(term_ys))
            },
            'area_ratio': {
                'mean': float(np.mean(area_ratios)),
                'std': float(np.std(area_ratios)),
                'min': float(np.min(area_ratios)),
                'max': float(np.max(area_ratios))
            },
            'duration_frames': {
                'mean': float(np.mean(durations)),
                'std': float(np.std(durations)),
                'min': int(np.min(durations)),
                'max': int(np.max(durations))
            },
            'y_travel': {
                'mean': float(np.mean(y_travels)),
                'std': float(np.std(y_travels)),
                'min': float(np.min(y_travels)),
                'max': float(np.max(y_travels))
            }
        }
    
    def _calculate_recommended_thresholds(self) -> dict:
        """Calculate recommended threshold values based on observed patterns"""
        entries = [t for t in self.completed_tracks if t['direction'] == 'IN']
        exits = [t for t in self.completed_tracks if t['direction'] == 'OUT']
        
        recommendations = {}
        
        if entries:
            entry_term_ys = [t['termination_y'] for t in entries]
            recommendations['near_edge_threshold'] = float(np.percentile(entry_term_ys, 25))
            
            entry_spawn_ys = [t['spawn_y'] for t in entries]
            recommendations['spawn_far_threshold'] = float(np.percentile(entry_spawn_ys, 75))
            
            entry_ratios = [t['area_ratio'] for t in entries]
            recommendations['area_growth_for_approach'] = float(np.percentile(entry_ratios, 25))
        
        if exits:
            exit_spawn_ys = [t['spawn_y'] for t in exits]
            recommendations['spawn_near_threshold'] = float(np.percentile(exit_spawn_ys, 25))
            
            exit_ratios = [t['area_ratio'] for t in exits]
            recommendations['area_shrink_for_depart'] = float(np.percentile(exit_ratios, 75))
        
        valid_tracks = [t for t in self.completed_tracks if t['direction']]
        if valid_tracks:
            durations = [t['duration_frames'] for t in valid_tracks]
            recommendations['min_track_duration_frames'] = int(np.percentile(durations, 10))
        
        return recommendations
    
    def generate_report(self) -> dict:
        """Generate comprehensive diagnostic report"""
        runtime = time.time() - self.start_time
        
        # Finalize any remaining active tracks as abandoned
        for track_id in list(self.track_histories.keys()):
            self.finalize_track(track_id, None, None)
        
        if not self.completed_tracks:
            return {
                'error': 'No tracks recorded',
                'runtime_seconds': runtime,
                'total_frames': self.frame_count
            }
        
        # Separate by direction
        entries = [t for t in self.completed_tracks if t['direction'] == 'IN']
        exits = [t for t in self.completed_tracks if t['direction'] == 'OUT']
        abandoned = [t for t in self.completed_tracks if t['direction'] is None]
        
        # Confidence breakdown
        high_conf = [t for t in self.completed_tracks if t.get('confidence') == 'HIGH']
        med_conf = [t for t in self.completed_tracks if t.get('confidence') == 'MEDIUM']
        low_conf = [t for t in self.completed_tracks if t.get('confidence') == 'LOW']
        
        # Calculate FPS
        avg_fps = self.frame_count / runtime if runtime > 0 else 0
        
        # Detection statistics
        avg_detections = np.mean(self.detection_counts) if self.detection_counts else 0
        max_detections = max(self.detection_counts) if self.detection_counts else 0
        
        report = {
            'session_info': {
                'runtime_seconds': round(runtime, 1),
                'runtime_formatted': f"{int(runtime // 60)}m {int(runtime % 60)}s",
                'total_frames': self.frame_count,
                'average_fps': round(avg_fps, 1),
                'timestamp': datetime.now().isoformat()
            },
            'detection_stats': {
                'avg_people_per_frame': round(avg_detections, 2),
                'max_simultaneous_people': max_detections,
                'total_tracks_analyzed': len(self.completed_tracks)
            },
            'counting_summary': {
                'total_entries': len(entries),
                'total_exits': len(exits),
                'abandoned_tracks': len(abandoned),
                'final_occupancy': len(entries) - len(exits)
            },
            'confidence_breakdown': {
                'high_confidence': len(high_conf),
                'medium_confidence': len(med_conf),
                'low_confidence': len(low_conf),
                'high_confidence_pct': round(100 * len(high_conf) / max(len(entries) + len(exits), 1), 1)
            },
            'entry_patterns': self._analyze_pattern(entries, 'ENTRY'),
            'exit_patterns': self._analyze_pattern(exits, 'EXIT'),
            'abandoned_patterns': self._analyze_pattern(abandoned, 'ABANDONED'),
            'recommended_thresholds': self._calculate_recommended_thresholds()
        }
        
        return report
    
    def print_report(self, report: dict):
        """Print formatted diagnostic report to console"""
        print("\n" + "="*70)
        print("                    DIAGNOSTIC REPORT")
        print("="*70)
        
        # Session Info
        info = report.get('session_info', {})
        print(f"\n📊 SESSION INFO")
        print(f"   Runtime: {info.get('runtime_formatted', 'N/A')}")
        print(f"   Frames processed: {info.get('total_frames', 0):,}")
        print(f"   Average FPS: {info.get('average_fps', 0)}")
        
        # Detection Stats
        det = report.get('detection_stats', {})
        print(f"\n👁 DETECTION STATS")
        print(f"   Avg people per frame: {det.get('avg_people_per_frame', 0)}")
        print(f"   Max simultaneous: {det.get('max_simultaneous_people', 0)}")
        print(f"   Total tracks analyzed: {det.get('total_tracks_analyzed', 0)}")
        
        # Counting Summary
        count = report.get('counting_summary', {})
        print(f"\n🚶 COUNTING SUMMARY")
        print(f"   Entries: {count.get('total_entries', 0)}")
        print(f"   Exits: {count.get('total_exits', 0)}")
        print(f"   Abandoned: {count.get('abandoned_tracks', 0)}")
        print(f"   Final occupancy: {count.get('final_occupancy', 0)}")
        
        # Confidence
        conf = report.get('confidence_breakdown', {})
        print(f"\n✅ CONFIDENCE BREAKDOWN")
        print(f"   HIGH:   {conf.get('high_confidence', 0)}")
        print(f"   MEDIUM: {conf.get('medium_confidence', 0)}")
        print(f"   LOW:    {conf.get('low_confidence', 0)}")
        print(f"   High confidence rate: {conf.get('high_confidence_pct', 0)}%")
        
        # Entry Patterns
        entry = report.get('entry_patterns', {})
        if entry.get('count', 0) > 0:
            print(f"\n📥 ENTRY PATTERNS (n={entry['count']})")
            print(f"   Spawn Y:      {entry['spawn_y']['mean']:.2f} ± {entry['spawn_y']['std']:.2f}")
            print(f"   Terminate Y:  {entry['termination_y']['mean']:.2f} ± {entry['termination_y']['std']:.2f}")
            print(f"   Area ratio:   {entry['area_ratio']['mean']:.2f} ± {entry['area_ratio']['std']:.2f}")
            print(f"   Duration:     {entry['duration_frames']['mean']:.0f} frames (range: {entry['duration_frames']['min']}-{entry['duration_frames']['max']})")
        
        # Exit Patterns
        exit_p = report.get('exit_patterns', {})
        if exit_p.get('count', 0) > 0:
            print(f"\n📤 EXIT PATTERNS (n={exit_p['count']})")
            print(f"   Spawn Y:      {exit_p['spawn_y']['mean']:.2f} ± {exit_p['spawn_y']['std']:.2f}")
            print(f"   Terminate Y:  {exit_p['termination_y']['mean']:.2f} ± {exit_p['termination_y']['std']:.2f}")
            print(f"   Area ratio:   {exit_p['area_ratio']['mean']:.2f} ± {exit_p['area_ratio']['std']:.2f}")
            print(f"   Duration:     {exit_p['duration_frames']['mean']:.0f} frames (range: {exit_p['duration_frames']['min']}-{exit_p['duration_frames']['max']})")
        
        # Recommendations
        rec = report.get('recommended_thresholds', {})
        if rec:
            print(f"\n⚙️ RECOMMENDED THRESHOLDS")
            for key, value in rec.items():
                print(f"   {key}: {value:.3f}" if isinstance(value, float) else f"   {key}: {value}")
        
        print("\n" + "="*70)
    
    def generate_plots(self, output_path: str = 'diagnostic_plots.png') -> bool:
        """Generate visualization plots. Returns True if successful."""
        if not HAS_MATPLOTLIB:
            print("⚠️ matplotlib not installed - skipping plots")
            return False
        
        if not self.completed_tracks:
            print("⚠️ No data to plot")
            return False
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        entries = [t for t in self.completed_tracks if t['direction'] == 'IN']
        exits = [t for t in self.completed_tracks if t['direction'] == 'OUT']
        abandoned = [t for t in self.completed_tracks if t['direction'] is None]
        
        # Plot 1: Spawn vs Termination positions
        ax = axes[0, 0]
        if entries:
            ax.scatter([t['spawn_y'] for t in entries], 
                      [t['termination_y'] for t in entries],
                      c='green', label=f'Entry (n={len(entries)})', alpha=0.6, s=50)
        if exits:
            ax.scatter([t['spawn_y'] for t in exits],
                      [t['termination_y'] for t in exits],
                      c='red', label=f'Exit (n={len(exits)})', alpha=0.6, s=50)
        if abandoned:
            ax.scatter([t['spawn_y'] for t in abandoned],
                      [t['termination_y'] for t in abandoned],
                      c='gray', label=f'Abandoned (n={len(abandoned)})', alpha=0.4, s=30)
        ax.set_xlabel('Spawn Y (0=top, 1=bottom)')
        ax.set_ylabel('Termination Y')
        ax.set_title('Track Start vs End Position')
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Area ratio distribution
        ax = axes[0, 1]
        if entries:
            ax.hist([t['area_ratio'] for t in entries], bins=15, alpha=0.6, 
                   label='Entry', color='green')
        if exits:
            ax.hist([t['area_ratio'] for t in exits], bins=15, alpha=0.6,
                   label='Exit', color='red')
        ax.axvline(x=1.0, color='black', linestyle='--', label='No change')
        ax.set_xlabel('Area Ratio (final/initial)')
        ax.set_ylabel('Count')
        ax.set_title('Bounding Box Area Change')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Track duration distribution
        ax = axes[0, 2]
        all_durations = [t['duration_frames'] for t in self.completed_tracks]
        colors = ['green' if t['direction'] == 'IN' else 'red' if t['direction'] == 'OUT' else 'gray' 
                  for t in self.completed_tracks]
        ax.bar(range(len(all_durations)), all_durations, color=colors, alpha=0.7)
        ax.set_xlabel('Track Index')
        ax.set_ylabel('Duration (frames)')
        ax.set_title('Track Durations')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Y-travel vs Area ratio
        ax = axes[1, 0]
        valid_tracks = [t for t in self.completed_tracks if t['direction']]
        if valid_tracks:
            colors = ['green' if t['direction'] == 'IN' else 'red' for t in valid_tracks]
            ax.scatter([t['y_travel'] for t in valid_tracks],
                      [t['area_ratio'] for t in valid_tracks],
                      c=colors, alpha=0.6, s=50)
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Y Travel (normalized)')
        ax.set_ylabel('Area Ratio')
        ax.set_title('Movement vs Size Change')
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Detections over time
        ax = axes[1, 1]
        if self.detection_counts:
            ax.plot(self.detection_counts, 'b-', alpha=0.7, linewidth=0.5)
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
        print(f"📈 Plots saved to {output_path}")
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
    model_path: str = "yolov8s.pt"  # Use 'yolov8s' for better accuracy
    detection_confidence: float = DETECTION_CONFIDENCE
    nms_iou_threshold: float = 0.7     # Prevent box merging for groups
    person_class_id: int = 0           # COCO class ID for 'person'
    
    # Frame processing
    frame_width: int = 640
    frame_height: int = 480
    process_every_n_frames: int = 1    # Process every frame for accuracy
    
    # ByteTrack settings
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    track_buffer: int = 30             # Frames to keep lost tracks
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
    
    # Standby slot settings - uses QUICK CONFIG value
    standby_timeout_seconds: float = STANDBY_TIMEOUT_SECONDS
    
    # Visualization
    show_visualization: bool = True
    show_debug_overlay: bool = True
    visualization_scale: float = 1.0
    
    # Output
    log_events: bool = True
    output_file: Optional[str] = None  # JSON file to log events
    
    # Diagnostics
    enable_diagnostics: bool = True    # Collect data for diagnostic report
    diagnostics_output_dir: str = "."  # Where to save diagnostic files
    generate_plots: bool = True        # Generate diagnostic plots on exit
    
    # Sensor Fusion (ESP32 radar/ToF integration)
    enable_fusion: bool = False              # Enable sensor fusion with ESP32
    fusion_port: str = "COM3"                # ESP32 serial port (COM3 on Windows, /dev/ttyUSB0 on Linux)
    fusion_strategy: str = "confirmation"    # 'confirmation', 'tiebreaker', or 'crossing'
    fusion_confidence_threshold: float = 0.5 # Min confidence to count after fusion
    
    # Direction flip - swap IN/OUT
    flip_direction: bool = FLIP_DIRECTION    # True = IN becomes OUT, OUT becomes IN
    
    # Calibration mode - shows raw sensor data for coordinate mapping
    calibrate_mode: bool = False


class ByteTrackWrapper:
    """
    Wrapper for ByteTrack integration with YOLOv8.
    Uses the built-in tracker from ultralytics.
    """
    
    def __init__(self, config: PeopleCounterConfig):
        self.config = config
        # ByteTrack config for ultralytics
        self.tracker_config = {
            'tracker_type': 'bytetrack',
            'track_high_thresh': config.track_high_thresh,
            'track_low_thresh': config.track_low_thresh,
            'new_track_thresh': config.new_track_thresh,
            'track_buffer': config.track_buffer,
            'match_thresh': config.match_thresh
        }


class PeopleCounter:
    """
    Main people counting system.
    
    Workflow:
    1. Capture frame from camera
    2. Run YOLOv8 detection (person class only)
    3. Track detections with ByteTrack
    4. Feed tracks to DirectionDetector
    5. Update unique visitor count with Standby Slot Method
    """
    
    def __init__(self, config: Optional[PeopleCounterConfig] = None):
        self.config = config or PeopleCounterConfig()
        
        # Initialize YOLO model
        logger.info(f"Loading YOLO model: {self.config.model_path}")
        self.model = YOLO(self.config.model_path)
        
        # Initialize direction detector
        self.config.direction_config.frame_width = self.config.frame_width
        self.config.direction_config.frame_height = self.config.frame_height
        self.direction_detector = DirectionDetector(self.config.direction_config)
        
        # Initialize diagnostics analyzer
        self.analyzer: Optional[TrackAnalyzer] = None
        if self.config.enable_diagnostics:
            self.analyzer = TrackAnalyzer()
            logger.info("Diagnostics enabled - will generate report on exit")
        
        # Standby slots for unique visitor counting
        self.standby_slots: Deque[StandbySlot] = deque()
        self.unique_visitors = 0
        
        # Video capture
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Event log
        self.events: List[dict] = []
        
        # Track management for analyzer
        self.active_track_ids: set = set()
        
        # Performance tracking
        self.frame_count = 0
        self.fps_history: Deque[float] = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # State
        self.running = False
        
        # Sensor fusion engine (optional - for ESP32 radar/ToF)
        self.fusion_engine: Optional['SensorFusionEngine'] = None
        if HAS_FUSION and self.config.enable_fusion:
            self.fusion_engine = SensorFusionEngine(
                port=self.config.fusion_port,
                strategy=self.config.fusion_strategy,
                flip_direction=self.config.flip_direction
            )
            logger.info(f"Sensor fusion initialized: {self.config.fusion_strategy} on {self.config.fusion_port}")
    
    def start(self):
        """Initialize video capture and start processing"""
        source = self.config.camera_source
        
        # Parse camera source
        if source.isdigit():
            source = int(source)
        
        logger.info(f"Opening video source: {source}")
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Video resolution: {actual_width}x{actual_height}")
        
        # Update config with actual resolution
        self.config.frame_width = actual_width
        self.config.frame_height = actual_height
        self.config.direction_config.frame_width = actual_width
        self.config.direction_config.frame_height = actual_height
        
        self.running = True
        logger.info("People counter started")
        
        # Start sensor fusion if enabled
        if self.fusion_engine:
            self.fusion_engine.start()
            logger.info("Sensor fusion engine started")
            
            # Start calibration mode if configured
            if self.config.calibrate_mode:
                self.fusion_engine.start_calibration()
                logger.info("Calibration mode started")
    
    def stop(self):
        """Stop processing and release resources"""
        self.running = False
        
        # Stop sensor fusion if enabled
        if self.fusion_engine:
            self.fusion_engine.stop()
            logger.info("Sensor fusion engine stopped")
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Generate diagnostic report
        if self.analyzer:
            print("\n⏳ Generating diagnostic report...")
            report = self.analyzer.generate_report()
            
            # Print to console
            self.analyzer.print_report(report)
            
            # Save JSON report
            output_dir = Path(self.config.diagnostics_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            report_path = output_dir / f'diagnostic_report_{timestamp}.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"📄 Report saved to {report_path}")
            
            # Generate plots
            if self.config.generate_plots:
                plots_path = output_dir / f'diagnostic_plots_{timestamp}.png'
                self.analyzer.generate_plots(str(plots_path))
            
            # Store report for programmatic access
            self.diagnostic_report = report
        
        # Save events if configured
        if self.config.output_file and self.events:
            with open(self.config.output_file, 'w') as f:
                json.dump({
                    'events': self.events,
                    'summary': {
                        'total_entries': self.direction_detector.entry_count,
                        'total_exits': self.direction_detector.exit_count,
                        'unique_visitors': self.unique_visitors,
                        'current_occupancy': self.get_current_occupancy()
                    }
                }, f, indent=2)
            logger.info(f"Events saved to {self.config.output_file}")
        
        logger.info("People counter stopped")
    
    def process_frame(self) -> Tuple[Optional[np.ndarray], List[dict]]:
        """
        Process a single frame and return visualization + events.
        
        Returns:
            (frame_with_visualization, list_of_crossing_events)
        """
        if not self.cap or not self.running:
            return None, []
        
        ret, frame = self.cap.read()
        if not ret:
            return None, []
        
        self.frame_count += 1
        current_time = time.time()
        
        # Calculate FPS
        frame_time = current_time - self.last_frame_time
        self.fps_history.append(1.0 / max(frame_time, 0.001))
        self.last_frame_time = current_time
        
        # Skip frames if configured
        if self.frame_count % self.config.process_every_n_frames != 0:
            return frame, []
        
        # Run YOLO with tracking
        results = self.model.track(
            frame,
            persist=True,
            conf=self.config.detection_confidence,
            iou=self.config.nms_iou_threshold,
            classes=[self.config.person_class_id],
            verbose=False,
            tracker="bytetrack.yaml"
        )
        
        # Extract tracked detections
        detections = []
        current_track_ids = set()
        
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            for i, (box, track_id) in enumerate(zip(boxes.xyxy, boxes.id)):
                x1, y1, x2, y2 = box.cpu().numpy()
                tid = int(track_id.cpu().numpy())
                detections.append((tid, (float(x1), float(y1), float(x2), float(y2))))
                current_track_ids.add(tid)
        
        # Record frame data for diagnostics
        if self.analyzer:
            self.analyzer.record_frame(detections, self.frame_count, self.config.frame_height)
        
        # Feed to direction detector
        crossing_events = self.direction_detector.update(detections)
        
        # Process crossing events with Standby Slot Method
        for event in crossing_events:
            self._handle_crossing_event(event, current_time)
            
            # Record completed track in analyzer
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
        
        # Track terminated tracks (for analyzer)
        if self.analyzer:
            terminated = self.active_track_ids - current_track_ids
            for tid in terminated:
                # Check if it wasn't already finalized as a crossing
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
        Uses sensor fusion if enabled for improved accuracy.
        
        Rules:
        - EXIT at occupancy > 0: Create a standby slot
        - EXIT at occupancy = 0: Person wasn't counted! Add to unique + create standby
        - ENTRY with standby slot: Consume slot (returning person)
        - ENTRY without standby: New unique visitor
        """
        direction = event['direction']
        trigger = event.get('trigger', 'unknown')  # 'threshold_crossing' or 'track_termination'
        
        # Log the crossing event with trigger type
        logger.info(f"Crossing: {direction} (conf={event.get('confidence', '?')}, "
                   f"trigger={trigger}, track={event['track_id']})")
        
        # Apply direction flip if enabled
        if self.config.flip_direction:
            direction = 'OUT' if direction == 'IN' else 'IN'
        
        # Use sensor fusion if enabled
        if self.fusion_engine:
            # Get track history for trend data
            track_history = get_track_history_for_event(
                self.direction_detector, 
                event['track_id']
            )
            
            # Create camera event for fusion
            camera_event = create_camera_event_from_detector_event(event, track_history)
            
            # Apply flip to camera event too
            if self.config.flip_direction:
                camera_event.direction = 'OUT' if camera_event.direction == 'IN' else 'IN'
            
            # Get fused decision
            decision = self.fusion_engine.process_camera_event(camera_event)
            
            if decision is None:
                # Crossing strategy returned None (waiting for sensor confirmation)
                logger.debug(f"Track {event['track_id']}: Waiting for sensor confirmation")
                return
            
            if decision.confidence < self.config.fusion_confidence_threshold:
                logger.warning(f"Track {event['track_id']}: Low fusion confidence "
                              f"({decision.confidence:.2f}), skipping")
                return
            
            # Use fused direction
            direction = decision.direction
            logger.info(f"Fusion decision: {direction} (conf={decision.confidence:.2f}, "
                       f"src={decision.source})")
        
        # Get current occupancy for negative occupancy check
        _, _, occupancy = self.direction_detector.get_counts()
        
        # Apply Standby Slot Method with negative occupancy fix
        if direction == 'OUT':
            # Check if occupancy is already 0 (person wasn't counted on entry)
            if occupancy <= 0:
                # This person was inside but never counted!
                # Add them to unique visitors AND create standby slot
                self.unique_visitors += 1
                logger.info(f"Uncounted exit detected! Added unique visitor #{self.unique_visitors}")
            
            # Always create standby slot on exit
            slot = StandbySlot(created_time=current_time, 
                              timeout_seconds=self.config.standby_timeout_seconds)
            self.standby_slots.append(slot)
            logger.debug(f"Created standby slot (total: {len(self.standby_slots)})")
            
        elif direction == 'IN':
            # Person entering - check for standby slot
            if self.standby_slots:
                # Consume oldest standby slot (returning person)
                self.standby_slots.popleft()
                logger.debug(f"Consumed standby slot (remaining: {len(self.standby_slots)})")
            else:
                # No standby slot - new unique visitor
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
        
        # Draw bounding boxes with track info
        for track_id, (x1, y1, x2, y2) in detections:
            # Get track state for coloring
            debug_info = self.direction_detector.get_debug_info(track_id)
            
            if debug_info:
                state = debug_info['state']
                if state == 'APPROACHING':
                    color = (0, 255, 0)  # Green - approaching
                elif state == 'DEPARTING':
                    color = (0, 0, 255)  # Red - departing
                elif state == 'STABLE':
                    color = (255, 255, 0)  # Cyan - stable
                else:
                    color = (128, 128, 128)  # Gray - nascent
            else:
                color = (255, 255, 255)  # White - unknown
            
            # Draw bbox
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw track ID and state
            label = f"ID:{track_id}"
            if debug_info:
                label += f" {debug_info['state'][:3]}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw area trend indicator
            if debug_info:
                trend = debug_info['area_trend']
                if trend == 'GROWING':
                    cv2.arrowedLine(frame, (int(x2) + 10, int(y2)), 
                                   (int(x2) + 10, int(y1)), (0, 255, 0), 2)
                elif trend == 'SHRINKING':
                    cv2.arrowedLine(frame, (int(x2) + 10, int(y1)), 
                                   (int(x2) + 10, int(y2)), (0, 0, 255), 2)
        
        # Draw crossing events
        for event in events:
            direction = event['direction']
            confidence = event['confidence']
            color = (0, 255, 0) if direction == 'IN' else (0, 0, 255)
            text = f"{direction} ({confidence})"
            cv2.putText(frame, text, (self.config.frame_width // 2 - 50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        
        # Draw stats overlay
        if self.config.show_debug_overlay:
            self._draw_stats_overlay(frame)
        
        return frame
    
    def _draw_stats_overlay(self, frame: np.ndarray):
        """Draw statistics overlay"""
        # Background for stats
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (250, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Stats text
        entries, exits, occupancy = self.direction_detector.get_counts()
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        
        stats = [
            f"FPS: {avg_fps:.1f}",
            f"Entries: {entries}",
            f"Exits: {exits}",
            f"Occupancy: {occupancy}",
            f"Unique: {self.unique_visitors}",
            f"Standby: {len(self.standby_slots)}"
        ]
        
        y_offset = 30
        for stat in stats:
            cv2.putText(frame, stat, (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 22
    
    def get_current_occupancy(self) -> int:
        """Get current number of people inside"""
        _, _, occupancy = self.direction_detector.get_counts()
        return max(0, occupancy)  # Don't go negative
    
    def get_unique_visitors(self) -> int:
        """Get total unique visitors"""
        return self.unique_visitors
    
    def run(self):
        """Main processing loop"""
        self.start()
        
        try:
            while self.running:
                frame, events = self.process_frame()
                
                if frame is None:
                    break
                
                if self.config.show_visualization:
                    # Scale if needed
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
                    # Reset counts
                    self.direction_detector.reset_counts()
                    self.unique_visitors = 0
                    self.standby_slots.clear()
                    logger.info("Counts reset")
                elif key == ord('s'):
                    # Print current stats
                    entries, exits, occupancy = self.direction_detector.get_counts()
                    print(f"\n=== Current Stats ===")
                    print(f"Entries: {entries}")
                    print(f"Exits: {exits}")
                    print(f"Occupancy: {occupancy}")
                    print(f"Unique Visitors: {self.unique_visitors}")
                    print(f"Standby Slots: {len(self.standby_slots)}")
                    print(f"Flip Direction: {self.config.flip_direction}")
                    print(f"====================\n")
                elif key == ord('c'):
                    # Toggle calibration mode
                    self.config.calibrate_mode = not self.config.calibrate_mode
                    if self.fusion_engine:
                        if self.config.calibrate_mode:
                            self.fusion_engine.start_calibration()
                        else:
                            self.fusion_engine.stop_calibration()
                    print(f"Calibration mode: {'ON' if self.config.calibrate_mode else 'OFF'}")
                elif key == ord('f'):
                    # Toggle direction flip
                    self.config.flip_direction = not self.config.flip_direction
                    if self.fusion_engine:
                        self.fusion_engine.set_flip(self.config.flip_direction)
                    print(f"Direction flip: {'ON (IN<->OUT swapped)' if self.config.flip_direction else 'OFF'}")
        
        finally:
            self.stop()


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='People Counter with Direction Detection')
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
    
    # Sensor fusion arguments
    parser.add_argument('--fusion', action='store_true',
                        help='Enable sensor fusion with ESP32 (radar/ToF)')
    parser.add_argument('--fusion-port', type=str, default='COM3',
                        help='ESP32 serial port (default: COM3)')
    parser.add_argument('--fusion-strategy', type=str, default='confirmation',
                        choices=['confirmation', 'tiebreaker', 'crossing'],
                        help='Fusion strategy: confirmation (default), tiebreaker, or crossing')
    
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
            print("⚠️  WARNING: sensor_fusion.py not found. Fusion disabled.")
            print("   Make sure sensor_fusion.py is in the same directory.")
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
    
    print("="*60)
    print("        PEOPLE COUNTER - Track Lifecycle Detection")
    print("="*60)
    print(f"\n📷 CAMERA SOURCE: {config.camera_source}")
    print(f"\n⚙️  CURRENT THRESHOLDS:")
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
        print(f"\n🔌 SENSOR FUSION ENABLED:")
        print(f"    port:     {config.fusion_port}")
        print(f"    strategy: {config.fusion_strategy}")
        print(f"    threshold:{config.fusion_confidence_threshold}")
    
    if config.calibrate_mode:
        print(f"\n🔧 CALIBRATION MODE: Watch console for CAL,... messages")
    
    print(f"\n💡 Edit QUICK CONFIG at top of people_counter.py to change defaults")
    print("="*60)
    print("  Controls:")
    print("    q - Quit and generate report")
    print("    r - Reset counts")
    print("    s - Show current stats")
    print("    c - Toggle calibration mode (sensor raw data)")
    print("    f - Toggle direction flip")
    print("="*60 + "\n")
    
    # Run counter
    counter = PeopleCounter(config)
    counter.run()


if __name__ == '__main__':
    main()

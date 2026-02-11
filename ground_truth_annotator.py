"""
Ground Truth Annotator: Video Review Tool for People Counter Optimization

Review recorded video with YOLO detections visible (bounding boxes + track IDs).
Manually label each tracked person as IN or OUT by their track ID.
Export ground truth data for analyzing optimal thresholds.

Controls:
    SPACE       - Pause/Resume playback
    LEFT/RIGHT  - Frame-by-frame when paused (hold for fast)
    +/-         - Speed up/slow down playback
    I           - Label a track as IN (enter track ID after)
    O           - Label a track as OUT (enter track ID after)
    U           - Undo last label
    D           - Delete label for a specific track
    R           - Toggle recording mode (record video from camera)
    S           - Save ground truth data
    T           - Toggle track trails (show movement history)
    Z           - Toggle zone lines overlay
    TAB         - Cycle info panel views
    Q/ESC       - Quit and save

Usage:
    # Review a recorded video
    python ground_truth_annotator.py --source recording.mp4
    
    # Record from camera first, then review
    python ground_truth_annotator.py --source 0 --record
    
    # Review with extra-low confidence to see early detections
    python ground_truth_annotator.py --source recording.mp4 --confidence 0.15
    
    # Use DroidCam
    python ground_truth_annotator.py --source "http://192.168.1.100:4747/video" --record

Author: Ahmad's Mosque Attendance System
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Set
import time
import json
import argparse
import os
from datetime import datetime
from pathlib import Path


# ============================================================================
#                        CONFIGURATION
# ============================================================================

# YOLO Settings - LOW confidence to catch early detections
DETECTION_CONFIDENCE = 0.15     # Very low - we want to see everything
MODEL_PATH = "yolov8s.pt"

# ByteTrack - more aggressive tracking to maintain IDs
BYTETRACK_CONFIG = "bytetrack.yaml"

# Display
WINDOW_NAME = "Ground Truth Annotator"
DEFAULT_PLAYBACK_SPEED = 1.0    # 1.0 = normal speed
INFO_PANEL_WIDTH = 320          # Right-side info panel width

# Colors (BGR)
COLOR_IN = (0, 200, 0)         # Green for IN
COLOR_OUT = (0, 0, 200)        # Red for OUT  
COLOR_UNLABELED = (200, 200, 0) # Cyan for unlabeled
COLOR_ACTIVE = (0, 255, 255)   # Yellow for currently active track
COLOR_LOST = (128, 128, 128)   # Gray for lost tracks
COLOR_PANEL_BG = (30, 30, 30)  # Dark panel background
COLOR_TEXT = (220, 220, 220)    # Light text
COLOR_HIGHLIGHT = (0, 180, 255) # Orange highlight

# ============================================================================


@dataclass
class TrackData:
    """Complete data for a single tracked person."""
    track_id: int
    first_frame: int = 0
    last_frame: int = 0
    first_seen_time: float = 0.0
    
    # Position history
    bbox_history: List[Tuple[int, int, int, int]] = field(default_factory=list)  # x1,y1,x2,y2
    centroid_history: List[Tuple[float, float]] = field(default_factory=list)     # cx, cy normalized
    area_history: List[float] = field(default_factory=list)                       # normalized area
    confidence_history: List[float] = field(default_factory=list)
    frame_numbers: List[int] = field(default_factory=list)
    
    # Manual label
    label: Optional[str] = None  # "IN" or "OUT" or None
    label_frame: Optional[int] = None  # Frame when label was assigned
    
    # Computed stats (updated each frame)
    max_confidence: float = 0.0
    avg_confidence: float = 0.0
    total_frames_visible: int = 0
    
    # Movement analysis
    y_trend: float = 0.0          # Positive = moving down
    area_trend: float = 0.0       # Positive = growing
    max_area: float = 0.0
    min_area: float = 999.0
    
    def update_stats(self):
        """Recalculate statistics."""
        if self.confidence_history:
            self.max_confidence = max(self.confidence_history)
            self.avg_confidence = sum(self.confidence_history) / len(self.confidence_history)
        self.total_frames_visible = len(self.frame_numbers)
        
        if self.area_history:
            self.max_area = max(self.area_history)
            self.min_area = min(self.area_history)
        
        # Y trend (linear regression on last N points)
        if len(self.centroid_history) >= 3:
            recent_y = [p[1] for p in self.centroid_history[-10:]]
            n = len(recent_y)
            x = list(range(n))
            x_mean = sum(x) / n
            y_mean = sum(recent_y) / n
            num = sum((x[i] - x_mean) * (recent_y[i] - y_mean) for i in range(n))
            den = sum((x[i] - x_mean) ** 2 for i in range(n))
            self.y_trend = num / den if den > 0 else 0
        
        # Area trend
        if len(self.area_history) >= 3:
            recent_a = self.area_history[-10:]
            n = len(recent_a)
            x = list(range(n))
            x_mean = sum(x) / n
            a_mean = sum(recent_a) / n
            num = sum((x[i] - x_mean) * (recent_a[i] - a_mean) for i in range(n))
            den = sum((x[i] - x_mean) ** 2 for i in range(n))
            self.area_trend = num / den if den > 0 else 0


@dataclass
class AnnotationState:
    """Persistent annotation state."""
    labels: Dict[int, str] = field(default_factory=dict)          # track_id -> "IN"/"OUT"
    label_history: List[Tuple[int, str]] = field(default_factory=list)  # For undo
    input_mode: Optional[str] = None  # "IN", "OUT", "DELETE" or None
    input_buffer: str = ""            # Track ID being typed


class GroundTruthAnnotator:
    """
    Video review tool with YOLO detection overlay and manual labeling.
    """
    
    def __init__(self, source, confidence=0.15, model_path="yolov8s.pt",
                 record=False, output_dir=".", playback_speed=1.0):
        self.source = source
        self.confidence = confidence
        self.model_path = model_path
        self.record = record
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Video state
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.recording_path: Optional[str] = None
        self.total_frames = 0
        self.current_frame = 0
        self.fps = 30.0
        self.frame_width = 640
        self.frame_height = 480
        self.paused = False
        self.playback_speed = playback_speed
        self.is_video_file = False
        
        # YOLO
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        
        # Track data
        self.tracks: Dict[int, TrackData] = {}
        self.active_track_ids: Set[int] = set()
        self.annotation = AnnotationState()
        
        # Display options
        self.show_trails = True
        self.show_zones = True
        self.info_panel_mode = 0  # 0=active tracks, 1=all tracks, 2=labeled only
        self.selected_track_id: Optional[int] = None
        
        # For clicking on tracks
        self.last_frame_detections: List[Dict] = []
        
        # Status messages
        self.status_msg = ""
        self.status_time = 0
        
    def set_status(self, msg: str, duration: float = 2.0):
        """Show a temporary status message."""
        self.status_msg = msg
        self.status_time = time.time() + duration
    
    def open_source(self):
        """Open video source."""
        # Determine if file or camera
        source = self.source
        if source.isdigit():
            source = int(source)
            self.is_video_file = False
        elif os.path.isfile(source):
            self.is_video_file = True
        else:
            # URL (DroidCam etc)
            self.is_video_file = False
        
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")
        
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        if self.is_video_file:
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Video: {self.frame_width}x{self.frame_height} @ {self.fps:.1f}fps, "
                  f"{self.total_frames} frames ({self.total_frames/self.fps:.1f}s)")
        else:
            print(f"Camera: {self.frame_width}x{self.frame_height} @ {self.fps:.1f}fps")
            if self.record:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.recording_path = str(self.output_dir / f"recording_{timestamp}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(
                    self.recording_path, fourcc, self.fps,
                    (self.frame_width, self.frame_height)
                )
                print(f"Recording to: {self.recording_path}")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run YOLO + ByteTrack and update track data."""
        self.current_frame += 1
        current_time = time.time()
        
        # Run YOLO with tracking
        results = self.model.track(
            frame,
            persist=True,
            conf=self.confidence,
            iou=0.7,
            classes=[0],  # person only
            verbose=False,
            tracker="bytetrack.yaml"
        )
        
        # Extract detections
        self.last_frame_detections = []
        current_ids = set()
        
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                track_id = int(boxes.id[i])
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i])
                
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                current_ids.add(track_id)
                
                # Normalized values
                cx = ((x1 + x2) / 2) / self.frame_width
                cy = ((y1 + y2) / 2) / self.frame_height
                area = ((x2 - x1) * (y2 - y1)) / (self.frame_width * self.frame_height)
                
                # Create or update track
                if track_id not in self.tracks:
                    self.tracks[track_id] = TrackData(
                        track_id=track_id,
                        first_frame=self.current_frame,
                        first_seen_time=current_time
                    )
                
                track = self.tracks[track_id]
                track.last_frame = self.current_frame
                track.bbox_history.append((x1, y1, x2, y2))
                track.centroid_history.append((cx, cy))
                track.area_history.append(area)
                track.confidence_history.append(conf)
                track.frame_numbers.append(self.current_frame)
                track.update_stats()
                
                # Carry over label from annotation state
                if track_id in self.annotation.labels:
                    track.label = self.annotation.labels[track_id]
                
                self.last_frame_detections.append({
                    'track_id': track_id,
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'centroid': (cx, cy),
                    'area': area
                })
        
        self.active_track_ids = current_ids
        return frame
    
    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw all visual overlays on the frame."""
        overlay = frame.copy()
        h, w = overlay.shape[:2]
        
        # Draw zone reference lines if enabled
        if self.show_zones:
            # Near edge threshold (from Counter_Cam.py config)
            near_y = int(0.584 * h)
            cv2.line(overlay, (0, near_y), (w, near_y), (0, 255, 255), 1)
            cv2.putText(overlay, "THRESHOLD 0.584", (5, near_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Spawn far
            far_y = int(0.382 * h)
            cv2.line(overlay, (0, far_y), (w, far_y), (100, 255, 100), 1)
            cv2.putText(overlay, "SPAWN_FAR 0.382", (5, far_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
            
            # Spawn near
            spawn_near_y = int(0.57 * h)
            cv2.line(overlay, (0, spawn_near_y), (w, spawn_near_y), (100, 100, 255), 1)
            cv2.putText(overlay, "SPAWN_NEAR 0.57", (5, spawn_near_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)
        
        # Draw each detection
        for det in self.last_frame_detections:
            tid = det['track_id']
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            track = self.tracks.get(tid)
            
            # Determine color based on label
            if track and track.label == "IN":
                color = COLOR_IN
                label_text = "IN"
            elif track and track.label == "OUT":
                color = COLOR_OUT
                label_text = "OUT"
            else:
                color = COLOR_UNLABELED
                label_text = "?"
            
            # Highlight selected track
            thickness = 3 if tid == self.selected_track_id else 2
            if tid == self.selected_track_id:
                color = COLOR_ACTIVE
            
            # Draw bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
            
            # Draw ID + confidence + label
            info_text = f"ID:{tid} {conf:.2f} [{label_text}]"
            text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(overlay, (x1, y1 - text_size[1] - 8), 
                         (x1 + text_size[0] + 4, y1), color, -1)
            cv2.putText(overlay, info_text, (x1 + 2, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Draw area and y-position below box
            if track:
                area_text = f"a:{det['area']:.4f} y:{det['centroid'][1]:.3f}"
                cv2.putText(overlay, area_text, (x1, y2 + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                
                # Draw movement arrows
                if track.y_trend != 0:
                    arrow_y = "↓" if track.y_trend > 0 else "↑"
                    arrow_a = "+" if track.area_trend > 0 else "-"
                    trend_text = f"y{arrow_y} a{arrow_a}"
                    cv2.putText(overlay, trend_text, (x1, y2 + 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            
            # Draw trail
            if self.show_trails and track and len(track.bbox_history) > 1:
                trail_points = []
                for bx1, by1, bx2, by2 in track.bbox_history[-20:]:
                    cx = (bx1 + bx2) // 2
                    cy = (by1 + by2) // 2
                    trail_points.append((cx, cy))
                for i in range(1, len(trail_points)):
                    alpha = i / len(trail_points)
                    thickness_t = max(1, int(alpha * 3))
                    cv2.line(overlay, trail_points[i-1], trail_points[i], 
                            color, thickness_t)
        
        # Draw input mode indicator
        if self.annotation.input_mode:
            mode_color = COLOR_IN if self.annotation.input_mode == "IN" else COLOR_OUT
            if self.annotation.input_mode == "DELETE":
                mode_color = (0, 128, 255)
            prompt = f"Enter Track ID for {self.annotation.input_mode}: {self.annotation.input_buffer}_"
            cv2.rectangle(overlay, (0, h - 40), (w, h), (0, 0, 0), -1)
            cv2.putText(overlay, prompt, (10, h - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        # Draw status message
        if self.status_msg and time.time() < self.status_time:
            cv2.rectangle(overlay, (0, 0), (w, 30), (0, 0, 0), -1)
            cv2.putText(overlay, self.status_msg, (10, 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_HIGHLIGHT, 2)
        
        # Playback info bar at top
        bar_text = f"Frame: {self.current_frame}"
        if self.total_frames > 0:
            bar_text += f"/{self.total_frames}"
            progress = self.current_frame / self.total_frames
            bar_text += f" ({progress*100:.1f}%)"
        bar_text += f" | Speed: {self.playback_speed:.1f}x"
        if self.paused:
            bar_text += " | PAUSED"
        if self.video_writer:
            bar_text += " | REC"
        bar_text += f" | Tracks: {len(self.active_track_ids)}"
        bar_text += f" | Labeled: {len(self.annotation.labels)}"
        
        cv2.rectangle(overlay, (0, 0), (w, 25), (40, 40, 40), -1)
        cv2.putText(overlay, bar_text, (5, 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1)
        
        # Draw progress bar for video files
        if self.total_frames > 0:
            bar_y = 25
            progress = self.current_frame / max(self.total_frames, 1)
            cv2.rectangle(overlay, (0, bar_y), (w, bar_y + 3), (60, 60, 60), -1)
            cv2.rectangle(overlay, (0, bar_y), (int(w * progress), bar_y + 3), COLOR_HIGHLIGHT, -1)
        
        return overlay
    
    def draw_info_panel(self, frame: np.ndarray) -> np.ndarray:
        """Draw the right-side information panel."""
        h = frame.shape[0]
        panel = np.zeros((h, INFO_PANEL_WIDTH, 3), dtype=np.uint8)
        panel[:] = COLOR_PANEL_BG
        
        y = 20
        line_h = 18
        
        # Panel title
        modes = ["ACTIVE TRACKS", "ALL TRACKS", "LABELED ONLY"]
        title = modes[self.info_panel_mode % len(modes)]
        cv2.putText(panel, f"=== {title} ===", (5, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HIGHLIGHT, 1)
        y += line_h + 5
        
        # Counters summary
        in_count = sum(1 for v in self.annotation.labels.values() if v == "IN")
        out_count = sum(1 for v in self.annotation.labels.values() if v == "OUT")
        cv2.putText(panel, f"IN: {in_count}  OUT: {out_count}  Total: {len(self.tracks)}", 
                   (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1)
        y += line_h + 5
        
        cv2.line(panel, (5, y), (INFO_PANEL_WIDTH - 5, y), (80, 80, 80), 1)
        y += 10
        
        # Get tracks to show based on mode
        if self.info_panel_mode == 0:
            # Active tracks only
            track_ids = sorted(self.active_track_ids)
        elif self.info_panel_mode == 1:
            # All tracks sorted by last seen (most recent first)
            track_ids = sorted(self.tracks.keys(), 
                             key=lambda tid: self.tracks[tid].last_frame, reverse=True)
        else:
            # Labeled only
            track_ids = sorted(self.annotation.labels.keys())
        
        for tid in track_ids:
            if y > h - 30:
                cv2.putText(panel, f"... +{len(track_ids) - track_ids.index(tid)} more",
                           (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_TEXT, 1)
                break
            
            track = self.tracks.get(tid)
            if not track:
                continue
            
            # Color based on label
            if tid in self.annotation.labels:
                label = self.annotation.labels[tid]
                color = COLOR_IN if label == "IN" else COLOR_OUT
                label_str = f"[{label}]"
            else:
                color = COLOR_UNLABELED if tid in self.active_track_ids else COLOR_LOST
                label_str = "[?]"
            
            # Highlight selected
            if tid == self.selected_track_id:
                cv2.rectangle(panel, (0, y - 12), (INFO_PANEL_WIDTH, y + line_h * 3 + 2), 
                            (60, 60, 60), -1)
            
            # Track ID and label
            id_text = f"ID:{tid:3d} {label_str}"
            cv2.putText(panel, id_text, (5, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            
            # Confidence
            conf_text = f"conf: {track.max_confidence:.2f} (avg:{track.avg_confidence:.2f})"
            cv2.putText(panel, conf_text, (15, y + line_h),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.33, (180, 180, 180), 1)
            
            # Area and movement
            if track.area_history:
                current_area = track.area_history[-1]
                area_text = f"area: {current_area:.4f} "
                if track.area_trend > 0.0001:
                    area_text += "GROWING"
                elif track.area_trend < -0.0001:
                    area_text += "SHRINKING"
                cv2.putText(panel, area_text, (15, y + line_h * 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.33, (180, 180, 180), 1)
            
            # Y position
            if track.centroid_history:
                cy = track.centroid_history[-1][1]
                y_text = f"y: {cy:.3f} "
                if track.y_trend > 0.001:
                    y_text += "DOWN↓"
                elif track.y_trend < -0.001:
                    y_text += "UP↑"
                cv2.putText(panel, y_text, (15, y + line_h * 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.33, (180, 180, 180), 1)
            
            y += line_h * 4 + 5
        
        # Help text at bottom
        help_y = h - 120
        cv2.line(panel, (5, help_y), (INFO_PANEL_WIDTH - 5, help_y), (80, 80, 80), 1)
        help_y += 15
        helps = [
            "I = label IN  |  O = label OUT",
            "U = undo  |  D = delete label",
            "SPACE = pause  |  </> = frame step",
            "+/- = speed  |  T = trails",
            "Z = zones  |  TAB = panel mode",
            "S = save  |  Q = quit & save",
            "Click box to select track"
        ]
        for line in helps:
            cv2.putText(panel, line, (5, help_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.33, (140, 140, 140), 1)
            help_y += 14
        
        # Combine frame and panel
        combined = np.hstack([frame, panel])
        return combined
    
    def handle_mouse(self, event, x, y, flags, param):
        """Handle mouse clicks to select tracks."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if click is on a detection box
            for det in self.last_frame_detections:
                bx1, by1, bx2, by2 = det['bbox']
                if bx1 <= x <= bx2 and by1 <= y <= by2:
                    self.selected_track_id = det['track_id']
                    self.set_status(f"Selected track ID: {det['track_id']}")
                    return
            self.selected_track_id = None
    
    def handle_key(self, key: int) -> bool:
        """Handle keyboard input. Returns False to quit."""
        
        # If in input mode, handle digit/enter/escape
        if self.annotation.input_mode:
            if key == 27:  # ESC - cancel input
                self.annotation.input_mode = None
                self.annotation.input_buffer = ""
                self.set_status("Cancelled")
                return True
            elif key == 13 or key == 10:  # Enter - confirm
                self._apply_label()
                return True
            elif key == 8 or key == 127:  # Backspace
                self.annotation.input_buffer = self.annotation.input_buffer[:-1]
                return True
            elif 48 <= key <= 57:  # Digits 0-9
                self.annotation.input_buffer += chr(key)
                return True
            return True
        
        # Normal mode
        if key == ord('q') or key == 27:
            self.save_ground_truth()
            return False
        elif key == ord(' '):
            self.paused = not self.paused
            self.set_status("PAUSED" if self.paused else "PLAYING")
        elif key == ord('i') or key == ord('I'):
            if self.selected_track_id is not None:
                # Quick label selected track
                self._label_track(self.selected_track_id, "IN")
            else:
                self.annotation.input_mode = "IN"
                self.annotation.input_buffer = ""
                self.set_status("Type track ID for IN, then Enter")
        elif key == ord('o') or key == ord('O'):
            if self.selected_track_id is not None:
                self._label_track(self.selected_track_id, "OUT")
            else:
                self.annotation.input_mode = "OUT"
                self.annotation.input_buffer = ""
                self.set_status("Type track ID for OUT, then Enter")
        elif key == ord('u') or key == ord('U'):
            self._undo_label()
        elif key == ord('d') or key == ord('D'):
            if self.selected_track_id is not None:
                self._delete_label(self.selected_track_id)
            else:
                self.annotation.input_mode = "DELETE"
                self.annotation.input_buffer = ""
                self.set_status("Type track ID to delete label, then Enter")
        elif key == ord('s') or key == ord('S'):
            self.save_ground_truth()
            self.set_status("Ground truth saved!")
        elif key == ord('t') or key == ord('T'):
            self.show_trails = not self.show_trails
            self.set_status(f"Trails: {'ON' if self.show_trails else 'OFF'}")
        elif key == ord('z') or key == ord('Z'):
            self.show_zones = not self.show_zones
            self.set_status(f"Zones: {'ON' if self.show_zones else 'OFF'}")
        elif key == 9:  # TAB
            self.info_panel_mode = (self.info_panel_mode + 1) % 3
        elif key == ord('+') or key == ord('='):
            self.playback_speed = min(4.0, self.playback_speed + 0.25)
            self.set_status(f"Speed: {self.playback_speed:.2f}x")
        elif key == ord('-') or key == ord('_'):
            self.playback_speed = max(0.1, self.playback_speed - 0.25)
            self.set_status(f"Speed: {self.playback_speed:.2f}x")
        elif key == 81 or key == 2:  # LEFT arrow
            if self.paused and self.is_video_file:
                # Step backward
                new_pos = max(0, self.current_frame - 2)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                self.current_frame = new_pos
        elif key == 83 or key == 3:  # RIGHT arrow  
            if self.paused and self.is_video_file:
                pass  # Just read next frame naturally
        
        return True
    
    def _apply_label(self):
        """Apply the current input buffer as a label."""
        try:
            track_id = int(self.annotation.input_buffer)
        except ValueError:
            self.set_status("Invalid track ID!")
            self.annotation.input_mode = None
            self.annotation.input_buffer = ""
            return
        
        if self.annotation.input_mode == "DELETE":
            self._delete_label(track_id)
        else:
            self._label_track(track_id, self.annotation.input_mode)
        
        self.annotation.input_mode = None
        self.annotation.input_buffer = ""
    
    def _label_track(self, track_id: int, direction: str):
        """Label a track as IN or OUT."""
        if track_id not in self.tracks:
            self.set_status(f"Track {track_id} not found!")
            return
        
        old_label = self.annotation.labels.get(track_id)
        self.annotation.labels[track_id] = direction
        self.annotation.label_history.append((track_id, old_label))
        self.tracks[track_id].label = direction
        self.tracks[track_id].label_frame = self.current_frame
        self.set_status(f"Track {track_id} labeled as {direction}")
    
    def _delete_label(self, track_id: int):
        """Remove label from a track."""
        if track_id in self.annotation.labels:
            old = self.annotation.labels.pop(track_id)
            self.annotation.label_history.append((track_id, old))
            if track_id in self.tracks:
                self.tracks[track_id].label = None
            self.set_status(f"Removed label from track {track_id}")
        else:
            self.set_status(f"Track {track_id} has no label")
    
    def _undo_label(self):
        """Undo the last label action."""
        if not self.annotation.label_history:
            self.set_status("Nothing to undo")
            return
        
        track_id, old_label = self.annotation.label_history.pop()
        if old_label is None:
            self.annotation.labels.pop(track_id, None)
            if track_id in self.tracks:
                self.tracks[track_id].label = None
            self.set_status(f"Undone: track {track_id} back to unlabeled")
        else:
            self.annotation.labels[track_id] = old_label
            if track_id in self.tracks:
                self.tracks[track_id].label = old_label
            self.set_status(f"Undone: track {track_id} back to {old_label}")
    
    def save_ground_truth(self):
        """Save all ground truth data to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"ground_truth_{timestamp}.json"
        
        # Build comprehensive export
        track_data = {}
        for tid, track in self.tracks.items():
            track_data[str(tid)] = {
                "track_id": tid,
                "label": self.annotation.labels.get(tid),
                "first_frame": track.first_frame,
                "last_frame": track.last_frame,
                "total_frames_visible": track.total_frames_visible,
                "max_confidence": round(track.max_confidence, 4),
                "avg_confidence": round(track.avg_confidence, 4),
                "max_area": round(track.max_area, 6),
                "min_area": round(track.min_area, 6),
                "area_trend": round(track.area_trend, 6),
                "y_trend": round(track.y_trend, 6),
                # First and last positions
                "first_centroid_y": round(track.centroid_history[0][1], 4) if track.centroid_history else None,
                "last_centroid_y": round(track.centroid_history[-1][1], 4) if track.centroid_history else None,
                "first_area": round(track.area_history[0], 6) if track.area_history else None,
                "last_area": round(track.area_history[-1], 6) if track.area_history else None,
                # Full history for deep analysis
                "centroid_y_history": [round(p[1], 4) for p in track.centroid_history],
                "area_history": [round(a, 6) for a in track.area_history],
                "confidence_history": [round(c, 4) for c in track.confidence_history],
                "frame_numbers": track.frame_numbers,
            }
        
        # Summary statistics
        in_tracks = {tid: d for tid, d in track_data.items() if d["label"] == "IN"}
        out_tracks = {tid: d for tid, d in track_data.items() if d["label"] == "OUT"}
        
        # Compute optimal threshold suggestions
        analysis = self._compute_analysis(in_tracks, out_tracks)
        
        output = {
            "metadata": {
                "source": str(self.source),
                "recording_path": self.recording_path,
                "timestamp": timestamp,
                "total_frames": self.current_frame,
                "fps": self.fps,
                "frame_size": [self.frame_width, self.frame_height],
                "yolo_confidence": self.confidence,
                "model": self.model_path,
            },
            "summary": {
                "total_tracks": len(self.tracks),
                "labeled_in": len(in_tracks),
                "labeled_out": len(out_tracks),
                "unlabeled": len(self.tracks) - len(in_tracks) - len(out_tracks),
            },
            "analysis": analysis,
            "tracks": track_data,
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"GROUND TRUTH SAVED: {output_path}")
        print(f"{'='*60}")
        print(f"Total tracks: {len(self.tracks)}")
        print(f"Labeled IN:   {len(in_tracks)}")
        print(f"Labeled OUT:  {len(out_tracks)}")
        print(f"Unlabeled:    {len(self.tracks) - len(in_tracks) - len(out_tracks)}")
        
        if analysis:
            print(f"\n--- THRESHOLD ANALYSIS ---")
            for key, val in analysis.items():
                if isinstance(val, dict):
                    print(f"\n{key}:")
                    for k, v in val.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"{key}: {val}")
        
        print(f"{'='*60}\n")
        return str(output_path)
    
    def _compute_analysis(self, in_tracks: Dict, out_tracks: Dict) -> Dict:
        """Analyze labeled tracks to find optimal thresholds."""
        if not in_tracks and not out_tracks:
            return {"note": "No labeled tracks - label some tracks for analysis"}
        
        analysis = {}
        
        # --- Confidence analysis ---
        in_confs = [t["max_confidence"] for t in in_tracks.values()]
        out_confs = [t["max_confidence"] for t in out_tracks.values()]
        all_confs = in_confs + out_confs
        
        if all_confs:
            analysis["confidence"] = {
                "min_of_labeled": round(min(all_confs), 4),
                "max_of_labeled": round(max(all_confs), 4),
                "avg_of_labeled": round(sum(all_confs) / len(all_confs), 4),
                "p10": round(sorted(all_confs)[max(0, len(all_confs)//10)], 4),
                "p25": round(sorted(all_confs)[max(0, len(all_confs)//4)], 4),
                "recommended_threshold": round(sorted(all_confs)[max(0, len(all_confs)//10)] * 0.9, 4),
                "note": "recommended = 90% of p10 (catches 90% of real people)"
            }
        
        # --- Area analysis ---
        in_areas_first = [t["first_area"] for t in in_tracks.values() if t["first_area"]]
        in_areas_last = [t["last_area"] for t in in_tracks.values() if t["last_area"]]
        out_areas_first = [t["first_area"] for t in out_tracks.values() if t["first_area"]]
        out_areas_last = [t["last_area"] for t in out_tracks.values() if t["last_area"]]
        
        in_area_ratios = []
        for t in in_tracks.values():
            if t["first_area"] and t["last_area"] and t["first_area"] > 0:
                in_area_ratios.append(t["last_area"] / t["first_area"])
        
        out_area_ratios = []
        for t in out_tracks.values():
            if t["first_area"] and t["last_area"] and t["first_area"] > 0:
                out_area_ratios.append(t["last_area"] / t["first_area"])
        
        analysis["area_ratios"] = {}
        if in_area_ratios:
            analysis["area_ratios"]["in_avg_ratio"] = round(sum(in_area_ratios) / len(in_area_ratios), 4)
            analysis["area_ratios"]["in_median_ratio"] = round(sorted(in_area_ratios)[len(in_area_ratios)//2], 4)
        if out_area_ratios:
            analysis["area_ratios"]["out_avg_ratio"] = round(sum(out_area_ratios) / len(out_area_ratios), 4)
            analysis["area_ratios"]["out_median_ratio"] = round(sorted(out_area_ratios)[len(out_area_ratios)//2], 4)
        
        if in_area_ratios and out_area_ratios:
            in_med = sorted(in_area_ratios)[len(in_area_ratios)//2]
            out_med = sorted(out_area_ratios)[len(out_area_ratios)//2]
            midpoint = (in_med + out_med) / 2
            analysis["area_ratios"]["recommended_growth_threshold"] = round(midpoint, 4)
            analysis["area_ratios"]["recommended_shrink_threshold"] = round(1 / midpoint if midpoint > 0 else 0.8, 4)
        
        # --- Y position analysis (spawn zones) ---
        in_first_y = [t["first_centroid_y"] for t in in_tracks.values() if t["first_centroid_y"] is not None]
        out_first_y = [t["first_centroid_y"] for t in out_tracks.values() if t["first_centroid_y"] is not None]
        in_last_y = [t["last_centroid_y"] for t in in_tracks.values() if t["last_centroid_y"] is not None]
        out_last_y = [t["last_centroid_y"] for t in out_tracks.values() if t["last_centroid_y"] is not None]
        
        analysis["y_positions"] = {}
        if in_first_y:
            analysis["y_positions"]["in_first_y_avg"] = round(sum(in_first_y) / len(in_first_y), 4)
            analysis["y_positions"]["in_first_y_min"] = round(min(in_first_y), 4)
            analysis["y_positions"]["in_first_y_max"] = round(max(in_first_y), 4)
        if out_first_y:
            analysis["y_positions"]["out_first_y_avg"] = round(sum(out_first_y) / len(out_first_y), 4)
            analysis["y_positions"]["out_first_y_min"] = round(min(out_first_y), 4)
            analysis["y_positions"]["out_first_y_max"] = round(max(out_first_y), 4)
        
        if in_first_y and out_first_y:
            in_avg_y = sum(in_first_y) / len(in_first_y)
            out_avg_y = sum(out_first_y) / len(out_first_y)
            analysis["y_positions"]["recommended_spawn_far"] = round(min(in_avg_y, out_avg_y) * 0.9, 4)
            analysis["y_positions"]["recommended_spawn_near"] = round(max(in_avg_y, out_avg_y) * 1.1, 4)
        
        # --- Y trend analysis (movement direction) ---
        in_y_trends = [t["y_trend"] for t in in_tracks.values()]
        out_y_trends = [t["y_trend"] for t in out_tracks.values()]
        
        analysis["movement"] = {}
        if in_y_trends:
            analysis["movement"]["in_y_trend_avg"] = round(sum(in_y_trends) / len(in_y_trends), 6)
        if out_y_trends:
            analysis["movement"]["out_y_trend_avg"] = round(sum(out_y_trends) / len(out_y_trends), 6)
        
        # --- Crossing threshold analysis ---
        if in_last_y and out_last_y:
            # Where do IN people end up? Where do OUT people end up?
            in_end_avg = sum(in_last_y) / len(in_last_y)
            out_end_avg = sum(out_last_y) / len(out_last_y)
            # The threshold should be between where IN and OUT tracks end
            threshold = (in_end_avg + out_end_avg) / 2
            analysis["crossing"] = {
                "in_endpoint_avg_y": round(in_end_avg, 4),
                "out_endpoint_avg_y": round(out_end_avg, 4),
                "recommended_near_edge_threshold": round(threshold, 4),
            }
        
        # --- Track duration analysis ---
        in_durations = [t["total_frames_visible"] for t in in_tracks.values()]
        out_durations = [t["total_frames_visible"] for t in out_tracks.values()]
        all_durations = in_durations + out_durations
        
        if all_durations:
            min_dur = min(all_durations)
            analysis["track_duration"] = {
                "min_frames_visible": min_dur,
                "avg_frames_visible": round(sum(all_durations) / len(all_durations), 1),
                "min_duration_ms": round(min_dur / self.fps * 1000, 0),
                "recommended_min_track_duration_ms": round(min_dur / self.fps * 1000 * 0.8, 0),
            }
        
        return analysis
    
    def run(self):
        """Main loop."""
        self.open_source()
        
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.handle_mouse)
        
        print("\n" + "="*50)
        print("GROUND TRUTH ANNOTATOR")
        print("="*50)
        print("Watch the video, label people as IN or OUT")
        print("Click a box to select, then press I or O")
        print("Or press I/O and type the track ID number")
        print("Press Q to quit and save analysis")
        print("="*50 + "\n")
        
        try:
            while True:
                if not self.paused or not self.is_video_file:
                    ret, frame = self.cap.read()
                    if not ret:
                        if self.is_video_file:
                            self.set_status("End of video - press Q to save & quit")
                            self.paused = True
                            # Wait for quit
                            while True:
                                key = cv2.waitKey(100) & 0xFF
                                if not self.handle_key(key):
                                    return
                                # Redraw with last frame
                                if hasattr(self, '_last_display'):
                                    cv2.imshow(WINDOW_NAME, self._last_display)
                            break
                        else:
                            continue
                    
                    # Record if enabled
                    if self.video_writer:
                        self.video_writer.write(frame)
                    
                    # Process
                    frame = self.process_frame(frame)
                    display = self.draw_overlay(frame)
                    display = self.draw_info_panel(display)
                    self._last_display = display
                elif hasattr(self, '_last_display'):
                    display = self._last_display
                    
                    # Still allow frame stepping when paused
                    key = cv2.waitKey(50) & 0xFF
                    if key != 255:
                        if not self.handle_key(key):
                            return
                        # If arrow right was pressed, read next frame
                        if (key == 83 or key == 3) and self.paused and self.is_video_file:
                            ret, frame = self.cap.read()
                            if ret:
                                frame = self.process_frame(frame)
                                display = self.draw_overlay(frame)
                                display = self.draw_info_panel(display)
                                self._last_display = display
                    
                    cv2.imshow(WINDOW_NAME, display)
                    continue
                
                cv2.imshow(WINDOW_NAME, display)
                
                # Calculate wait time based on playback speed
                wait_ms = max(1, int((1000 / self.fps) / self.playback_speed))
                key = cv2.waitKey(wait_ms) & 0xFF
                
                if key != 255:
                    if not self.handle_key(key):
                        return
        
        finally:
            if self.video_writer:
                self.video_writer.release()
                print(f"Recording saved to: {self.recording_path}")
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            
            # Auto-save on exit
            if self.annotation.labels:
                self.save_ground_truth()


def main():
    parser = argparse.ArgumentParser(description='Ground Truth Annotator for People Counter')
    parser.add_argument('--source', type=str, default='0',
                        help='Video source: file path, camera index, or URL')
    parser.add_argument('--confidence', type=float, default=0.15,
                        help='YOLO confidence threshold (default: 0.15 for early detection)')
    parser.add_argument('--model', type=str, default='yolov8s.pt',
                        help='YOLO model path')
    parser.add_argument('--record', action='store_true',
                        help='Record video from camera source')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory for output files')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Initial playback speed')
    
    args = parser.parse_args()
    
    annotator = GroundTruthAnnotator(
        source=args.source,
        confidence=args.confidence,
        model_path=args.model,
        record=args.record,
        output_dir=args.output_dir,
        playback_speed=args.speed,
    )
    annotator.run()


if __name__ == "__main__":
    main()

"""
Ground Truth Annotator: Video Review Tool for People Counter Optimization

Background scanner finds all track IDs while you annotate in real-time.
No waiting - start labeling immediately!

Controls:
    SPACE       - Pause/Resume playback
    LEFT/RIGHT  - Step 1 frame backward/forward
    < / >       - Jump 10 frames
    [ / ]       - Jump 100 frames
    H / E       - Jump to start / end
    I           - Label selected/typed track as IN
    O           - Label selected/typed track as OUT
    N           - Jump to next unlabeled track
    U           - Undo last label
    D           - Delete label for a track
    S           - Save ground truth data
    T           - Toggle track trails
    Z           - Toggle zone lines overlay
    TAB         - Cycle info panel views
    +/-         - Speed up/slow down playback
    Q/ESC       - Quit and save

Usage:
    python ground_truth_annotator.py --source recording.mp4
    python ground_truth_annotator.py --source recording.mp4 --confidence 0.15

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
import threading
from datetime import datetime
from pathlib import Path


# ============================================================================
#                        CONFIGURATION
# ============================================================================

MODEL_PATH = "yolov8s.pt"
DETECTION_CONFIDENCE = 0.15
SCAN_FRAME_SKIP = 3             # Process every Nth frame during background scan

WINDOW_NAME = "Ground Truth Annotator"
INFO_PANEL_WIDTH = 320

# Colors (BGR)
COLOR_IN = (0, 200, 0)
COLOR_OUT = (0, 0, 200)
COLOR_UNLABELED = (200, 200, 0)
COLOR_ACTIVE = (0, 255, 255)
COLOR_LOST = (128, 128, 128)
COLOR_PANEL_BG = (30, 30, 30)
COLOR_TEXT = (220, 220, 220)
COLOR_HIGHLIGHT = (0, 180, 255)
COLOR_SCAN_BAR = (0, 200, 200)

# ============================================================================


@dataclass
class TrackData:
    """Complete data for a single tracked person."""
    track_id: int
    first_frame: int = 0
    last_frame: int = 0
    first_seen_time: float = 0.0

    bbox_history: List[Tuple[int, int, int, int]] = field(default_factory=list)
    centroid_history: List[Tuple[float, float]] = field(default_factory=list)
    area_history: List[float] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)
    frame_numbers: List[int] = field(default_factory=list)

    label: Optional[str] = None
    label_frame: Optional[int] = None

    max_confidence: float = 0.0
    avg_confidence: float = 0.0
    total_frames_visible: int = 0
    y_trend: float = 0.0
    area_trend: float = 0.0
    max_area: float = 0.0
    min_area: float = 999.0

    # Representative bbox at first appearance
    first_bbox: Optional[Tuple[int, int, int, int]] = None

    def update_stats(self):
        if self.confidence_history:
            self.max_confidence = max(self.confidence_history)
            self.avg_confidence = sum(self.confidence_history) / len(self.confidence_history)
        self.total_frames_visible = len(self.frame_numbers)
        if self.area_history:
            self.max_area = max(self.area_history)
            self.min_area = min(self.area_history)
        if len(self.centroid_history) >= 3:
            recent_y = [p[1] for p in self.centroid_history[-10:]]
            n = len(recent_y)
            x = list(range(n))
            xm = sum(x) / n
            ym = sum(recent_y) / n
            num = sum((x[i] - xm) * (recent_y[i] - ym) for i in range(n))
            den = sum((x[i] - xm) ** 2 for i in range(n))
            self.y_trend = num / den if den > 0 else 0
        if len(self.area_history) >= 3:
            recent_a = self.area_history[-10:]
            n = len(recent_a)
            x = list(range(n))
            xm = sum(x) / n
            am = sum(recent_a) / n
            num = sum((x[i] - xm) * (recent_a[i] - am) for i in range(n))
            den = sum((x[i] - xm) ** 2 for i in range(n))
            self.area_trend = num / den if den > 0 else 0


@dataclass
class AnnotationState:
    labels: Dict[int, str] = field(default_factory=dict)
    label_history: List[Tuple[int, str]] = field(default_factory=list)
    input_mode: Optional[str] = None
    input_buffer: str = ""


# ============================================================================
#  BACKGROUND SCANNER - runs YOLO on separate thread
# ============================================================================

class BackgroundScanner:
    """Scans video with YOLO in background thread, populating tracks dict."""

    def __init__(self, video_path: str, model_path: str, confidence: float,
                 frame_width: int, frame_height: int,
                 frame_skip: int = SCAN_FRAME_SKIP):
        self.video_path = video_path
        self.model_path = model_path
        self.confidence = confidence
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_skip = frame_skip

        # Shared state (protected by lock)
        self.lock = threading.Lock()
        self.tracks: Dict[int, TrackData] = {}
        self.scan_frame = 0         # current scan position
        self.total_frames = 0
        self.scan_complete = False
        self.scan_running = False

        # Per-frame detection cache: frame_num -> list of detection dicts
        # Only stores detections for scanned frames (memory efficient)
        self.frame_detections: Dict[int, List[Dict]] = {}

        self._thread: Optional[threading.Thread] = None

    def start(self):
        self.scan_running = True
        self._thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.scan_running = False
        if self._thread:
            self._thread.join(timeout=3)

    @property
    def progress(self) -> float:
        if self.total_frames <= 0:
            return 0
        return min(1.0, self.scan_frame / self.total_frames)

    def get_tracks_snapshot(self) -> Dict[int, TrackData]:
        """Return a shallow copy of tracks dict (safe to read)."""
        with self.lock:
            return dict(self.tracks)

    def get_nearest_detections(self, frame_num: int) -> List[Dict]:
        """Get detections for nearest scanned frame."""
        with self.lock:
            # Exact match
            if frame_num in self.frame_detections:
                return self.frame_detections[frame_num]
            # Find nearest scanned frame within tolerance
            best_frame = None
            best_dist = float('inf')
            for f in self.frame_detections:
                d = abs(f - frame_num)
                if d < best_dist:
                    best_dist = d
                    best_frame = f
            if best_frame is not None and best_dist <= self.frame_skip + 1:
                return self.frame_detections[best_frame]
        return []

    def _scan_loop(self):
        """Background scan: read every Nth frame, run YOLO+ByteTrack."""
        print(f"[SCANNER] Starting background scan (every {self.frame_skip} frames)...")
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("[SCANNER] ERROR: Cannot open video!")
            self.scan_complete = True
            return

        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        model = YOLO(self.model_path)

        frame_area = self.frame_width * self.frame_height
        frame_num = 0

        while self.scan_running:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1

            # Skip frames for speed
            if frame_num % self.frame_skip != 0:
                continue

            self.scan_frame = frame_num

            # Run YOLO with tracking
            results = model.track(
                frame,
                persist=True,
                conf=self.confidence,
                iou=0.7,
                classes=[0],
                verbose=False,
                tracker="bytetrack.yaml"
            )

            detections = []
            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    track_id = int(boxes.id[i])
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i])
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    cx = ((x1 + x2) / 2) / self.frame_width
                    cy = ((y1 + y2) / 2) / self.frame_height
                    area = ((x2 - x1) * (y2 - y1)) / frame_area

                    det = {
                        'track_id': track_id,
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf,
                        'centroid': (cx, cy),
                        'area': area,
                    }
                    detections.append(det)

                    with self.lock:
                        if track_id not in self.tracks:
                            self.tracks[track_id] = TrackData(
                                track_id=track_id,
                                first_frame=frame_num,
                                first_seen_time=time.time(),
                                first_bbox=(x1, y1, x2, y2),
                            )
                        track = self.tracks[track_id]
                        track.last_frame = frame_num
                        track.bbox_history.append((x1, y1, x2, y2))
                        track.centroid_history.append((cx, cy))
                        track.area_history.append(area)
                        track.confidence_history.append(conf)
                        track.frame_numbers.append(frame_num)
                        track.update_stats()

            # Cache detections for this frame
            with self.lock:
                self.frame_detections[frame_num] = detections

            # Print progress periodically
            if frame_num % (self.frame_skip * 100) == 0:
                pct = frame_num / max(self.total_frames, 1) * 100
                n_tracks = len(self.tracks)
                print(f"[SCANNER] {pct:.0f}% ({frame_num}/{self.total_frames}) - {n_tracks} tracks found")

        cap.release()
        self.scan_frame = self.total_frames
        self.scan_complete = True
        n_tracks = len(self.tracks)
        print(f"[SCANNER] Complete! Found {n_tracks} tracks in {self.total_frames} frames.")


# ============================================================================
#  MAIN ANNOTATOR
# ============================================================================

class GroundTruthAnnotator:

    def __init__(self, source, confidence=0.15, model_path="yolov8s.pt",
                 record=False, output_dir=".", playback_speed=1.0,
                 frame_skip=SCAN_FRAME_SKIP):
        self.source = source
        self.confidence = confidence
        self.model_path = model_path
        self.record = record
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_skip = frame_skip

        # Video state (display thread only)
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

        # Background scanner
        self.scanner: Optional[BackgroundScanner] = None

        # Annotation
        self.annotation = AnnotationState()

        # Display
        self.show_trails = True
        self.show_zones = True
        self.info_panel_mode = 2  # Start on UNLABELED ONLY
        self.selected_track_id: Optional[int] = None
        self.last_frame_detections: List[Dict] = []
        self.panel_track_regions: List[Tuple[int, int, int, int, int]] = []

        # Status
        self.status_msg = ""
        self.status_time = 0

    def set_status(self, msg: str, duration: float = 2.0):
        self.status_msg = msg
        self.status_time = time.time() + duration

    @property
    def tracks(self) -> Dict[int, TrackData]:
        """Get tracks from scanner (thread-safe)."""
        if self.scanner:
            return self.scanner.get_tracks_snapshot()
        return {}

    def open_source(self):
        source = self.source
        if source.isdigit():
            source = int(source)
            self.is_video_file = False
        elif os.path.isfile(source):
            self.is_video_file = True
        else:
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

    def start_scanner(self):
        """Start the background YOLO scanner for video files."""
        if not self.is_video_file:
            return
        self.scanner = BackgroundScanner(
            video_path=self.source,
            model_path=self.model_path,
            confidence=self.confidence,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            frame_skip=self.frame_skip,
        )
        self.scanner.start()

    def read_frame_at(self, frame_num: int) -> Optional[np.ndarray]:
        """Read a raw frame at a specific position (no YOLO)."""
        if not self.cap:
            return None
        frame_num = max(0, min(frame_num, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame_num + 1  # 1-based
            return frame
        return None

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw detection boxes + overlays on a raw frame."""
        overlay = frame.copy()
        h, w = overlay.shape[:2]
        tracks = self.tracks

        # Zone lines
        if self.show_zones:
            near_y = int(0.584 * h)
            cv2.line(overlay, (0, near_y), (w, near_y), (0, 255, 255), 1)
            cv2.putText(overlay, "THRESHOLD 0.584", (5, near_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            far_y = int(0.382 * h)
            cv2.line(overlay, (0, far_y), (w, far_y), (100, 255, 100), 1)
            cv2.putText(overlay, "SPAWN_FAR 0.382", (5, far_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
            spawn_near_y = int(0.57 * h)
            cv2.line(overlay, (0, spawn_near_y), (w, spawn_near_y), (100, 100, 255), 1)
            cv2.putText(overlay, "SPAWN_NEAR 0.57", (5, spawn_near_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)

        # Get detections for current frame from scanner cache
        if self.scanner:
            self.last_frame_detections = self.scanner.get_nearest_detections(self.current_frame)
        else:
            self.last_frame_detections = []

        # Draw detections
        for det in self.last_frame_detections:
            tid = det['track_id']
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            track = tracks.get(tid)
            label = self.annotation.labels.get(tid)

            if label == "IN":
                color = COLOR_IN
                label_text = "IN"
            elif label == "OUT":
                color = COLOR_OUT
                label_text = "OUT"
            else:
                color = COLOR_UNLABELED
                label_text = "?"

            thickness = 3 if tid == self.selected_track_id else 2
            if tid == self.selected_track_id:
                color = COLOR_ACTIVE

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

            info_text = f"ID:{tid} {conf:.2f} [{label_text}]"
            ts = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(overlay, (x1, y1 - ts[1] - 8), (x1 + ts[0] + 4, y1), color, -1)
            cv2.putText(overlay, info_text, (x1 + 2, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            if track:
                area_text = f"a:{det['area']:.4f} y:{det['centroid'][1]:.3f}"
                cv2.putText(overlay, area_text, (x1, y2 + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                if track.y_trend != 0:
                    arrow_y = "v" if track.y_trend > 0 else "^"
                    arrow_a = "+" if track.area_trend > 0 else "-"
                    cv2.putText(overlay, f"y{arrow_y} a{arrow_a}", (x1, y2 + 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

            # Trail
            if self.show_trails and track and len(track.bbox_history) > 1:
                trail_pts = []
                for bx1, by1, bx2, by2 in track.bbox_history[-20:]:
                    trail_pts.append(((bx1 + bx2) // 2, (by1 + by2) // 2))
                for i in range(1, len(trail_pts)):
                    cv2.line(overlay, trail_pts[i-1], trail_pts[i], color, max(1, int(i / len(trail_pts) * 3)))

        # Input mode prompt
        if self.annotation.input_mode:
            mode_color = COLOR_IN if self.annotation.input_mode == "IN" else COLOR_OUT
            if self.annotation.input_mode == "DELETE":
                mode_color = (0, 128, 255)
            prompt = f"Enter Track ID for {self.annotation.input_mode}: {self.annotation.input_buffer}_"
            cv2.rectangle(overlay, (0, h - 40), (w, h), (0, 0, 0), -1)
            cv2.putText(overlay, prompt, (10, h - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

        # Status message
        if self.status_msg and time.time() < self.status_time:
            cv2.rectangle(overlay, (0, 0), (w, 30), (0, 0, 0), -1)
            cv2.putText(overlay, self.status_msg, (10, 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_HIGHLIGHT, 2)

        # Top info bar
        n_tracks = len(tracks)
        n_labeled = len(self.annotation.labels)
        bar_text = f"Frame: {self.current_frame}"
        if self.total_frames > 0:
            bar_text += f"/{self.total_frames} ({self.current_frame/self.total_frames*100:.1f}%)"
        bar_text += f" | Speed: {self.playback_speed:.1f}x"
        if self.paused:
            bar_text += " | PAUSED"
        bar_text += f" | Tracks: {n_tracks} | Labeled: {n_labeled}"

        cv2.rectangle(overlay, (0, 0), (w, 25), (40, 40, 40), -1)
        cv2.putText(overlay, bar_text, (5, 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1)

        # Video progress bar
        if self.total_frames > 0:
            bar_y = 25
            progress = self.current_frame / max(self.total_frames, 1)
            cv2.rectangle(overlay, (0, bar_y), (w, bar_y + 4), (60, 60, 60), -1)
            cv2.rectangle(overlay, (0, bar_y), (int(w * progress), bar_y + 4), COLOR_HIGHLIGHT, -1)

            # Scan progress bar (underneath, thinner)
            if self.scanner and not self.scanner.scan_complete:
                scan_bar_y = bar_y + 4
                cv2.rectangle(overlay, (0, scan_bar_y), (w, scan_bar_y + 2), (40, 40, 40), -1)
                cv2.rectangle(overlay, (0, scan_bar_y), (int(w * self.scanner.progress), scan_bar_y + 2),
                            COLOR_SCAN_BAR, -1)

        return overlay

    def draw_info_panel(self, frame: np.ndarray) -> np.ndarray:
        """Draw the side info panel showing tracks, labels, and help."""
        h = frame.shape[0]
        panel = np.zeros((h, INFO_PANEL_WIDTH, 3), dtype=np.uint8)
        panel[:] = COLOR_PANEL_BG
        self.panel_track_regions = []
        tracks = self.tracks

        y = 20
        line_h = 18

        # Scan status
        if self.scanner and not self.scanner.scan_complete:
            scan_text = f"SCANNING: {self.scanner.progress*100:.0f}% ({len(tracks)} tracks)"
            cv2.putText(panel, scan_text, (5, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_SCAN_BAR, 1)
            y += line_h + 2
        elif self.scanner and self.scanner.scan_complete:
            cv2.putText(panel, f"SCAN DONE: {len(tracks)} tracks", (5, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
            y += line_h + 2

        # Panel title
        modes = ["ACTIVE (near frame)", "ALL TRACKS", "UNLABELED ONLY"]
        title = modes[self.info_panel_mode % len(modes)]
        cv2.putText(panel, f"=== {title} ===", (5, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_HIGHLIGHT, 1)
        y += line_h + 3

        # Counters
        in_count = sum(1 for v in self.annotation.labels.values() if v == "IN")
        out_count = sum(1 for v in self.annotation.labels.values() if v == "OUT")
        unlabeled = len(tracks) - len(self.annotation.labels)
        cv2.putText(panel, f"IN: {in_count}  OUT: {out_count}  ?: {unlabeled}",
                   (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1)
        y += line_h + 3

        cv2.line(panel, (5, y), (INFO_PANEL_WIDTH - 5, y), (80, 80, 80), 1)
        y += 8

        # Build track list based on mode
        if self.info_panel_mode == 0:
            # Tracks visible near current frame (within 30 frames)
            cf = self.current_frame
            track_ids = sorted(
                [tid for tid, t in tracks.items()
                 if t.first_frame <= cf + 30 and t.last_frame >= cf - 30],
                key=lambda tid: tracks[tid].first_frame
            )
        elif self.info_panel_mode == 1:
            track_ids = sorted(tracks.keys(), key=lambda tid: tracks[tid].first_frame)
        else:
            track_ids = sorted(
                [tid for tid in tracks.keys() if tid not in self.annotation.labels],
                key=lambda tid: tracks[tid].first_frame
            )

        for tid in track_ids:
            if y > h - 150:
                remaining = len(track_ids) - track_ids.index(tid)
                cv2.putText(panel, f"... +{remaining} more (scroll N to see)",
                           (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_TEXT, 1)
                break

            track = tracks.get(tid)
            if not track:
                continue

            # Color
            label = self.annotation.labels.get(tid)
            if label:
                color = COLOR_IN if label == "IN" else COLOR_OUT
                label_str = f"[{label}]"
            else:
                color = COLOR_UNLABELED
                label_str = "[?]"

            # Background highlight for selected
            bg_y1 = y - 10
            bg_y2 = y + line_h * 2 + 4
            if tid == self.selected_track_id:
                cv2.rectangle(panel, (0, bg_y1), (INFO_PANEL_WIDTH, bg_y2), (60, 60, 60), -1)

            # Clickable region
            self.panel_track_regions.append((0, bg_y1, INFO_PANEL_WIDTH, bg_y2, tid))

            # ID line
            id_text = f"ID:{tid:3d} {label_str}"
            cv2.putText(panel, id_text, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            # Frame range
            frame_text = f"F{track.first_frame}-{track.last_frame}"
            cv2.putText(panel, frame_text, (160, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

            # Stats line
            y += line_h
            stats_text = f"conf:{track.max_confidence:.2f}"
            if track.centroid_history:
                first_y = track.centroid_history[0][1]
                last_y = track.centroid_history[-1][1]
                direction = "v" if last_y > first_y else "^"
                stats_text += f"  y:{first_y:.2f}{direction}{last_y:.2f}"
            cv2.putText(panel, stats_text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (180, 180, 180), 1)

            # Area info
            y += line_h
            if track.area_history:
                first_a = track.area_history[0]
                last_a = track.area_history[-1]
                a_direction = "+" if last_a > first_a else "-"
                area_text = f"area:{first_a:.4f}{a_direction}{last_a:.4f}  frames:{track.total_frames_visible}"
                cv2.putText(panel, area_text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (180, 180, 180), 1)

            y += line_h + 6

        # Help text at bottom
        help_y = h - 140
        cv2.line(panel, (5, help_y), (INFO_PANEL_WIDTH - 5, help_y), (80, 80, 80), 1)
        help_y += 15
        helps = [
            "CLICK ID = jump to track",
            "LEFT/RIGHT = 1 frame  |  <> = 10",
            "[] = 100 frames  |  H/E = start/end",
            "I = IN  |  O = OUT  |  N = next ?",
            "U = undo  |  D = delete label",
            "SPACE = pause  |  +/- = speed",
            "T = trails  |  Z = zones",
            "TAB = panel mode  |  S = save",
            "Q = quit & save"
        ]
        for line in helps:
            cv2.putText(panel, line, (5, help_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.33, (140, 140, 140), 1)
            help_y += 14

        return np.hstack([frame, panel])

    def handle_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Progress bar click (top of video area)
        if self.is_video_file and self.total_frames > 0 and 25 <= y <= 32 and x < self.frame_width:
            progress = x / self.frame_width
            target = int(progress * self.total_frames)
            self.paused = True
            self._seek_and_display(target)
            return

        # Panel click (right side)
        if x >= self.frame_width:
            panel_x = x - self.frame_width
            for px1, py1, px2, py2, tid in self.panel_track_regions:
                if px1 <= panel_x <= px2 and py1 <= y <= py2:
                    tracks = self.tracks
                    track = tracks.get(tid)
                    if track and self.is_video_file:
                        self.selected_track_id = tid
                        self.paused = True
                        self._seek_and_display(track.first_frame)
                        self.set_status(f"Track {tid} @ frame {track.first_frame}")
                    return

        # Detection box click
        if x < self.frame_width:
            for det in self.last_frame_detections:
                bx1, by1, bx2, by2 = det['bbox']
                if bx1 <= x <= bx2 and by1 <= y <= by2:
                    self.selected_track_id = det['track_id']
                    self.set_status(f"Selected track ID: {det['track_id']}")
                    return
            self.selected_track_id = None

    def handle_key(self, key: int):
        """Returns False to quit, 'redraw' to refresh display, True otherwise."""
        # Input mode (typing track ID)
        if self.annotation.input_mode:
            if key == 27:  # ESC
                self.annotation.input_mode = None
                self.annotation.input_buffer = ""
                self.set_status("Cancelled")
                return True
            elif key == 13 or key == 10:  # Enter
                self._apply_label()
                return "redraw"
            elif key == 8 or key == 127:  # Backspace
                self.annotation.input_buffer = self.annotation.input_buffer[:-1]
                return True
            elif 48 <= key <= 57:  # Digits 0-9
                self.annotation.input_buffer += chr(key)
                return True
            return True

        # Quit
        if key == ord('q') or key == 27:
            self.save_ground_truth()
            return False
        # Pause
        elif key == ord(' '):
            self.paused = not self.paused
            self.set_status("PAUSED" if self.paused else "PLAYING")
        # Label IN
        elif key == ord('i') or key == ord('I'):
            if self.selected_track_id is not None:
                self._label_track(self.selected_track_id, "IN")
                return "redraw"
            else:
                self.annotation.input_mode = "IN"
                self.annotation.input_buffer = ""
                self.set_status("Type track ID for IN, then Enter")
        # Label OUT
        elif key == ord('o') or key == ord('O'):
            if self.selected_track_id is not None:
                self._label_track(self.selected_track_id, "OUT")
                return "redraw"
            else:
                self.annotation.input_mode = "OUT"
                self.annotation.input_buffer = ""
                self.set_status("Type track ID for OUT, then Enter")
        # Undo
        elif key == ord('u') or key == ord('U'):
            self._undo_label()
            return "redraw"
        # Delete
        elif key == ord('d') or key == ord('D'):
            if self.selected_track_id is not None:
                self._delete_label(self.selected_track_id)
                return "redraw"
            else:
                self.annotation.input_mode = "DELETE"
                self.annotation.input_buffer = ""
                self.set_status("Type track ID to delete label, then Enter")
        # Save
        elif key == ord('s') or key == ord('S'):
            self.save_ground_truth()
            self.set_status("Ground truth saved!")
        # Trails
        elif key == ord('t') or key == ord('T'):
            self.show_trails = not self.show_trails
            self.set_status(f"Trails: {'ON' if self.show_trails else 'OFF'}")
            return "redraw"
        # Zones
        elif key == ord('z') or key == ord('Z'):
            self.show_zones = not self.show_zones
            self.set_status(f"Zones: {'ON' if self.show_zones else 'OFF'}")
            return "redraw"
        # Tab - cycle panel mode
        elif key == 9:
            self.info_panel_mode = (self.info_panel_mode + 1) % 3
            return "redraw"
        # Speed
        elif key == ord('+') or key == ord('='):
            self.playback_speed = min(8.0, self.playback_speed + 0.25)
            self.set_status(f"Speed: {self.playback_speed:.2f}x")
        elif key == ord('-') or key == ord('_'):
            self.playback_speed = max(0.1, self.playback_speed - 0.25)
            self.set_status(f"Speed: {self.playback_speed:.2f}x")
        # Navigation
        elif key == 81 or key == 2:  # LEFT arrow
            if self.is_video_file:
                self.paused = True
                self._seek_and_display(self.current_frame - 1)
        elif key == 83 or key == 3:  # RIGHT arrow
            if self.is_video_file:
                self.paused = True
                self._seek_and_display(self.current_frame + 1)
        elif key == ord(',') or key == ord('<'):  # 10 frames back
            if self.is_video_file:
                self.paused = True
                self._seek_and_display(self.current_frame - 10)
        elif key == ord('.') or key == ord('>'):  # 10 frames forward
            if self.is_video_file:
                self.paused = True
                self._seek_and_display(self.current_frame + 10)
        elif key == ord('['):  # 100 frames back
            if self.is_video_file:
                self.paused = True
                self._seek_and_display(self.current_frame - 100)
        elif key == ord(']'):  # 100 frames forward
            if self.is_video_file:
                self.paused = True
                self._seek_and_display(self.current_frame + 100)
        elif key == ord('h') or key == ord('H'):  # Home
            if self.is_video_file:
                self.paused = True
                self._seek_and_display(0)
        elif key == ord('e') or key == ord('E'):  # End
            if self.is_video_file:
                self.paused = True
                self._seek_and_display(self.total_frames - 1)
        # Next unlabeled
        elif key == ord('n') or key == ord('N'):
            self._jump_to_next_unlabeled()

        return True

    def _seek_and_display(self, frame_num: int):
        """Seek to frame and update display (fast - no YOLO)."""
        frame = self.read_frame_at(frame_num)
        if frame is not None:
            display = self.draw_overlay(frame)
            display = self.draw_info_panel(display)
            self._last_display = display
            cv2.imshow(WINDOW_NAME, display)

    def _apply_label(self):
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
        tracks = self.tracks
        if track_id not in tracks:
            self.set_status(f"Track {track_id} not found yet (scan may not have reached it)")
            return
        old_label = self.annotation.labels.get(track_id)
        self.annotation.labels[track_id] = direction
        self.annotation.label_history.append((track_id, old_label))
        unlabeled = len(tracks) - len(self.annotation.labels)
        self.set_status(f"Track {track_id} = {direction}  ({unlabeled} unlabeled remaining)")

    def _delete_label(self, track_id: int):
        if track_id in self.annotation.labels:
            old = self.annotation.labels.pop(track_id)
            self.annotation.label_history.append((track_id, old))
            self.set_status(f"Removed label from track {track_id}")
        else:
            self.set_status(f"Track {track_id} has no label")

    def _undo_label(self):
        if not self.annotation.label_history:
            self.set_status("Nothing to undo")
            return
        track_id, old_label = self.annotation.label_history.pop()
        if old_label is None:
            self.annotation.labels.pop(track_id, None)
            self.set_status(f"Undone: track {track_id} unlabeled")
        else:
            self.annotation.labels[track_id] = old_label
            self.set_status(f"Undone: track {track_id} = {old_label}")

    def _jump_to_next_unlabeled(self):
        if not self.is_video_file:
            self.set_status("Only works with video files")
            return
        tracks = self.tracks
        unlabeled = sorted(
            [tid for tid in tracks.keys() if tid not in self.annotation.labels],
            key=lambda tid: tracks[tid].first_frame
        )
        if not unlabeled:
            self.set_status("All tracks labeled!")
            return

        # Find next unlabeled after current frame
        next_track = None
        for tid in unlabeled:
            if tracks[tid].first_frame > self.current_frame:
                next_track = tid
                break
        if next_track is None:
            next_track = unlabeled[0]  # Wrap around

        track = tracks[next_track]
        self.selected_track_id = next_track
        self.paused = True
        self._seek_and_display(track.first_frame)
        self.set_status(f"Track {next_track} - {len(unlabeled)} unlabeled remaining")

    def save_ground_truth(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"ground_truth_{timestamp}.json"
        tracks = self.tracks

        if not tracks:
            print("No tracks to save yet (scanner may still be running)")
            return None

        track_data = {}
        for tid, track in tracks.items():
            track_data[str(tid)] = {
                "track_id": tid,
                "label": self.annotation.labels.get(tid),
                "first_frame": track.first_frame,
                "last_frame": track.last_frame,
                "total_frames_visible": track.total_frames_visible,
                "max_confidence": round(track.max_confidence, 4),
                "avg_confidence": round(track.avg_confidence, 4),
                "max_area": round(track.max_area, 6),
                "min_area": round(track.min_area, 6) if track.min_area < 999 else 0,
                "area_trend": round(track.area_trend, 6),
                "y_trend": round(track.y_trend, 6),
                "first_centroid_y": round(track.centroid_history[0][1], 4) if track.centroid_history else None,
                "last_centroid_y": round(track.centroid_history[-1][1], 4) if track.centroid_history else None,
                "first_area": round(track.area_history[0], 6) if track.area_history else None,
                "last_area": round(track.area_history[-1], 6) if track.area_history else None,
                "centroid_y_history": [round(p[1], 4) for p in track.centroid_history],
                "area_history": [round(a, 6) for a in track.area_history],
                "confidence_history": [round(c, 4) for c in track.confidence_history],
                "frame_numbers": track.frame_numbers,
            }

        in_tracks = {tid: d for tid, d in track_data.items() if d["label"] == "IN"}
        out_tracks = {tid: d for tid, d in track_data.items() if d["label"] == "OUT"}
        analysis = self._compute_analysis(in_tracks, out_tracks)

        output = {
            "metadata": {
                "source": str(self.source),
                "recording_path": self.recording_path,
                "timestamp": timestamp,
                "total_frames": self.total_frames,
                "fps": self.fps,
                "frame_size": [self.frame_width, self.frame_height],
                "yolo_confidence": self.confidence,
                "model": self.model_path,
                "scan_frame_skip": self.frame_skip,
            },
            "summary": {
                "total_tracks": len(tracks),
                "labeled_in": len(in_tracks),
                "labeled_out": len(out_tracks),
                "unlabeled": len(tracks) - len(in_tracks) - len(out_tracks),
            },
            "analysis": analysis,
            "tracks": track_data,
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\n{'='*60}")
        print(f"GROUND TRUTH SAVED: {output_path}")
        print(f"{'='*60}")
        print(f"Total tracks: {len(tracks)}")
        print(f"Labeled IN:   {len(in_tracks)}")
        print(f"Labeled OUT:  {len(out_tracks)}")
        print(f"Unlabeled:    {len(tracks) - len(in_tracks) - len(out_tracks)}")

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
        if not in_tracks and not out_tracks:
            return {"note": "No labeled tracks - label some for analysis"}

        analysis = {}

        in_confs = [t["max_confidence"] for t in in_tracks.values()]
        out_confs = [t["max_confidence"] for t in out_tracks.values()]
        all_confs = in_confs + out_confs
        if all_confs:
            analysis["confidence"] = {
                "min": round(min(all_confs), 4),
                "max": round(max(all_confs), 4),
                "avg": round(sum(all_confs) / len(all_confs), 4),
                "recommended_threshold": round(sorted(all_confs)[max(0, len(all_confs)//10)] * 0.9, 4),
            }

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
            analysis["area_ratios"]["in_median"] = round(sorted(in_area_ratios)[len(in_area_ratios)//2], 4)
        if out_area_ratios:
            analysis["area_ratios"]["out_median"] = round(sorted(out_area_ratios)[len(out_area_ratios)//2], 4)

        in_first_y = [t["first_centroid_y"] for t in in_tracks.values() if t["first_centroid_y"] is not None]
        out_first_y = [t["first_centroid_y"] for t in out_tracks.values() if t["first_centroid_y"] is not None]
        in_last_y = [t["last_centroid_y"] for t in in_tracks.values() if t["last_centroid_y"] is not None]
        out_last_y = [t["last_centroid_y"] for t in out_tracks.values() if t["last_centroid_y"] is not None]

        analysis["y_positions"] = {}
        if in_first_y:
            analysis["y_positions"]["in_spawn_avg"] = round(sum(in_first_y) / len(in_first_y), 4)
        if out_first_y:
            analysis["y_positions"]["out_spawn_avg"] = round(sum(out_first_y) / len(out_first_y), 4)
        if in_first_y and out_first_y:
            analysis["y_positions"]["recommended_spawn_far"] = round(
                min(sum(in_first_y)/len(in_first_y), sum(out_first_y)/len(out_first_y)) * 0.9, 4)
            analysis["y_positions"]["recommended_spawn_near"] = round(
                max(sum(in_first_y)/len(in_first_y), sum(out_first_y)/len(out_first_y)) * 1.1, 4)

        if in_last_y and out_last_y:
            in_end = sum(in_last_y) / len(in_last_y)
            out_end = sum(out_last_y) / len(out_last_y)
            analysis["crossing"] = {
                "in_endpoint_y": round(in_end, 4),
                "out_endpoint_y": round(out_end, 4),
                "recommended_threshold": round((in_end + out_end) / 2, 4),
            }

        all_durations = [t["total_frames_visible"] for t in {**in_tracks, **out_tracks}.values()]
        if all_durations:
            min_dur = min(all_durations)
            analysis["track_duration"] = {
                "min_frames": min_dur,
                "avg_frames": round(sum(all_durations) / len(all_durations), 1),
                "recommended_min_ms": round(min_dur / self.fps * 1000 * 0.8, 0),
            }

        return analysis

    def run(self):
        """Main loop: display video, handle input, while scanner runs in background."""
        self.open_source()

        # Start background scanner for video files
        if self.is_video_file:
            self.paused = True
            self.start_scanner()

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.handle_mouse)

        print("\n" + "="*60)
        print("GROUND TRUTH ANNOTATOR (Background Scan)")
        print("="*60)
        if self.is_video_file:
            print(f"Scanner running in background (every {self.frame_skip} frames)")
            print("Tracks will appear as scanner discovers them")
        print("\nNAVIGATION:")
        print("  SPACE    = Pause/Play")
        print("  LEFT/RIGHT = Step 1 frame")
        print("  < >      = Jump 10 frames")
        print("  [ ]      = Jump 100 frames")
        print("  H/E      = Go to start/end")
        print("\nLABELING:")
        print("  Click ID in panel = Jump to track")
        print("  Click box + I/O   = Label as IN/OUT")
        print("  N                 = Next unlabeled track")
        print("  U/D               = Undo / Delete label")
        print("\nOTHER: S=save T=trails Z=zones TAB=panel Q=quit")
        print("="*60 + "\n")

        self._last_display = None
        last_panel_refresh = 0

        try:
            # Show first frame immediately (no YOLO needed)
            frame = self.read_frame_at(0)
            if frame is not None:
                display = self.draw_overlay(frame)
                display = self.draw_info_panel(display)
                self._last_display = display
                cv2.imshow(WINDOW_NAME, display)

            while True:
                need_redraw = False

                if not self.paused:
                    # Playing - read sequential frames (fast, no YOLO)
                    ret, frame = self.cap.read()
                    if not ret:
                        if self.is_video_file:
                            self.set_status("End of video - press Q to save")
                            self.paused = True
                        else:
                            continue
                    else:
                        self.current_frame += 1
                        if self.video_writer:
                            self.video_writer.write(frame)
                        display = self.draw_overlay(frame)
                        display = self.draw_info_panel(display)
                        self._last_display = display
                else:
                    # Paused - periodically refresh panel to show scanner progress
                    now = time.time()
                    if self.scanner and not self.scanner.scan_complete and now - last_panel_refresh > 0.5:
                        need_redraw = True
                        last_panel_refresh = now

                if need_redraw:
                    # Re-read current frame and redraw
                    frame = self.read_frame_at(max(0, self.current_frame - 1))
                    if frame is not None:
                        display = self.draw_overlay(frame)
                        display = self.draw_info_panel(display)
                        self._last_display = display

                if self._last_display is not None:
                    cv2.imshow(WINDOW_NAME, self._last_display)

                wait_ms = 30 if self.paused else max(1, int((1000 / self.fps) / self.playback_speed))
                key = cv2.waitKey(wait_ms) & 0xFF

                if key != 255:
                    result = self.handle_key(key)
                    if result is False:
                        return
                    elif result == "redraw":
                        frame = self.read_frame_at(max(0, self.current_frame - 1))
                        if frame is not None:
                            display = self.draw_overlay(frame)
                            display = self.draw_info_panel(display)
                            self._last_display = display

        finally:
            if self.scanner:
                self.scanner.stop()
            if self.video_writer:
                self.video_writer.release()
                print(f"Recording saved to: {self.recording_path}")
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            if self.annotation.labels:
                self.save_ground_truth()


def main():
    parser = argparse.ArgumentParser(description='Ground Truth Annotator (Background Scan)')
    parser.add_argument('--source', type=str, default='0',
                        help='Video source: file path, camera index, or URL')
    parser.add_argument('--confidence', type=float, default=DETECTION_CONFIDENCE,
                        help='YOLO confidence threshold (default: 0.15)')
    parser.add_argument('--model', type=str, default=MODEL_PATH,
                        help='YOLO model path')
    parser.add_argument('--record', action='store_true',
                        help='Record video from camera source')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory for output files')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Initial playback speed')
    parser.add_argument('--frame-skip', type=int, default=SCAN_FRAME_SKIP,
                        help=f'Scan every Nth frame (default: {SCAN_FRAME_SKIP})')

    args = parser.parse_args()

    annotator = GroundTruthAnnotator(
        source=args.source,
        confidence=args.confidence,
        model_path=args.model,
        record=args.record,
        output_dir=args.output_dir,
        playback_speed=args.speed,
        frame_skip=args.frame_skip,
    )
    annotator.run()


if __name__ == "__main__":
    main()

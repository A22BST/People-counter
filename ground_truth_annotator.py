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
    P           - Toggle full path overlay (all labeled tracks)
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
import math
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
INFO_PANEL_WIDTH = 380

# ── Color Palette (BGR) ──────────────────────────────────────────────────────
# Softer, higher-contrast palette designed for long annotation sessions.
COLOR_IN = (80, 210, 80)           # Muted green
COLOR_OUT = (80, 80, 220)          # Muted red-blue
COLOR_UNLABELED = (0, 190, 220)    # Amber/yellow
COLOR_ACTIVE = (0, 245, 255)       # Bright yellow highlight
COLOR_LOST = (110, 110, 110)
COLOR_PANEL_BG = (25, 25, 28)      # Near-black
COLOR_PANEL_CARD = (40, 40, 45)    # Card background
COLOR_PANEL_CARD_SEL = (55, 55, 65)  # Selected card
COLOR_TEXT = (210, 210, 210)
COLOR_TEXT_DIM = (130, 130, 135)
COLOR_TEXT_BRIGHT = (245, 245, 245)
COLOR_HIGHLIGHT = (0, 180, 255)    # Orange accent
COLOR_SCAN_BAR = (200, 200, 0)     # Cyan
COLOR_BAR_BG = (45, 45, 48)
COLOR_DIVIDER = (65, 65, 70)
COLOR_SUCCESS = (80, 200, 80)
COLOR_WARN = (0, 160, 230)

# Font helpers
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL = 0.38
FONT_NORMAL = 0.44
FONT_LARGE = 0.52
FONT_XLARGE = 0.62

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
                 frame_skip=SCAN_FRAME_SKIP, load_file=None):
        self.source = source
        self.confidence = confidence
        self.model_path = model_path
        self.record = record
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_skip = frame_skip
        self.load_file = load_file

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
        self.show_full_paths = False  # Toggle with P - shows ALL labeled paths
        self.info_panel_mode = 2  # Start on UNLABELED ONLY
        self.selected_track_id: Optional[int] = None
        self.last_frame_detections: List[Dict] = []
        self.panel_track_regions: List[Tuple[int, int, int, int, int]] = []

        # Auto-save
        self.autosave_interval = 60  # seconds
        self.last_autosave = time.time()
        self.save_count = 0

        # Status
        self.status_msg = ""
        self.status_time = 0

    def set_status(self, msg: str, duration: float = 2.0):
        self.status_msg = msg
        self.status_time = time.time() + duration

    def load_annotations(self, filepath: str):
        """Load labels from a previous ground truth JSON file."""
        try:
            with open(filepath) as f:
                data = json.load(f)
            tracks = data.get("tracks", {})
            loaded = 0
            for tid_str, tdata in tracks.items():
                label = tdata.get("label")
                if label in ("IN", "OUT"):
                    self.annotation.labels[int(tid_str)] = label
                    loaded += 1
            print(f"[LOAD] Loaded {loaded} labels from {filepath}")
            print(f"[LOAD]   IN: {sum(1 for v in self.annotation.labels.values() if v == 'IN')}")
            print(f"[LOAD]   OUT: {sum(1 for v in self.annotation.labels.values() if v == 'OUT')}")
            self.set_status(f"Loaded {loaded} labels from previous session", 5.0)
        except Exception as e:
            print(f"[LOAD] ERROR loading {filepath}: {e}")
            self.set_status(f"Error loading: {e}", 5.0)

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

        # ── Zone lines (subtle, dashed-look) ─────────────────────────
        if self.show_zones:
            for y_norm, label, color in [
                (0.382, "SPAWN FAR",   (120, 220, 120)),
                (0.57,  "SPAWN NEAR",  (120, 120, 220)),
                (0.584, "THRESHOLD",   (0, 220, 220)),
            ]:
                py = int(y_norm * h)
                # Dashed line effect
                dash_len = 12
                for x_start in range(0, w, dash_len * 2):
                    cv2.line(overlay, (x_start, py), (min(x_start + dash_len, w), py), color, 1)
                # Small label on right side with background
                txt = f"{label} {y_norm}"
                ts = cv2.getTextSize(txt, FONT, FONT_SMALL, 1)[0]
                tx = w - ts[0] - 8
                cv2.rectangle(overlay, (tx - 4, py - ts[1] - 4), (w, py + 4), (0, 0, 0), -1)
                cv2.putText(overlay, txt, (tx, py), FONT, FONT_SMALL, color, 1, cv2.LINE_AA)

        # ── Detections from scanner ──────────────────────────────────
        if self.scanner:
            self.last_frame_detections = self.scanner.get_nearest_detections(self.current_frame)
        else:
            self.last_frame_detections = []

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

            is_selected = (tid == self.selected_track_id)
            box_color = COLOR_ACTIVE if is_selected else color
            thickness = 2

            # Box with corner accents instead of full rectangle for cleaner look
            cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, thickness)
            if is_selected:
                corner_len = min(15, (x2 - x1) // 3, (y2 - y1) // 3)
                for cx, cy, dx, dy in [
                    (x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)
                ]:
                    cv2.line(overlay, (cx, cy), (cx + dx * corner_len, cy), COLOR_ACTIVE, 3)
                    cv2.line(overlay, (cx, cy), (cx, cy + dy * corner_len), COLOR_ACTIVE, 3)

            # ── Top label pill ──
            id_str = f" {tid} "
            label_str = f" {label_text} "
            conf_str = f" {conf:.0%} "

            # ID pill
            id_sz = cv2.getTextSize(id_str, FONT, FONT_NORMAL, 1)[0]
            lab_sz = cv2.getTextSize(label_str, FONT, FONT_NORMAL, 1)[0]
            conf_sz = cv2.getTextSize(conf_str, FONT, FONT_SMALL, 1)[0]

            pill_h = id_sz[1] + 10
            pill_y = y1 - pill_h - 2
            if pill_y < 0:
                pill_y = y2 + 2  # Flip below if too close to top

            # ID background
            cv2.rectangle(overlay, (x1, pill_y), (x1 + id_sz[0] + 2, pill_y + pill_h), box_color, -1)
            cv2.putText(overlay, id_str, (x1 + 1, pill_y + pill_h - 4),
                       FONT, FONT_NORMAL, (0, 0, 0), 1, cv2.LINE_AA)

            # Label pill (right of ID)
            lx = x1 + id_sz[0] + 3
            lab_bg = COLOR_IN if label_text == "IN" else (COLOR_OUT if label_text == "OUT" else (80, 80, 80))
            cv2.rectangle(overlay, (lx, pill_y), (lx + lab_sz[0] + 2, pill_y + pill_h), lab_bg, -1)
            cv2.putText(overlay, label_str, (lx + 1, pill_y + pill_h - 4),
                       FONT, FONT_NORMAL, (255, 255, 255), 1, cv2.LINE_AA)

            # Confidence (smaller, dimmer, right of label)
            cx_pos = lx + lab_sz[0] + 5
            cv2.putText(overlay, conf_str, (cx_pos, pill_y + pill_h - 4),
                       FONT, FONT_SMALL, (180, 180, 180), 1, cv2.LINE_AA)

            # ── Bottom info (area + direction arrows) ──
            if track:
                # Direction arrow below box
                if track.y_trend != 0:
                    arrow = chr(0x2193) if track.y_trend > 0 else chr(0x2191)  # Unicode arrows may not render; use text
                    arrow_y_txt = "v" if track.y_trend > 0 else "^"
                    arrow_a_txt = "+" if track.area_trend > 0 else "-"
                    info_txt = f"y{arrow_y_txt} a{arrow_a_txt}"
                    cv2.putText(overlay, info_txt, (x1, y2 + 16),
                               FONT, FONT_SMALL, (*box_color[:3],), 1, cv2.LINE_AA)

            # ── Trail ──
            if self.show_trails and track and len(track.bbox_history) > 1:
                trail_pts = []
                for bx1, by1, bx2, by2 in track.bbox_history[-25:]:
                    trail_pts.append(((bx1 + bx2) // 2, (by1 + by2) // 2))
                for i in range(1, len(trail_pts)):
                    alpha = i / len(trail_pts)
                    t = max(1, int(alpha * 2.5))
                    # Fade color
                    fade = tuple(int(c * alpha) for c in box_color)
                    cv2.line(overlay, trail_pts[i-1], trail_pts[i], fade, t, cv2.LINE_AA)

        # ── Full path overlay for ALL labeled tracks (P key) ─────────
        if self.show_full_paths:
            for tid, label_dir in self.annotation.labels.items():
                track = tracks.get(tid)
                if not track or len(track.centroid_history) < 2:
                    continue
                path_color = COLOR_IN if label_dir == "IN" else COLOR_OUT
                pts = []
                for cx_norm, cy_norm in track.centroid_history:
                    pts.append((int(cx_norm * w), int(cy_norm * h)))
                for i in range(1, len(pts)):
                    alpha = i / len(pts)
                    t = max(1, int(alpha * 3))
                    fade = tuple(int(c * (0.3 + 0.7 * alpha)) for c in path_color)
                    cv2.line(overlay, pts[i-1], pts[i], fade, t, cv2.LINE_AA)
                cv2.circle(overlay, pts[0], 5, path_color, -1, cv2.LINE_AA)
                cv2.putText(overlay, str(tid), (pts[0][0] + 7, pts[0][1] - 3),
                           FONT, FONT_SMALL, path_color, 1, cv2.LINE_AA)
                cv2.drawMarker(overlay, pts[-1], path_color, cv2.MARKER_CROSS, 8, 2)

        # ── Input mode prompt (bottom bar) ───────────────────────────
        if self.annotation.input_mode:
            mode_color = COLOR_IN if self.annotation.input_mode == "IN" else COLOR_OUT
            if self.annotation.input_mode == "DELETE":
                mode_color = COLOR_WARN
            # Semi-transparent bar
            bar_overlay = overlay.copy()
            cv2.rectangle(bar_overlay, (0, h - 44), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(bar_overlay, 0.85, overlay, 0.15, 0, overlay)
            prompt = f"  Enter Track ID for {self.annotation.input_mode}: {self.annotation.input_buffer}_"
            cv2.putText(overlay, prompt, (8, h - 14),
                       FONT, FONT_LARGE, mode_color, 2, cv2.LINE_AA)

        # ── Status message (top toast) ───────────────────────────────
        if self.status_msg and time.time() < self.status_time:
            # Semi-transparent background
            toast_overlay = overlay.copy()
            cv2.rectangle(toast_overlay, (0, 0), (w, 32), (0, 0, 0), -1)
            cv2.addWeighted(toast_overlay, 0.8, overlay, 0.2, 0, overlay)
            cv2.putText(overlay, f"  {self.status_msg}", (6, 22),
                       FONT, FONT_LARGE, COLOR_HIGHLIGHT, 1, cv2.LINE_AA)

        # ── Top info bar ─────────────────────────────────────────────
        n_tracks = len(tracks)
        n_labeled = len(self.annotation.labels)
        in_count = sum(1 for v in self.annotation.labels.values() if v == "IN")
        out_count = sum(1 for v in self.annotation.labels.values() if v == "OUT")

        bar_h = 28
        # Semi-transparent bar background
        bar_ov = overlay.copy()
        cv2.rectangle(bar_ov, (0, 0), (w, bar_h), COLOR_BAR_BG, -1)
        cv2.addWeighted(bar_ov, 0.75, overlay, 0.25, 0, overlay)

        # Frame info (left)
        frame_txt = f"Frame {self.current_frame}"
        if self.total_frames > 0:
            pct = self.current_frame / self.total_frames * 100
            frame_txt += f" / {self.total_frames}  ({pct:.0f}%)"
        cv2.putText(overlay, frame_txt, (8, 19), FONT, FONT_NORMAL, COLOR_TEXT, 1, cv2.LINE_AA)

        # Speed + pause (center)
        speed_txt = f"{self.playback_speed:.1f}x"
        if self.paused:
            speed_txt = "II PAUSED"
        speed_sz = cv2.getTextSize(speed_txt, FONT, FONT_NORMAL, 1)[0]
        cv2.putText(overlay, speed_txt, (w // 2 - speed_sz[0] // 2, 19),
                   FONT, FONT_NORMAL,
                   COLOR_WARN if self.paused else COLOR_TEXT_DIM, 1, cv2.LINE_AA)

        # Counts (right)
        counts_txt = f"IN:{in_count}  OUT:{out_count}  ?:{n_tracks - n_labeled}"
        counts_sz = cv2.getTextSize(counts_txt, FONT, FONT_NORMAL, 1)[0]
        cv2.putText(overlay, counts_txt, (w - counts_sz[0] - 8, 19),
                   FONT, FONT_NORMAL, COLOR_TEXT, 1, cv2.LINE_AA)

        # ── Progress bar ─────────────────────────────────────────────
        if self.total_frames > 0:
            bar_y = bar_h
            progress = self.current_frame / max(self.total_frames, 1)
            cv2.rectangle(overlay, (0, bar_y), (w, bar_y + 3), (50, 50, 53), -1)
            cv2.rectangle(overlay, (0, bar_y), (int(w * progress), bar_y + 3), COLOR_HIGHLIGHT, -1)
            # Playhead dot
            px = int(w * progress)
            cv2.circle(overlay, (px, bar_y + 1), 4, COLOR_HIGHLIGHT, -1, cv2.LINE_AA)

            # Scan progress bar (underneath)
            if self.scanner and not self.scanner.scan_complete:
                scan_y = bar_y + 3
                cv2.rectangle(overlay, (0, scan_y), (w, scan_y + 2), (35, 35, 38), -1)
                cv2.rectangle(overlay, (0, scan_y), (int(w * self.scanner.progress), scan_y + 2),
                            COLOR_SCAN_BAR, -1)

        return overlay

    def draw_info_panel(self, frame: np.ndarray) -> np.ndarray:
        """Draw the side info panel showing tracks, labels, and help."""
        h = frame.shape[0]
        pw = INFO_PANEL_WIDTH
        panel = np.zeros((h, pw, 3), dtype=np.uint8)
        panel[:] = COLOR_PANEL_BG
        self.panel_track_regions = []
        tracks = self.tracks

        y = 14
        pad = 12  # left padding

        # ── Header section ───────────────────────────────────────────
        # Scan status badge
        if self.scanner and not self.scanner.scan_complete:
            pct = self.scanner.progress * 100
            badge_txt = f"  SCANNING {pct:.0f}%  "
            bsz = cv2.getTextSize(badge_txt, FONT, FONT_NORMAL, 1)[0]
            # Animated-feel progress background
            prog_w = int((pw - 2 * pad) * self.scanner.progress)
            cv2.rectangle(panel, (pad, y - 2), (pad + prog_w, y + bsz[1] + 6), (35, 70, 70), -1)
            cv2.rectangle(panel, (pad, y - 2), (pw - pad, y + bsz[1] + 6), COLOR_DIVIDER, 1)
            cv2.putText(panel, badge_txt, (pad + 4, y + bsz[1] + 1),
                       FONT, FONT_NORMAL, COLOR_SCAN_BAR, 1, cv2.LINE_AA)
            trk_cnt = f"{len(tracks)} tracks"
            ts2 = cv2.getTextSize(trk_cnt, FONT, FONT_SMALL, 1)[0]
            cv2.putText(panel, trk_cnt, (pw - pad - ts2[0], y + bsz[1] + 1),
                       FONT, FONT_SMALL, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
            y += bsz[1] + 14
        elif self.scanner and self.scanner.scan_complete:
            done_txt = f"  DONE  {len(tracks)} tracks  "
            dsz = cv2.getTextSize(done_txt, FONT, FONT_NORMAL, 1)[0]
            cv2.rectangle(panel, (pad, y - 2), (pad + dsz[0] + 8, y + dsz[1] + 6), (30, 55, 30), -1)
            cv2.putText(panel, done_txt, (pad + 4, y + dsz[1] + 1),
                       FONT, FONT_NORMAL, COLOR_SUCCESS, 1, cv2.LINE_AA)
            y += dsz[1] + 14

        # ── Counts row ───────────────────────────────────────────────
        in_count = sum(1 for v in self.annotation.labels.values() if v == "IN")
        out_count = sum(1 for v in self.annotation.labels.values() if v == "OUT")
        unlabeled = len(tracks) - len(self.annotation.labels)

        # Draw colored count pills
        cx = pad
        for count_val, count_label, bg_color, fg_color in [
            (in_count,  "IN",  (30, 65, 30),  COLOR_IN),
            (out_count, "OUT", (55, 30, 30),  COLOR_OUT),
            (unlabeled, "?",   (55, 55, 30),  COLOR_UNLABELED),
        ]:
            pill_txt = f" {count_label}:{count_val} "
            psz = cv2.getTextSize(pill_txt, FONT, FONT_LARGE, 1)[0]
            pill_w = psz[0] + 6
            pill_h = psz[1] + 10
            cv2.rectangle(panel, (cx, y), (cx + pill_w, y + pill_h), bg_color, -1)
            cv2.rectangle(panel, (cx, y), (cx + pill_w, y + pill_h), fg_color, 1)
            cv2.putText(panel, pill_txt, (cx + 3, y + pill_h - 4),
                       FONT, FONT_LARGE, fg_color, 1, cv2.LINE_AA)
            cx += pill_w + 6
        y += pill_h + 10

        # ── Mode selector tabs ───────────────────────────────────────
        modes = ["NEARBY", "ALL", "UNLABELED"]
        tab_x = pad
        for i, mode_name in enumerate(modes):
            is_active = (self.info_panel_mode % len(modes) == i)
            msz = cv2.getTextSize(mode_name, FONT, FONT_SMALL, 1)[0]
            tab_w = msz[0] + 12
            if is_active:
                cv2.rectangle(panel, (tab_x, y), (tab_x + tab_w, y + 20), COLOR_HIGHLIGHT, -1)
                cv2.putText(panel, mode_name, (tab_x + 6, y + 15),
                           FONT, FONT_SMALL, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                cv2.rectangle(panel, (tab_x, y), (tab_x + tab_w, y + 20), COLOR_DIVIDER, 1)
                cv2.putText(panel, mode_name, (tab_x + 6, y + 15),
                           FONT, FONT_SMALL, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
            tab_x += tab_w + 4
        y += 28

        # ── Divider ──────────────────────────────────────────────────
        cv2.line(panel, (pad, y), (pw - pad, y), COLOR_DIVIDER, 1)
        y += 8

        # ── Track list ───────────────────────────────────────────────
        # Build track list based on mode
        if self.info_panel_mode == 0:
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

        card_h = 44  # height per track card
        help_area_h = 135
        max_y = h - help_area_h

        for idx, tid in enumerate(track_ids):
            if y + card_h > max_y:
                remaining = len(track_ids) - idx
                cv2.putText(panel, f"  + {remaining} more  (press N to navigate)",
                           (pad, y + 14), FONT, FONT_SMALL, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
                y += 20
                break

            track = tracks.get(tid)
            if not track:
                continue

            label = self.annotation.labels.get(tid)
            is_selected = (tid == self.selected_track_id)

            # ── Card background ──
            card_y1 = y
            card_y2 = y + card_h
            bg = COLOR_PANEL_CARD_SEL if is_selected else COLOR_PANEL_CARD
            cv2.rectangle(panel, (pad - 2, card_y1), (pw - pad + 2, card_y2), bg, -1)

            # Left color strip
            if label == "IN":
                strip_color = COLOR_IN
            elif label == "OUT":
                strip_color = COLOR_OUT
            else:
                strip_color = COLOR_UNLABELED
            cv2.rectangle(panel, (pad - 2, card_y1), (pad + 2, card_y2), strip_color, -1)

            if is_selected:
                cv2.rectangle(panel, (pad - 2, card_y1), (pw - pad + 2, card_y2), COLOR_ACTIVE, 1)

            # Clickable region
            self.panel_track_regions.append((0, card_y1, pw, card_y2, tid))

            # ── Row 1: ID + label + frame range ──
            row1_y = card_y1 + 16
            # ID
            id_txt = f"ID {tid}"
            cv2.putText(panel, id_txt, (pad + 8, row1_y),
                       FONT, FONT_NORMAL, COLOR_TEXT_BRIGHT, 1, cv2.LINE_AA)

            # Label badge
            if label:
                lb_txt = f" {label} "
                lb_sz = cv2.getTextSize(lb_txt, FONT, FONT_SMALL, 1)[0]
                lb_x = pad + 65
                lb_bg = COLOR_IN if label == "IN" else COLOR_OUT
                cv2.rectangle(panel, (lb_x, row1_y - lb_sz[1] - 2), (lb_x + lb_sz[0] + 4, row1_y + 3), lb_bg, -1)
                cv2.putText(panel, lb_txt, (lb_x + 2, row1_y),
                           FONT, FONT_SMALL, (255, 255, 255), 1, cv2.LINE_AA)

            # Frame range (right-aligned)
            fr_txt = f"F{track.first_frame}-{track.last_frame}"
            fr_sz = cv2.getTextSize(fr_txt, FONT, FONT_SMALL, 1)[0]
            cv2.putText(panel, fr_txt, (pw - pad - fr_sz[0], row1_y),
                       FONT, FONT_SMALL, COLOR_TEXT_DIM, 1, cv2.LINE_AA)

            # ── Row 2: stats ──
            row2_y = card_y1 + 34
            stats_parts = []
            stats_parts.append(f"c:{track.max_confidence:.0%}")
            if track.centroid_history:
                fy = track.centroid_history[0][1]
                ly = track.centroid_history[-1][1]
                arrow = "v" if ly > fy else "^"
                stats_parts.append(f"y:{fy:.2f}{arrow}{ly:.2f}")
            stats_parts.append(f"{track.total_frames_visible}f")
            stats_line = "   ".join(stats_parts)
            cv2.putText(panel, stats_line, (pad + 8, row2_y),
                       FONT, FONT_SMALL, COLOR_TEXT_DIM, 1, cv2.LINE_AA)

            y = card_y2 + 3  # gap between cards

        # ── Help section (bottom) ────────────────────────────────────
        help_y = h - help_area_h
        cv2.line(panel, (pad, help_y), (pw - pad, help_y), COLOR_DIVIDER, 1)
        help_y += 6

        # Help title
        cv2.putText(panel, "KEYBOARD SHORTCUTS", (pad, help_y + 12),
                   FONT, FONT_SMALL, COLOR_HIGHLIGHT, 1, cv2.LINE_AA)
        help_y += 22

        helps = [
            ("Navigate",   "<  >  [  ]  H  E"),
            ("Label",      "I = IN   O = OUT"),
            ("Select",     "Click box or panel"),
            ("Browse",     "N = next unlabeled"),
            ("Edit",       "U = undo  D = delete"),
            ("Playback",   "SPACE  +/-  speed"),
            ("Display",    "T trail  Z zone  P path"),
            ("Other",      "TAB mode  S save  Q quit"),
        ]
        for label_txt, keys_txt in helps:
            cv2.putText(panel, label_txt, (pad + 2, help_y),
                       FONT, FONT_SMALL, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
            cv2.putText(panel, keys_txt, (pad + 72, help_y),
                       FONT, FONT_SMALL, COLOR_TEXT, 1, cv2.LINE_AA)
            help_y += 14

        return np.hstack([frame, panel])

    def handle_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Progress bar click (top of video area)
        if self.is_video_file and self.total_frames > 0 and 24 <= y <= 36 and x < self.frame_width:
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
        # Full paths
        elif key == ord('p') or key == ord('P'):
            self.show_full_paths = not self.show_full_paths
            self.set_status(f"Full paths: {'ON' if self.show_full_paths else 'OFF'} ({len(self.annotation.labels)} labeled)")
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
            # Compute per-step velocity if we have enough path points
            velocities = []
            if len(track.centroid_history) >= 2 and len(track.frame_numbers) >= 2:
                for i in range(1, len(track.centroid_history)):
                    dx = track.centroid_history[i][0] - track.centroid_history[i-1][0]
                    dy = track.centroid_history[i][1] - track.centroid_history[i-1][1]
                    dt = track.frame_numbers[i] - track.frame_numbers[i-1]
                    if dt > 0:
                        velocities.append({
                            "vx": round(dx / dt, 6),
                            "vy": round(dy / dt, 6),
                            "speed": round((dx**2 + dy**2)**0.5 / dt, 6),
                        })

            # Compute path length (total distance traveled)
            path_length = 0.0
            if len(track.centroid_history) >= 2:
                for i in range(1, len(track.centroid_history)):
                    dx = track.centroid_history[i][0] - track.centroid_history[i-1][0]
                    dy = track.centroid_history[i][1] - track.centroid_history[i-1][1]
                    path_length += (dx**2 + dy**2)**0.5

            # Straight-line displacement
            displacement = 0.0
            if len(track.centroid_history) >= 2:
                dx = track.centroid_history[-1][0] - track.centroid_history[0][0]
                dy = track.centroid_history[-1][1] - track.centroid_history[0][1]
                displacement = (dx**2 + dy**2)**0.5

            # Linearity ratio (1.0 = perfectly straight line)
            linearity = round(displacement / path_length, 4) if path_length > 0 else 0.0

            # Direction angle (degrees, 0=right, 90=down, etc.)
            direction_angle = None
            if len(track.centroid_history) >= 2:
                dx = track.centroid_history[-1][0] - track.centroid_history[0][0]
                dy = track.centroid_history[-1][1] - track.centroid_history[0][1]
                direction_angle = round(math.degrees(math.atan2(dy, dx)), 2)

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
                "first_centroid_x": round(track.centroid_history[0][0], 4) if track.centroid_history else None,
                "last_centroid_x": round(track.centroid_history[-1][0], 4) if track.centroid_history else None,
                "first_area": round(track.area_history[0], 6) if track.area_history else None,
                "last_area": round(track.area_history[-1], 6) if track.area_history else None,
                # Full 2D path (normalized 0-1)
                "path": [{"x": round(p[0], 4), "y": round(p[1], 4)} for p in track.centroid_history],
                "centroid_x_history": [round(p[0], 4) for p in track.centroid_history],
                "centroid_y_history": [round(p[1], 4) for p in track.centroid_history],
                "bbox_history": [list(b) for b in track.bbox_history],
                "area_history": [round(a, 6) for a in track.area_history],
                "confidence_history": [round(c, 4) for c in track.confidence_history],
                "frame_numbers": track.frame_numbers,
                # Path metrics
                "path_length": round(path_length, 6),
                "displacement": round(displacement, 6),
                "linearity": linearity,
                "direction_angle": direction_angle,
                "velocities": velocities,
                "avg_speed": round(sum(v["speed"] for v in velocities) / len(velocities), 6) if velocities else 0,
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

        # Load previous annotations if specified
        if self.load_file:
            self.load_annotations(self.load_file)

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
        print("\nOTHER: S=save T=trails P=paths Z=zones TAB=panel Q=quit")
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

                # Auto-save periodically
                now = time.time()
                if self.annotation.labels and now - self.last_autosave > self.autosave_interval:
                    self.save_ground_truth()
                    self.last_autosave = now
                    self.set_status(f"Auto-saved ({len(self.annotation.labels)} labels)", 2.0)

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
    parser.add_argument('--load', type=str, default=None,
                        help='Load labels from a previous ground_truth_*.json file')

    args = parser.parse_args()

    annotator = GroundTruthAnnotator(
        source=args.source,
        confidence=args.confidence,
        model_path=args.model,
        record=args.record,
        output_dir=args.output_dir,
        playback_speed=args.speed,
        frame_skip=args.frame_skip,
        load_file=args.load,
    )
    annotator.run()


if __name__ == "__main__":
    main()

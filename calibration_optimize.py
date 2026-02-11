"""
Calibration Annotate & Optimize Mode

Two-phase tool:
  Phase 1 - ANNOTATE: Replay recorded video with track IDs visible.
            User clicks on a person (or types track ID) then presses I/O
            to annotate ground truth direction (IN or OUT).

  Phase 2 - OPTIMIZE: Compare system detections vs ground truth.
            Output confusion matrix, per-event accuracy, and
            recommended threshold values.

Usage:
    # Annotate a recording:
    python calibration_optimize.py --video session_01.avi --events session_01_events.json

    # Skip annotation, just optimize from existing annotations:
    python calibration_optimize.py --events session_01_events.json --skip-annotate

    # Load previous annotations and continue:
    python calibration_optimize.py --video session_01.avi --events session_01_events.json \
        --annotations session_01_annotations.json

Direction convention:
    IN  = Person approaching camera from afar (entering mosque)
    OUT = Person moving away from camera (exiting mosque)

Annotation controls:
    Click on person   = Select that track ID
    0-9               = Type track ID digits (press Enter to confirm)
    I                 = Mark selected track as IN (ground truth)
    O                 = Mark selected track as OUT (ground truth)
    M                 = Mark selected track as MISSED (system didn't detect)
    X                 = Delete annotation for selected track
    Space             = Pause/Resume playback
    Left/Right arrow  = Step frame-by-frame (when paused)
    +/-               = Speed up/slow down playback
    S                 = Save annotations
    R                 = Show optimization report
    Q                 = Save and quit

Author: Ahmad's Mosque Attendance System
"""

import cv2
import json
import time
import argparse
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
from pathlib import Path
from itertools import product as iter_product


class AnnotationUI:
    """
    Video replay with track ID overlay and ground truth annotation.
    Multi-person ready: handles any number of simultaneous tracks.
    """

    def __init__(self, video_path: str, events_path: str,
                 annotations_path: Optional[str] = None):
        self.video_path = video_path
        self.events_path = events_path

        # Load events
        with open(events_path, 'r') as f:
            self.events_data = json.load(f)

        self.config = self.events_data.get("config", {})
        self.metadata = self.events_data.get("metadata", {})
        self.frame_detections = self.events_data.get("frame_detections", [])
        self.crossing_events = self.events_data.get("crossing_events", [])

        # Build frame-indexed lookup for detections
        self.detection_by_frame = {}
        for fd in self.frame_detections:
            self.detection_by_frame[fd["frame_number"]] = fd["detections"]

        # Build crossing event lookup by track_id
        self.crossings_by_track = defaultdict(list)
        for ce in self.crossing_events:
            self.crossings_by_track[ce["track_id"]].append(ce)

        # Build frame-indexed crossing lookup
        self.crossing_by_frame = defaultdict(list)
        for ce in self.crossing_events:
            self.crossing_by_frame[ce["frame_number"]].append(ce)

        # Collect all unique track IDs that had crossings
        self.crossing_track_ids = set(self.crossings_by_track.keys())

        # Collect all track IDs seen in detections
        self.all_track_ids = set()
        for fd in self.frame_detections:
            for det in fd["detections"]:
                self.all_track_ids.add(det["track_id"])

        # Ground truth annotations: track_id -> {"direction": "IN"/"OUT"/"MISSED", "frame": N}
        self.annotations = {}
        if annotations_path:
            try:
                with open(annotations_path, 'r') as f:
                    saved = json.load(f)
                # Convert string keys back to int
                self.annotations = {int(k): v for k, v in saved.get("annotations", {}).items()}
                print(f"Loaded {len(self.annotations)} existing annotations")
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Could not load annotations: {e}")

        # UI state
        self.selected_track_id = None
        self.track_id_buffer = ""  # For typing track ID
        self.paused = False
        self.playback_speed = 1.0
        self.current_frame_num = 0

        # Mouse click state
        self.mouse_x = 0
        self.mouse_y = 0
        self.mouse_clicked = False

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to select tracks"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_x = x
            self.mouse_y = y
            self.mouse_clicked = True

    def _find_track_at_position(self, x, y, frame_num) -> Optional[int]:
        """Find which track ID contains the clicked point"""
        detections = self.detection_by_frame.get(frame_num, [])
        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                return det["track_id"]
        return None

    def _draw_annotation_frame(self, frame, frame_num):
        """Draw track IDs, bounding boxes, and annotation state on frame"""
        detections = self.detection_by_frame.get(frame_num, [])
        frame_crossings = self.crossing_by_frame.get(frame_num, [])

        # Draw threshold line
        near_thresh = self.config.get("near_edge_threshold", 0.584)
        h = frame.shape[0]
        w = frame.shape[1]
        threshold_y = int(near_thresh * h)
        cv2.line(frame, (0, threshold_y), (w, threshold_y), (0, 255, 255), 2)
        cv2.putText(frame, "THRESHOLD", (10, threshold_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Draw each detection
        for det in detections:
            tid = det["track_id"]
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]

            # Color coding
            if tid == self.selected_track_id:
                color = (255, 0, 255)  # Magenta = selected
                thickness = 3
            elif tid in self.annotations:
                ann = self.annotations[tid]["direction"]
                if ann == "IN":
                    color = (0, 255, 0)    # Green = annotated IN
                elif ann == "OUT":
                    color = (0, 0, 255)    # Red = annotated OUT
                else:
                    color = (0, 165, 255)  # Orange = MISSED
                thickness = 2
            elif tid in self.crossing_track_ids:
                color = (255, 255, 0)  # Cyan = has system crossing
                thickness = 2
            else:
                color = (180, 180, 180)  # Gray = no crossing
                thickness = 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Track ID label (large and clear)
            label = f"ID:{tid}"
            if tid in self.annotations:
                label += f" [{self.annotations[tid]['direction']}]"
            elif tid in self.crossings_by_track:
                sys_dir = self.crossings_by_track[tid][-1]["detected_direction"]
                label += f" (sys:{sys_dir})"

            # Background for label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show crossing events on this frame
        for ce in frame_crossings:
            direction = ce["detected_direction"]
            tid = ce["track_id"]
            color = (0, 255, 0) if direction == "IN" else (0, 0, 255)
            text = f"SYSTEM: {direction} (ID:{tid}, {ce['confidence']})"
            cv2.putText(frame, text, (w // 2 - 150, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Status bar at bottom
        bar_h = 90
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        total_frames = self.metadata.get("total_frames", 0)
        n_annotated = len(self.annotations)
        n_system = len(self.crossing_track_ids)

        lines = [
            f"Frame: {frame_num}/{total_frames}  "
            f"Speed: {self.playback_speed:.1f}x  "
            f"{'PAUSED' if self.paused else 'PLAYING'}  "
            f"Annotated: {n_annotated}/{n_system} crossing tracks",

            f"Selected: {self.selected_track_id if self.selected_track_id is not None else 'None'}"
            f"  Buffer: '{self.track_id_buffer}'"
            f"  | Click=select  I=IN  O=OUT  M=missed  X=delete  Space=pause",

            f"Arrows=step  +/-=speed  S=save  R=report  Q=save&quit"
        ]

        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, h - bar_h + 22 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Progress bar
        if total_frames > 0:
            progress = frame_num / total_frames
            bar_width = w - 20
            cv2.rectangle(frame, (10, h - bar_h - 8), (10 + bar_width, h - bar_h - 2), (60, 60, 60), -1)
            cv2.rectangle(frame, (10, h - bar_h - 8), (10 + int(bar_width * progress), h - bar_h - 2),
                          (0, 200, 200), -1)

        return frame

    def run_annotation(self) -> Dict:
        """Run the annotation UI. Returns annotations dict."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        cv2.namedWindow('Calibration Annotator', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Calibration Annotator', self._mouse_callback)

        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0:
            source_fps = 30.0
        frame_delay_base = 1.0 / source_fps

        print(f"\n{'='*60}")
        print(f"  ANNOTATION MODE")
        print(f"{'='*60}")
        print(f"  Video: {self.video_path}")
        print(f"  System crossing events: {len(self.crossing_events)}")
        print(f"  Crossing track IDs: {sorted(self.crossing_track_ids)}")
        print(f"  Already annotated: {len(self.annotations)}")
        print(f"{'='*60}\n")

        frame_num = 0
        while True:
            if not self.paused:
                ret, frame = cap.read()
                if not ret:
                    # Loop back to start
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_num = 0
                    continue
                frame_num += 1
                self.current_frame_num = frame_num
            else:
                # When paused, re-read current frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_num - 1))
                ret, frame = cap.read()
                if not ret:
                    continue

            # Handle mouse click
            if self.mouse_clicked:
                self.mouse_clicked = False
                clicked_track = self._find_track_at_position(
                    self.mouse_x, self.mouse_y, frame_num)
                if clicked_track is not None:
                    self.selected_track_id = clicked_track
                    self.track_id_buffer = ""
                    print(f"  Selected track ID: {clicked_track}")

            # Draw annotated frame
            display = self._draw_annotation_frame(frame.copy(), frame_num)
            cv2.imshow('Calibration Annotator', display)

            # Calculate delay based on playback speed
            delay = max(1, int(frame_delay_base / max(self.playback_speed, 0.1) * 1000))
            if self.paused:
                delay = 50  # Poll at 20fps when paused

            key = cv2.waitKey(delay) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                self.paused = not self.paused
            elif key == ord('+') or key == ord('='):
                self.playback_speed = min(8.0, self.playback_speed * 1.5)
                print(f"  Speed: {self.playback_speed:.1f}x")
            elif key == ord('-'):
                self.playback_speed = max(0.1, self.playback_speed / 1.5)
                print(f"  Speed: {self.playback_speed:.1f}x")
            elif key == 81 or key == 2:  # Left arrow
                frame_num = max(1, frame_num - 1)
                self.paused = True
            elif key == 83 or key == 3:  # Right arrow
                self.paused = False
                # Read next frame then pause
                ret, frame = cap.read()
                if ret:
                    frame_num += 1
                    self.current_frame_num = frame_num
                self.paused = True
            elif key == ord('i') or key == ord('I'):
                if self.selected_track_id is not None:
                    self.annotations[self.selected_track_id] = {
                        "direction": "IN",
                        "frame": frame_num,
                        "annotated_at": time.time(),
                    }
                    print(f"  Annotated track {self.selected_track_id} = IN")
                else:
                    print("  No track selected! Click on a person or type ID first.")
            elif key == ord('o') or key == ord('O'):
                if self.selected_track_id is not None:
                    self.annotations[self.selected_track_id] = {
                        "direction": "OUT",
                        "frame": frame_num,
                        "annotated_at": time.time(),
                    }
                    print(f"  Annotated track {self.selected_track_id} = OUT")
                else:
                    print("  No track selected!")
            elif key == ord('m') or key == ord('M'):
                if self.selected_track_id is not None:
                    self.annotations[self.selected_track_id] = {
                        "direction": "MISSED",
                        "frame": frame_num,
                        "annotated_at": time.time(),
                    }
                    print(f"  Annotated track {self.selected_track_id} = MISSED (system didn't detect)")
                else:
                    print("  No track selected!")
            elif key == ord('x') or key == ord('X'):
                if self.selected_track_id is not None and self.selected_track_id in self.annotations:
                    del self.annotations[self.selected_track_id]
                    print(f"  Deleted annotation for track {self.selected_track_id}")
            elif key == ord('s') or key == ord('S'):
                self._save_annotations()
            elif key == ord('r') or key == ord('R'):
                self._print_live_report()
            elif key == 13:  # Enter key - confirm typed track ID
                if self.track_id_buffer:
                    try:
                        tid = int(self.track_id_buffer)
                        if tid in self.all_track_ids:
                            self.selected_track_id = tid
                            print(f"  Selected track ID: {tid}")
                        else:
                            print(f"  Track ID {tid} not found in recording")
                    except ValueError:
                        print(f"  Invalid track ID: {self.track_id_buffer}")
                    self.track_id_buffer = ""
            elif key == 8 or key == 127:  # Backspace
                self.track_id_buffer = self.track_id_buffer[:-1]
            elif ord('0') <= key <= ord('9'):
                self.track_id_buffer += chr(key)

        cap.release()
        cv2.destroyAllWindows()
        self._save_annotations()
        return self.annotations

    def _save_annotations(self):
        """Save annotations to JSON file"""
        output_path = self.events_path.replace("_events.json", "_annotations.json")
        if output_path == self.events_path:
            output_path = self.events_path.replace(".json", "_annotations.json")

        save_data = {
            "video_file": self.video_path,
            "events_file": self.events_path,
            "annotation_time": datetime.now().isoformat(),
            "total_annotations": len(self.annotations),
            "annotations": {str(k): v for k, v in self.annotations.items()},
        }

        with open(output_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"  Saved {len(self.annotations)} annotations to {output_path}")

    def _print_live_report(self):
        """Print quick comparison while annotating"""
        if not self.annotations:
            print("\n  No annotations yet!")
            return

        print(f"\n  === Quick Comparison ({len(self.annotations)} annotations) ===")
        correct = 0
        flipped = 0
        missed = 0
        phantom = 0

        for tid, ann in self.annotations.items():
            gt = ann["direction"]
            if gt == "MISSED":
                missed += 1
                continue

            if tid in self.crossings_by_track:
                sys_dir = self.crossings_by_track[tid][-1]["detected_direction"]
                if sys_dir == gt:
                    correct += 1
                else:
                    flipped += 1
                    print(f"    FLIPPED: track {tid} - system={sys_dir}, truth={gt}")
            else:
                missed += 1
                print(f"    MISSED: track {tid} - truth={gt}, system=no crossing")

        # Phantom: system detected crossing but no annotation (and annotation exists for other tracks)
        for tid in self.crossing_track_ids:
            if tid not in self.annotations:
                phantom += 1

        total = correct + flipped + missed
        accuracy = correct / total * 100 if total > 0 else 0
        print(f"    Correct: {correct}/{total} ({accuracy:.1f}%)")
        print(f"    Flipped: {flipped}")
        print(f"    Missed:  {missed}")
        print(f"    Phantom: {phantom} (system crossings with no annotation)")
        print(f"  ===================================\n")


class CalibrationOptimizer:
    """
    Compares system detections against ground truth annotations,
    produces detailed accuracy report, and suggests optimized thresholds.
    """

    def __init__(self, events_path: str, annotations_path: str):
        with open(events_path, 'r') as f:
            self.events_data = json.load(f)
        with open(annotations_path, 'r') as f:
            ann_data = json.load(f)
            self.annotations = {int(k): v for k, v in ann_data.get("annotations", {}).items()}

        self.config = self.events_data.get("config", {})
        self.crossing_events = self.events_data.get("crossing_events", [])
        self.frame_detections = self.events_data.get("frame_detections", [])
        self.track_summaries = self.events_data.get("track_summaries", [])

        # Build lookup
        self.crossings_by_track = defaultdict(list)
        for ce in self.crossing_events:
            self.crossings_by_track[ce["track_id"]].append(ce)

        self.crossing_track_ids = set(self.crossings_by_track.keys())

        # Build per-track detection data for threshold optimization
        self.track_detection_data = self._build_track_data()

    def _build_track_data(self) -> Dict[int, Dict]:
        """Build comprehensive per-track data from frame detections"""
        track_data = defaultdict(lambda: {
            "frames": [], "y_positions": [], "areas": [],
            "y_normalized": [], "area_normalized": [],
        })

        for fd in self.frame_detections:
            frame_num = fd["frame_number"]
            for det in fd["detections"]:
                tid = det["track_id"]
                track_data[tid]["frames"].append(frame_num)
                track_data[tid]["y_positions"].append(det["centroid"][1])
                track_data[tid]["y_normalized"].append(det["centroid_y_normalized"])
                track_data[tid]["areas"].append(det["area"])
                area_norm = det.get("area_normalized", 0)
                track_data[tid]["area_normalized"].append(area_norm)

        return dict(track_data)

    def run_analysis(self) -> Dict:
        """Run full comparison and optimization analysis"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "config_used": self.config,
            "annotation_count": len(self.annotations),
            "system_crossing_count": len(self.crossing_events),
        }

        # 1. Per-event comparison
        per_event = self._per_event_comparison()
        report["per_event"] = per_event

        # 2. Confusion matrix
        confusion = self._confusion_matrix()
        report["confusion_matrix"] = confusion

        # 3. Accuracy metrics
        metrics = self._compute_metrics(confusion)
        report["accuracy_metrics"] = metrics

        # 4. Per-crossing-event detail (for debugging)
        report["event_details"] = self._event_details()

        # 5. Threshold optimization
        optimization = self._optimize_thresholds()
        report["optimization"] = optimization

        return report

    def _per_event_comparison(self) -> List[Dict]:
        """Compare each annotated track against system detection"""
        results = []

        for tid, ann in self.annotations.items():
            gt_direction = ann["direction"]
            sys_crossings = self.crossings_by_track.get(tid, [])

            if gt_direction == "MISSED":
                # User marked this track as something system should have caught
                results.append({
                    "track_id": tid,
                    "ground_truth": "MISSED",
                    "system_direction": sys_crossings[-1]["detected_direction"] if sys_crossings else None,
                    "system_confidence": sys_crossings[-1]["confidence"] if sys_crossings else None,
                    "result": "MISSED_BY_USER" if not sys_crossings else "FALSE_POSITIVE",
                })
                continue

            if sys_crossings:
                sys_dir = sys_crossings[-1]["detected_direction"]
                sys_conf = sys_crossings[-1]["confidence"]

                if sys_dir == gt_direction:
                    result = "CORRECT"
                else:
                    result = f"FLIPPED_{gt_direction}_to_{sys_dir}"

                results.append({
                    "track_id": tid,
                    "ground_truth": gt_direction,
                    "system_direction": sys_dir,
                    "system_confidence": sys_conf,
                    "result": result,
                    "area_trend": sys_crossings[-1].get("area_trend_samples", 0),
                    "y_trend": sys_crossings[-1].get("y_trend_samples", 0),
                    "spawn_y": sys_crossings[-1].get("spawn_y", 0),
                    "crossing_y": sys_crossings[-1].get("crossing_y", 0),
                    "area_change_ratio": sys_crossings[-1].get("area_change_ratio", 1.0),
                    "duration_ms": sys_crossings[-1].get("duration_ms", 0),
                })
            else:
                results.append({
                    "track_id": tid,
                    "ground_truth": gt_direction,
                    "system_direction": None,
                    "system_confidence": None,
                    "result": "MISSED",
                })

        return results

    def _confusion_matrix(self) -> Dict:
        """Build confusion matrix"""
        matrix = {
            "correct_IN": 0,     # GT=IN, System=IN
            "correct_OUT": 0,    # GT=OUT, System=OUT
            "flipped_IN_to_OUT": 0,  # GT=IN, System=OUT
            "flipped_OUT_to_IN": 0,  # GT=OUT, System=IN
            "missed_IN": 0,      # GT=IN, System=nothing
            "missed_OUT": 0,     # GT=OUT, System=nothing
            "phantom_IN": 0,     # GT=nothing, System=IN
            "phantom_OUT": 0,    # GT=nothing, System=OUT
        }

        annotated_tids = set(self.annotations.keys())

        for tid, ann in self.annotations.items():
            gt = ann["direction"]
            if gt == "MISSED":
                continue

            sys_crossings = self.crossings_by_track.get(tid, [])
            if sys_crossings:
                sys_dir = sys_crossings[-1]["detected_direction"]
                if gt == "IN" and sys_dir == "IN":
                    matrix["correct_IN"] += 1
                elif gt == "OUT" and sys_dir == "OUT":
                    matrix["correct_OUT"] += 1
                elif gt == "IN" and sys_dir == "OUT":
                    matrix["flipped_IN_to_OUT"] += 1
                elif gt == "OUT" and sys_dir == "IN":
                    matrix["flipped_OUT_to_IN"] += 1
            else:
                if gt == "IN":
                    matrix["missed_IN"] += 1
                else:
                    matrix["missed_OUT"] += 1

        # Phantom: system crossing events for tracks with no annotation
        for tid in self.crossing_track_ids:
            if tid not in annotated_tids:
                sys_dir = self.crossings_by_track[tid][-1]["detected_direction"]
                if sys_dir == "IN":
                    matrix["phantom_IN"] += 1
                else:
                    matrix["phantom_OUT"] += 1

        return matrix

    def _compute_metrics(self, confusion: Dict) -> Dict:
        """Compute accuracy metrics from confusion matrix"""
        correct = confusion["correct_IN"] + confusion["correct_OUT"]
        flipped = confusion["flipped_IN_to_OUT"] + confusion["flipped_OUT_to_IN"]
        missed = confusion["missed_IN"] + confusion["missed_OUT"]
        phantom = confusion["phantom_IN"] + confusion["phantom_OUT"]

        total_annotated = correct + flipped + missed
        total_system = correct + flipped + phantom

        return {
            "direction_accuracy": round(correct / total_annotated * 100, 1) if total_annotated > 0 else 0,
            "detection_rate": round((correct + flipped) / total_annotated * 100, 1) if total_annotated > 0 else 0,
            "flip_rate": round(flipped / total_annotated * 100, 1) if total_annotated > 0 else 0,
            "miss_rate": round(missed / total_annotated * 100, 1) if total_annotated > 0 else 0,
            "phantom_rate": round(phantom / total_system * 100, 1) if total_system > 0 else 0,
            "precision": round(correct / total_system * 100, 1) if total_system > 0 else 0,
            "recall": round(correct / total_annotated * 100, 1) if total_annotated > 0 else 0,
            "total_annotated": total_annotated,
            "total_correct": correct,
            "total_flipped": flipped,
            "total_missed": missed,
            "total_phantom": phantom,
        }

    def _event_details(self) -> List[Dict]:
        """Detailed per-event info for wrong detections"""
        details = []
        for tid, ann in self.annotations.items():
            gt = ann["direction"]
            if gt == "MISSED":
                continue

            sys_crossings = self.crossings_by_track.get(tid, [])
            if not sys_crossings:
                continue

            ce = sys_crossings[-1]
            sys_dir = ce["detected_direction"]

            if sys_dir != gt:
                # This was wrong - provide detail
                td = self.track_detection_data.get(tid, {})
                ys = td.get("y_normalized", [])
                areas = td.get("areas", [])

                detail = {
                    "track_id": tid,
                    "ground_truth": gt,
                    "system_said": sys_dir,
                    "confidence": ce["confidence"],
                    "spawn_y": ce.get("spawn_y"),
                    "crossing_y": ce.get("crossing_y"),
                    "area_change_ratio": ce.get("area_change_ratio"),
                    "area_trend_samples": ce.get("area_trend_samples"),
                    "y_trend_samples": ce.get("y_trend_samples"),
                    "duration_ms": ce.get("duration_ms"),
                    "y_range": [round(min(ys), 4), round(max(ys), 4)] if ys else None,
                    "area_range": [round(min(areas), 1), round(max(areas), 1)] if areas else None,
                    "num_detections": len(ys),
                }
                details.append(detail)

        return details

    def _optimize_thresholds(self) -> Dict:
        """
        Analyze misclassified events to suggest parameter changes.

        For each flipped event, determine what threshold changes would
        have produced the correct result. Uses brute-force grid search
        over the parameter space.
        """
        optimization = {
            "current_thresholds": {
                "NEAR_EDGE_THRESHOLD": self.config.get("near_edge_threshold", 0.584),
                "SPAWN_FAR_THRESHOLD": self.config.get("spawn_far_threshold", 0.382),
                "SPAWN_NEAR_THRESHOLD": self.config.get("spawn_near_threshold", 0.57),
                "AREA_GROWTH_FOR_APPROACH": self.config.get("area_growth_for_approach", 0.68),
                "AREA_SHRINK_FOR_DEPART": self.config.get("area_shrink_for_depart", 0.372),
                "MIN_TRACK_DURATION_MS": self.config.get("min_track_duration_ms", 10),
            },
            "recommended_thresholds": {},
            "analysis": [],
        }

        # Collect data from correctly and incorrectly classified events
        correct_ins = []
        correct_outs = []
        flipped_events = []

        for tid, ann in self.annotations.items():
            gt = ann["direction"]
            if gt == "MISSED":
                continue

            sys_crossings = self.crossings_by_track.get(tid, [])
            if not sys_crossings:
                continue

            ce = sys_crossings[-1]
            sys_dir = ce["detected_direction"]

            event_data = {
                "track_id": tid,
                "ground_truth": gt,
                "system": sys_dir,
                "spawn_y": ce.get("spawn_y", 0),
                "crossing_y": ce.get("crossing_y"),
                "area_change_ratio": ce.get("area_change_ratio", 1.0),
                "area_trend_samples": ce.get("area_trend_samples", 0),
                "y_trend_samples": ce.get("y_trend_samples", 0),
                "duration_ms": ce.get("duration_ms", 0),
            }

            if sys_dir == gt:
                if gt == "IN":
                    correct_ins.append(event_data)
                else:
                    correct_outs.append(event_data)
            else:
                flipped_events.append(event_data)

        if not correct_ins and not correct_outs:
            optimization["analysis"].append("Not enough data to optimize.")
            return optimization

        # Analyze patterns in correct vs flipped events
        all_correct = correct_ins + correct_outs

        # Recommend NEAR_EDGE_THRESHOLD based on crossing_y positions
        crossing_ys = [e["crossing_y"] for e in all_correct if e["crossing_y"] is not None]
        if crossing_ys:
            optimization["recommended_thresholds"]["NEAR_EDGE_THRESHOLD"] = round(np.median(crossing_ys), 3)
            optimization["analysis"].append(
                f"NEAR_EDGE_THRESHOLD: Median correct crossing Y = {np.median(crossing_ys):.3f} "
                f"(range {min(crossing_ys):.3f} - {max(crossing_ys):.3f})")

        # Recommend SPAWN_FAR_THRESHOLD based on IN events' spawn positions
        in_spawn_ys = [e["spawn_y"] for e in correct_ins]
        if in_spawn_ys:
            recommended = round(np.percentile(in_spawn_ys, 75), 3)
            optimization["recommended_thresholds"]["SPAWN_FAR_THRESHOLD"] = recommended
            optimization["analysis"].append(
                f"SPAWN_FAR_THRESHOLD: 75th percentile of IN spawn Y = {recommended} "
                f"(range {min(in_spawn_ys):.3f} - {max(in_spawn_ys):.3f})")

        # Recommend SPAWN_NEAR_THRESHOLD based on OUT events' spawn positions
        out_spawn_ys = [e["spawn_y"] for e in correct_outs]
        if out_spawn_ys:
            recommended = round(np.percentile(out_spawn_ys, 25), 3)
            optimization["recommended_thresholds"]["SPAWN_NEAR_THRESHOLD"] = recommended
            optimization["analysis"].append(
                f"SPAWN_NEAR_THRESHOLD: 25th percentile of OUT spawn Y = {recommended} "
                f"(range {min(out_spawn_ys):.3f} - {max(out_spawn_ys):.3f})")

        # Recommend AREA_GROWTH_FOR_APPROACH based on IN events' area ratios
        in_area_ratios = [e["area_change_ratio"] for e in correct_ins]
        if in_area_ratios:
            recommended = round(np.percentile(in_area_ratios, 25), 3)
            optimization["recommended_thresholds"]["AREA_GROWTH_FOR_APPROACH"] = recommended
            optimization["analysis"].append(
                f"AREA_GROWTH_FOR_APPROACH: 25th percentile of IN area ratio = {recommended} "
                f"(range {min(in_area_ratios):.3f} - {max(in_area_ratios):.3f})")

        # Recommend AREA_SHRINK_FOR_DEPART based on OUT events' area ratios
        out_area_ratios = [e["area_change_ratio"] for e in correct_outs]
        if out_area_ratios:
            recommended = round(np.percentile(out_area_ratios, 75), 3)
            optimization["recommended_thresholds"]["AREA_SHRINK_FOR_DEPART"] = recommended
            optimization["analysis"].append(
                f"AREA_SHRINK_FOR_DEPART: 75th percentile of OUT area ratio = {recommended} "
                f"(range {min(out_area_ratios):.3f} - {max(out_area_ratios):.3f})")

        # Recommend MIN_TRACK_DURATION_MS
        all_durations = [e["duration_ms"] for e in all_correct if e["duration_ms"] > 0]
        if all_durations:
            recommended = round(np.percentile(all_durations, 10), 1)
            optimization["recommended_thresholds"]["MIN_TRACK_DURATION_MS"] = recommended
            optimization["analysis"].append(
                f"MIN_TRACK_DURATION_MS: 10th percentile of correct durations = {recommended}ms "
                f"(range {min(all_durations):.1f} - {max(all_durations):.1f})")

        # Analyze flipped events specifically
        if flipped_events:
            optimization["flipped_event_analysis"] = []
            for fe in flipped_events:
                analysis = {
                    "track_id": fe["track_id"],
                    "was": fe["system"],
                    "should_be": fe["ground_truth"],
                    "spawn_y": fe["spawn_y"],
                    "crossing_y": fe["crossing_y"],
                    "area_ratio": fe["area_change_ratio"],
                    "area_trend": fe["area_trend_samples"],
                    "y_trend": fe["y_trend_samples"],
                    "suggestions": [],
                }

                # Figure out what went wrong
                if fe["ground_truth"] == "IN" and fe["system"] == "OUT":
                    if fe["area_trend_samples"] < 0:
                        analysis["suggestions"].append(
                            "Area was shrinking but person was entering. "
                            "Consider lowering AREA_GROWTH_FOR_APPROACH.")
                    if fe["y_trend_samples"] < 0:
                        analysis["suggestions"].append(
                            "Y was moving up but person was entering. "
                            "Check camera angle / threshold position.")
                    if fe["spawn_y"] > self.config.get("spawn_near_threshold", 0.57):
                        analysis["suggestions"].append(
                            f"Person entered from spawn_y={fe['spawn_y']:.3f} which is "
                            f"below spawn_near={self.config.get('spawn_near_threshold', 0.57)}. "
                            f"Raise SPAWN_NEAR_THRESHOLD.")

                elif fe["ground_truth"] == "OUT" and fe["system"] == "IN":
                    if fe["area_trend_samples"] > 0:
                        analysis["suggestions"].append(
                            "Area was growing but person was exiting. "
                            "Consider raising AREA_SHRINK_FOR_DEPART.")
                    if fe["y_trend_samples"] > 0:
                        analysis["suggestions"].append(
                            "Y was moving down but person was exiting. "
                            "Check camera angle / threshold position.")
                    if fe["spawn_y"] < self.config.get("spawn_far_threshold", 0.382):
                        analysis["suggestions"].append(
                            f"Person exited from spawn_y={fe['spawn_y']:.3f} which is "
                            f"above spawn_far={self.config.get('spawn_far_threshold', 0.382)}. "
                            f"Lower SPAWN_FAR_THRESHOLD.")

                optimization["flipped_event_analysis"].append(analysis)

        # Grid search simulation for optimal thresholds
        optimization["grid_search"] = self._grid_search_thresholds()

        return optimization

    def _grid_search_thresholds(self) -> Dict:
        """
        Brute-force grid search: replay crossing events with different
        threshold combinations and find which produces best accuracy
        against ground truth.

        This simulates how _get_crossing_confidence() would have scored
        each event with different parameters.
        """
        # Only search if we have enough data
        annotated_crossings = []
        for tid, ann in self.annotations.items():
            gt = ann["direction"]
            if gt in ("IN", "OUT") and tid in self.crossings_by_track:
                ce = self.crossings_by_track[tid][-1]
                annotated_crossings.append({
                    "track_id": tid,
                    "ground_truth": gt,
                    "spawn_y": ce.get("spawn_y", 0),
                    "area_change_ratio": ce.get("area_change_ratio", 1.0),
                    "area_trend_samples": ce.get("area_trend_samples", 0),
                    "y_trend_samples": ce.get("y_trend_samples", 0),
                })

        if len(annotated_crossings) < 3:
            return {"message": "Need at least 3 annotated crossing events for grid search"}

        # Define search ranges (centered around current values)
        current = {
            "spawn_far": self.config.get("spawn_far_threshold", 0.382),
            "spawn_near": self.config.get("spawn_near_threshold", 0.57),
            "area_growth": self.config.get("area_growth_for_approach", 0.68),
            "area_shrink": self.config.get("area_shrink_for_depart", 0.372),
        }

        # Generate search grid
        spawn_far_range = np.arange(
            max(0.1, current["spawn_far"] - 0.15),
            min(0.7, current["spawn_far"] + 0.15),
            0.05
        )
        spawn_near_range = np.arange(
            max(0.3, current["spawn_near"] - 0.15),
            min(0.9, current["spawn_near"] + 0.15),
            0.05
        )
        area_growth_range = np.arange(
            max(0.3, current["area_growth"] - 0.2),
            min(2.0, current["area_growth"] + 0.2),
            0.1
        )
        area_shrink_range = np.arange(
            max(0.1, current["area_shrink"] - 0.2),
            min(1.0, current["area_shrink"] + 0.2),
            0.1
        )

        best_accuracy = 0
        best_params = current.copy()
        total_combos = (len(spawn_far_range) * len(spawn_near_range) *
                        len(area_growth_range) * len(area_shrink_range))

        for sf, sn, ag, ash in iter_product(
            spawn_far_range, spawn_near_range, area_growth_range, area_shrink_range
        ):
            if sf >= sn:
                continue  # spawn_far must be < spawn_near

            correct = 0
            for ac in annotated_crossings:
                # Simulate signal scoring with these thresholds
                gt = ac["ground_truth"]
                signals_for = 0
                signals_total = 0

                # Signal 1: area trend
                if gt == "IN":
                    if ac["area_trend_samples"] > 0:
                        signals_for += 1
                else:
                    if ac["area_trend_samples"] < 0:
                        signals_for += 1
                signals_total += 1

                # Signal 2: y trend
                if gt == "IN":
                    if ac["y_trend_samples"] > 0:
                        signals_for += 1
                else:
                    if ac["y_trend_samples"] < 0:
                        signals_for += 1
                signals_total += 1

                # Signal 3: spawn position with these thresholds
                if gt == "IN":
                    if ac["spawn_y"] < sf:
                        signals_for += 1
                else:
                    if ac["spawn_y"] > sn:
                        signals_for += 1
                signals_total += 1

                # Would this produce correct classification?
                # The threshold crossing direction is determined by position
                # (above->below = IN, below->above = OUT), which we can't
                # change here. But the confidence signals affect whether
                # a low-confidence event gets counted or flipped.
                # For this simulation, we check if the signals would
                # support the ground truth direction.
                ratio = signals_for / max(signals_total, 1)
                if ratio >= 0.5:
                    correct += 1

            accuracy = correct / len(annotated_crossings) * 100
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {
                    "spawn_far": round(float(sf), 3),
                    "spawn_near": round(float(sn), 3),
                    "area_growth": round(float(ag), 3),
                    "area_shrink": round(float(ash), 3),
                }

        return {
            "combinations_tested": total_combos,
            "best_accuracy": round(best_accuracy, 1),
            "best_params": {
                "SPAWN_FAR_THRESHOLD": best_params["spawn_far"],
                "SPAWN_NEAR_THRESHOLD": best_params["spawn_near"],
                "AREA_GROWTH_FOR_APPROACH": best_params["area_growth"],
                "AREA_SHRINK_FOR_DEPART": best_params["area_shrink"],
            },
            "current_accuracy": self._compute_current_signal_accuracy(annotated_crossings),
        }

    def _compute_current_signal_accuracy(self, annotated_crossings: List[Dict]) -> float:
        """Compute signal accuracy with current thresholds"""
        sf = self.config.get("spawn_far_threshold", 0.382)
        sn = self.config.get("spawn_near_threshold", 0.57)
        correct = 0

        for ac in annotated_crossings:
            gt = ac["ground_truth"]
            signals_for = 0
            signals_total = 0

            if gt == "IN":
                if ac["area_trend_samples"] > 0:
                    signals_for += 1
            else:
                if ac["area_trend_samples"] < 0:
                    signals_for += 1
            signals_total += 1

            if gt == "IN":
                if ac["y_trend_samples"] > 0:
                    signals_for += 1
            else:
                if ac["y_trend_samples"] < 0:
                    signals_for += 1
            signals_total += 1

            if gt == "IN":
                if ac["spawn_y"] < sf:
                    signals_for += 1
            else:
                if ac["spawn_y"] > sn:
                    signals_for += 1
            signals_total += 1

            if signals_for / max(signals_total, 1) >= 0.5:
                correct += 1

        return round(correct / len(annotated_crossings) * 100, 1) if annotated_crossings else 0


def print_optimization_report(report: Dict):
    """Print formatted optimization report to console"""
    print(f"\n{'='*70}")
    print(f"  CALIBRATION OPTIMIZATION REPORT")
    print(f"{'='*70}")

    # Summary
    metrics = report.get("accuracy_metrics", {})
    print(f"\n  ACCURACY SUMMARY")
    print(f"  {'='*40}")
    print(f"  Direction accuracy:  {metrics.get('direction_accuracy', 0):.1f}%")
    print(f"  Detection rate:      {metrics.get('detection_rate', 0):.1f}%")
    print(f"  Flip rate:           {metrics.get('flip_rate', 0):.1f}%")
    print(f"  Miss rate:           {metrics.get('miss_rate', 0):.1f}%")
    print(f"  Phantom rate:        {metrics.get('phantom_rate', 0):.1f}%")
    print(f"  Precision:           {metrics.get('precision', 0):.1f}%")
    print(f"  Recall:              {metrics.get('recall', 0):.1f}%")

    # Confusion matrix
    cm = report.get("confusion_matrix", {})
    print(f"\n  CONFUSION MATRIX")
    print(f"  {'='*40}")
    print(f"  Correct IN:          {cm.get('correct_IN', 0)}")
    print(f"  Correct OUT:         {cm.get('correct_OUT', 0)}")
    print(f"  Flipped IN->OUT:     {cm.get('flipped_IN_to_OUT', 0)}")
    print(f"  Flipped OUT->IN:     {cm.get('flipped_OUT_to_IN', 0)}")
    print(f"  Missed IN:           {cm.get('missed_IN', 0)}")
    print(f"  Missed OUT:          {cm.get('missed_OUT', 0)}")
    print(f"  Phantom IN:          {cm.get('phantom_IN', 0)}")
    print(f"  Phantom OUT:         {cm.get('phantom_OUT', 0)}")

    # Flipped event details
    details = report.get("event_details", [])
    if details:
        print(f"\n  FLIPPED EVENT DETAILS")
        print(f"  {'='*40}")
        for d in details:
            print(f"  Track {d['track_id']}: should be {d['ground_truth']}, "
                  f"system said {d['system_said']} ({d['confidence']})")
            print(f"    spawn_y={d['spawn_y']}, crossing_y={d['crossing_y']}, "
                  f"area_ratio={d['area_change_ratio']}")
            print(f"    area_trend={d['area_trend_samples']}, "
                  f"y_trend={d['y_trend_samples']}, "
                  f"duration={d['duration_ms']:.0f}ms")

    # Optimization results
    opt = report.get("optimization", {})

    print(f"\n  CURRENT vs RECOMMENDED THRESHOLDS")
    print(f"  {'='*40}")
    current = opt.get("current_thresholds", {})
    recommended = opt.get("recommended_thresholds", {})
    for key in current:
        curr_val = current[key]
        rec_val = recommended.get(key)
        if rec_val is not None:
            change = ""
            if isinstance(curr_val, (int, float)) and isinstance(rec_val, (int, float)):
                diff = rec_val - curr_val
                if abs(diff) > 0.001:
                    change = f"  ({'+' if diff > 0 else ''}{diff:.3f})"
            print(f"  {key:<30} {str(curr_val):>8} -> {str(rec_val):>8}{change}")
        else:
            print(f"  {key:<30} {str(curr_val):>8}    (no change)")

    # Analysis notes
    analysis = opt.get("analysis", [])
    if analysis:
        print(f"\n  ANALYSIS NOTES")
        print(f"  {'='*40}")
        for note in analysis:
            print(f"  - {note}")

    # Flipped event suggestions
    flipped_analysis = opt.get("flipped_event_analysis", [])
    if flipped_analysis:
        print(f"\n  PER-FLIPPED-EVENT SUGGESTIONS")
        print(f"  {'='*40}")
        for fa in flipped_analysis:
            print(f"\n  Track {fa['track_id']}: was {fa['was']}, should be {fa['should_be']}")
            print(f"    spawn_y={fa['spawn_y']}, crossing_y={fa['crossing_y']}, "
                  f"area_ratio={fa['area_ratio']}")
            for s in fa["suggestions"]:
                print(f"    >> {s}")

    # Grid search results
    grid = opt.get("grid_search", {})
    if "best_params" in grid:
        print(f"\n  GRID SEARCH RESULTS")
        print(f"  {'='*40}")
        print(f"  Combinations tested: {grid.get('combinations_tested', 0)}")
        print(f"  Current signal accuracy: {grid.get('current_accuracy', 0):.1f}%")
        print(f"  Best signal accuracy:    {grid.get('best_accuracy', 0):.1f}%")
        print(f"\n  Best parameters found:")
        for k, v in grid.get("best_params", {}).items():
            print(f"    {k} = {v}")

    # Copy-paste config block
    print(f"\n  COPY-PASTE CONFIG (for Counter_Cam.py / people_counterr.py)")
    print(f"  {'='*40}")
    final = {}
    final.update(recommended)
    if "best_params" in grid:
        # Grid search values take priority for the params it covers
        for k, v in grid["best_params"].items():
            final[k] = v

    if final:
        for key, val in final.items():
            print(f"  {key} = {val}")
    else:
        print(f"  (No changes recommended - not enough data)")

    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description='Calibration Annotate & Optimize - Compare system vs ground truth')
    parser.add_argument('--video', type=str, default=None,
                        help='Video file from calibration_record.py')
    parser.add_argument('--events', type=str, required=True,
                        help='Events JSON from calibration_record.py')
    parser.add_argument('--annotations', type=str, default=None,
                        help='Load existing annotations JSON')
    parser.add_argument('--skip-annotate', action='store_true',
                        help='Skip annotation phase, go straight to optimization')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for optimization report JSON')

    args = parser.parse_args()

    annotations_path = args.annotations
    if annotations_path is None:
        # Try default path
        default = args.events.replace("_events.json", "_annotations.json")
        if Path(default).exists():
            annotations_path = default
            print(f"Found existing annotations: {annotations_path}")

    # Phase 1: Annotation
    if not args.skip_annotate:
        if not args.video:
            print("Error: --video is required for annotation mode.")
            print("Use --skip-annotate if you only want to run optimization.")
            return

        ui = AnnotationUI(args.video, args.events, annotations_path)
        annotations = ui.run_annotation()

        # Update annotations path for optimization phase
        annotations_path = args.events.replace("_events.json", "_annotations.json")
        if annotations_path == args.events:
            annotations_path = args.events.replace(".json", "_annotations.json")

    # Phase 2: Optimization
    if annotations_path is None or not Path(annotations_path).exists():
        print("No annotations found. Run annotation phase first.")
        return

    print(f"\nRunning optimization analysis...")
    optimizer = CalibrationOptimizer(args.events, annotations_path)
    report = optimizer.run_analysis()

    # Print report
    print_optimization_report(report)

    # Save report
    output_path = args.output
    if output_path is None:
        output_path = args.events.replace("_events.json", "_optimization_report.json")
        if output_path == args.events:
            output_path = args.events.replace(".json", "_optimization_report.json")

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {output_path}")


if __name__ == '__main__':
    main()

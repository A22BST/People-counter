"""
Calibration Record Mode - Observe and Log Counter_Cam Behavior

Runs the full Counter_Cam pipeline (YOLOv8 + ByteTrack + DirectionDetector)
and records everything for later annotation and optimization.

Saves:
- Video file (.avi) of the session
- JSON event log with every system event: detections, crossings, track data

Does NOT change any counting behavior - purely observational.

Usage:
    # Record from webcam:
    python calibration_record.py --source 0 --output session_01

    # Record from video file:
    python calibration_record.py --source test_video.mp4 --output session_01

    # Record with custom thresholds:
    python calibration_record.py --source 0 --output session_01 --near-threshold 0.584

Output:
    session_01.avi          - Recorded video
    session_01_events.json  - All system events and metadata

Then use calibration_optimize.py to annotate and optimize.

Author: Ahmad's Mosque Attendance System
"""

import cv2
import json
import time
import argparse
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from collections import deque
from pathlib import Path

from ultralytics import YOLO
from direction_detector import DirectionDetector, DirectionDetectorConfig, TrackState


# ============================================================================
#                      QUICK CONFIG (mirrors Counter_Cam.py)
# ============================================================================

CAMERA_SOURCE = "0"
DETECTION_CONFIDENCE = 0.3
NEAR_EDGE_THRESHOLD = 0.584
SPAWN_FAR_THRESHOLD = 0.382
SPAWN_NEAR_THRESHOLD = 0.57
AREA_GROWTH_FOR_APPROACH = 0.68
AREA_SHRINK_FOR_DEPART = 0.372
MIN_TRACK_DURATION_MS = 10
COUNT_ON_CROSSING = True
CROSSING_HYSTERESIS = 0.03
FLIP_DIRECTION = False

# ============================================================================


class CalibrationRecorder:
    """
    Records Counter_Cam behavior without modifying it.

    Hooks into DirectionDetector signals to log every event,
    while simultaneously saving the video for later replay.
    """

    def __init__(self, source, output_prefix: str,
                 model_path: str = "yolov8s.pt",
                 detection_confidence: float = DETECTION_CONFIDENCE,
                 near_edge_threshold: float = NEAR_EDGE_THRESHOLD,
                 spawn_far_threshold: float = SPAWN_FAR_THRESHOLD,
                 spawn_near_threshold: float = SPAWN_NEAR_THRESHOLD,
                 area_growth_for_approach: float = AREA_GROWTH_FOR_APPROACH,
                 area_shrink_for_depart: float = AREA_SHRINK_FOR_DEPART,
                 min_track_duration_ms: float = MIN_TRACK_DURATION_MS,
                 crossing_hysteresis: float = CROSSING_HYSTERESIS,
                 flip_direction: bool = FLIP_DIRECTION):

        self.source = source
        self.output_prefix = output_prefix
        self.model_path = model_path
        self.flip_direction = flip_direction

        # Store config for logging
        self.config_dict = {
            "source": str(source),
            "model": model_path,
            "detection_confidence": detection_confidence,
            "near_edge_threshold": near_edge_threshold,
            "spawn_far_threshold": spawn_far_threshold,
            "spawn_near_threshold": spawn_near_threshold,
            "area_growth_for_approach": area_growth_for_approach,
            "area_shrink_for_depart": area_shrink_for_depart,
            "min_track_duration_ms": min_track_duration_ms,
            "crossing_hysteresis": crossing_hysteresis,
            "count_on_crossing": COUNT_ON_CROSSING,
            "flip_direction": flip_direction,
        }

        # Build DirectionDetector config (same as Counter_Cam)
        self.detector_config = DirectionDetectorConfig(
            near_edge_threshold=near_edge_threshold,
            spawn_far_threshold=spawn_far_threshold,
            spawn_near_threshold=spawn_near_threshold,
            area_growth_for_approach=area_growth_for_approach,
            area_shrink_for_depart=area_shrink_for_depart,
            min_track_duration_ms=min_track_duration_ms,
            count_on_crossing=COUNT_ON_CROSSING,
            crossing_hysteresis=crossing_hysteresis,
        )
        self.detection_confidence = detection_confidence

        # Components (initialized in start())
        self.model = None
        self.cap = None
        self.video_writer = None
        self.direction_detector = None

        # Frame info
        self.frame_width = 0
        self.frame_height = 0
        self.frame_count = 0
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()

        # Event log storage
        self.events_log = {
            "metadata": {},
            "config": self.config_dict,
            "frame_detections": [],   # Per-frame detection data
            "crossing_events": [],    # Direction crossing events
            "track_summaries": [],    # Per-track summary when track ends
        }

        # Track active track IDs for detecting terminations
        self.active_track_ids = set()

        self.running = False

    def start(self):
        """Initialize all components"""
        print("Loading YOLO model...")
        self.model = YOLO(self.model_path)

        # Open video source
        source = self.source
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Get source FPS (for video files), default to 30 for cameras
        source_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0 or source_fps > 120:
            source_fps = 30.0

        # Update detector config with actual frame dimensions
        self.detector_config.frame_width = self.frame_width
        self.detector_config.frame_height = self.frame_height

        # Initialize direction detector (same as Counter_Cam uses)
        self.direction_detector = DirectionDetector(self.detector_config)

        # Initialize video writer
        video_path = f"{self.output_prefix}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(
            video_path, fourcc, source_fps,
            (self.frame_width, self.frame_height)
        )

        # Store metadata
        self.events_log["metadata"] = {
            "start_time": datetime.now().isoformat(),
            "video_file": video_path,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "source_fps": source_fps,
        }

        self.running = True

        print(f"\n{'='*60}")
        print(f"  CALIBRATION RECORDER STARTED")
        print(f"{'='*60}")
        print(f"  Video: {self.frame_width}x{self.frame_height} @ {source_fps:.0f}fps")
        print(f"  Output video: {video_path}")
        print(f"  Output events: {self.output_prefix}_events.json")
        print(f"  Thresholds:")
        print(f"    near_edge:  {self.detector_config.near_edge_threshold}")
        print(f"    spawn_far:  {self.detector_config.spawn_far_threshold}")
        print(f"    spawn_near: {self.detector_config.spawn_near_threshold}")
        print(f"    area_growth: {self.detector_config.area_growth_for_approach}")
        print(f"    area_shrink: {self.detector_config.area_shrink_for_depart}")
        if self.flip_direction:
            print(f"  Direction: FLIPPED (IN<->OUT)")
        print(f"\n  Controls: [q]=quit  [s]=stats")
        print(f"{'='*60}\n")

    def process_frame(self) -> Optional[np.ndarray]:
        """Process one frame: detect, track, log, and record"""
        if not self.cap or not self.running:
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        self.frame_count += 1
        current_time = time.time()

        # FPS
        frame_time = current_time - self.last_frame_time
        self.fps_history.append(1.0 / max(frame_time, 0.001))
        self.last_frame_time = current_time

        # Run YOLO with ByteTrack (same as Counter_Cam)
        results = self.model.track(
            frame,
            persist=True,
            conf=self.detection_confidence,
            classes=[0],  # person only
            verbose=False,
            tracker="bytetrack.yaml"
        )

        # Extract detections
        detections = []
        detection_records = []
        current_track_ids = set()

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                track_id = int(boxes.id[i].cpu().numpy())
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu().numpy())
                x1, y1, x2, y2 = bbox

                centroid_x = (x1 + x2) / 2
                centroid_y = (y1 + y2) / 2
                area = (x2 - x1) * (y2 - y1)

                detections.append((track_id, (x1, y1, x2, y2)))
                current_track_ids.add(track_id)

                detection_records.append({
                    "track_id": track_id,
                    "bbox": [round(v, 1) for v in bbox],
                    "confidence": round(conf, 3),
                    "centroid": [round(centroid_x, 1), round(centroid_y, 1)],
                    "centroid_y_normalized": round(centroid_y / self.frame_height, 4),
                    "area": round(area, 1),
                    "area_normalized": round(area / (self.frame_width * self.frame_height), 6),
                })

        # Log frame detections (only frames with detections to save space)
        if detection_records:
            self.events_log["frame_detections"].append({
                "timestamp": current_time,
                "frame_number": self.frame_count,
                "detections": detection_records,
            })

        # Feed to direction detector (hooks into existing signals)
        crossing_events = self.direction_detector.update(detections)

        # Log crossing events
        for event in crossing_events:
            direction = event["direction"]
            if self.flip_direction:
                direction = "OUT" if direction == "IN" else "IN"

            crossing_record = {
                "timestamp": current_time,
                "frame_number": self.frame_count,
                "track_id": event["track_id"],
                "detected_direction": direction,
                "original_direction": event["direction"],
                "confidence": event["confidence"],
                "trigger": event["trigger"],
                "duration_ms": round(event.get("duration_ms", 0), 1),
                "spawn_y": round(event.get("spawn_y", 0), 4),
                "crossing_y": round(event.get("crossing_y", 0), 4) if event.get("crossing_y") else None,
                "area_change_ratio": round(event.get("area_change_ratio", 1.0), 4),
                "area_trend_samples": event.get("area_trend_samples", 0),
                "y_trend_samples": event.get("y_trend_samples", 0),
            }
            self.events_log["crossing_events"].append(crossing_record)

            print(f"  [{self.frame_count:>6}] CROSSING: track {event['track_id']:>3} -> "
                  f"{direction} (conf={event['confidence']}, "
                  f"area_trend={event.get('area_trend_samples', 0)}, "
                  f"y_trend={event.get('y_trend_samples', 0)})")

        # Log track summaries for terminated tracks
        terminated_ids = self.active_track_ids - current_track_ids
        for tid in terminated_ids:
            track = self.direction_detector.tracks.get(tid)
            if track:
                summary = self._build_track_summary(track)
                if summary:
                    self.events_log["track_summaries"].append(summary)

        self.active_track_ids = current_track_ids

        # Write raw frame to video (no overlay, for clean replay)
        self.video_writer.write(frame)

        # Draw visualization for live display
        display_frame = self._draw_overlay(frame.copy(), detections, crossing_events)

        return display_frame

    def _build_track_summary(self, track) -> Optional[dict]:
        """Build summary for a terminated track from DirectionDetector's TrackHistory"""
        if not track.bbox_history:
            return None

        areas = [a for _, a in track.area_history]
        ys = [y for _, y in track.centroid_y_history]

        return {
            "track_id": track.track_id,
            "spawn_time": track.spawn_time,
            "spawn_y_normalized": round(track.spawn_y_normalized, 4),
            "spawn_area": round(track.spawn_area, 1),
            "last_update_time": track.last_update_time,
            "duration_ms": round((track.last_update_time - track.spawn_time) * 1000, 1),
            "num_samples": len(track.bbox_history),
            "has_crossed": track.has_crossed_threshold,
            "crossed_direction": track.crossed_direction,
            "area_trend_samples": track.area_trend_samples,
            "y_trend_samples": track.y_trend_samples,
            "final_state": track.state.name,
            "min_area": round(min(areas), 1) if areas else None,
            "max_area": round(max(areas), 1) if areas else None,
            "min_y": round(min(ys), 4) if ys else None,
            "max_y": round(max(ys), 4) if ys else None,
            "state_history": [(s.name if hasattr(s, 'name') else str(s))
                              for _, s in track.state_history],
        }

    def _draw_overlay(self, frame, detections, crossing_events):
        """Draw visualization overlay on frame"""
        # Draw threshold line
        threshold_y = int(self.detector_config.near_edge_threshold * self.frame_height)
        cv2.line(frame, (0, threshold_y), (self.frame_width, threshold_y), (0, 255, 255), 2)
        cv2.putText(frame, "THRESHOLD", (10, threshold_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Draw hysteresis zone
        hyst = self.detector_config.crossing_hysteresis
        upper = int((self.detector_config.near_edge_threshold - hyst) * self.frame_height)
        lower = int((self.detector_config.near_edge_threshold + hyst) * self.frame_height)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, upper), (self.frame_width, lower), (0, 255, 255), -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # Draw spawn zone lines
        spawn_far_y = int(self.detector_config.spawn_far_threshold * self.frame_height)
        spawn_near_y = int(self.detector_config.spawn_near_threshold * self.frame_height)
        cv2.line(frame, (0, spawn_far_y), (self.frame_width, spawn_far_y), (255, 200, 0), 1)
        cv2.putText(frame, "SPAWN_FAR", (self.frame_width - 120, spawn_far_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
        cv2.line(frame, (0, spawn_near_y), (self.frame_width, spawn_near_y), (0, 200, 255), 1)
        cv2.putText(frame, "SPAWN_NEAR", (self.frame_width - 130, spawn_near_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

        # Draw detections with track info
        for track_id, (x1, y1, x2, y2) in detections:
            # Color by state from direction detector
            debug = self.direction_detector.get_debug_info(track_id) if hasattr(self.direction_detector, 'get_debug_info') else None
            track = self.direction_detector.tracks.get(track_id)

            if track:
                state = track.state
                if state == TrackState.APPROACHING:
                    color = (0, 255, 0)    # Green
                elif state == TrackState.DEPARTING:
                    color = (0, 0, 255)    # Red
                elif state == TrackState.CROSSED_IN:
                    color = (0, 200, 0)
                elif state == TrackState.CROSSED_OUT:
                    color = (0, 0, 200)
                elif state == TrackState.STABLE:
                    color = (255, 255, 0)  # Cyan
                else:
                    color = (128, 128, 128)
            else:
                color = (255, 255, 255)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Track ID label (large, visible)
            label = f"ID:{track_id}"
            if track:
                label += f" {track.state.name[:3]}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Flash crossing events
        for event in crossing_events:
            direction = event["direction"]
            if self.flip_direction:
                direction = "OUT" if direction == "IN" else "IN"
            color = (0, 255, 0) if direction == "IN" else (0, 0, 255)
            text = f"{direction} (ID:{event['track_id']}, {event['confidence']})"
            cv2.putText(frame, text, (self.frame_width // 2 - 100, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Stats overlay
        overlay_bg = frame.copy()
        cv2.rectangle(overlay_bg, (5, 5), (280, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay_bg, 0.7, frame, 0.3, 0, frame)

        entries, exits, occ = self.direction_detector.get_counts()
        fps = np.mean(self.fps_history) if self.fps_history else 0
        n_crossings = len(self.events_log["crossing_events"])

        stats = [
            f"REC  Frame: {self.frame_count}  FPS: {fps:.1f}",
            f"IN: {entries}  OUT: {exits}  Occ: {occ}",
            f"Crossings logged: {n_crossings}",
            f"Active tracks: {len(self.active_track_ids)}",
            f"[s]=stats  [q]=save & quit",
        ]
        for i, s in enumerate(stats):
            cv2.putText(frame, s, (10, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Recording indicator (blinking red dot)
        if self.frame_count % 30 < 15:
            cv2.circle(frame, (self.frame_width - 20, 20), 8, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (self.frame_width - 60, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return frame

    def stop(self):
        """Stop recording and finalize"""
        self.running = False

        # Finalize any remaining active tracks
        for tid in list(self.direction_detector.tracks.keys()):
            track = self.direction_detector.tracks[tid]
            summary = self._build_track_summary(track)
            if summary:
                self.events_log["track_summaries"].append(summary)

        self.events_log["metadata"]["end_time"] = datetime.now().isoformat()
        self.events_log["metadata"]["total_frames"] = self.frame_count
        self.events_log["metadata"]["total_crossing_events"] = len(self.events_log["crossing_events"])
        self.events_log["metadata"]["total_tracks"] = len(self.events_log["track_summaries"])

        entries, exits, _ = self.direction_detector.get_counts()
        self.events_log["metadata"]["system_count_in"] = entries
        self.events_log["metadata"]["system_count_out"] = exits

        if self.video_writer:
            self.video_writer.release()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def save(self):
        """Save event log to JSON"""
        json_path = f"{self.output_prefix}_events.json"
        with open(json_path, 'w') as f:
            json.dump(self.events_log, f, indent=2)

        entries = self.events_log["metadata"].get("system_count_in", 0)
        exits = self.events_log["metadata"].get("system_count_out", 0)
        n_crossings = len(self.events_log["crossing_events"])
        n_tracks = len(self.events_log["track_summaries"])

        print(f"\n{'='*60}")
        print(f"  RECORDING SAVED")
        print(f"{'='*60}")
        print(f"  Video: {self.output_prefix}.avi")
        print(f"  Events: {json_path}")
        print(f"  Frames: {self.frame_count}")
        print(f"  Crossing events: {n_crossings}")
        print(f"  Tracks recorded: {n_tracks}")
        print(f"  System count: IN={entries}, OUT={exits}")
        print(f"{'='*60}")
        print(f"\n  Next: python calibration_optimize.py "
              f"--video {self.output_prefix}.avi "
              f"--events {json_path}")

    def run(self):
        """Main recording loop"""
        self.start()

        try:
            while self.running:
                frame = self.process_frame()
                if frame is None:
                    print("\nEnd of video source.")
                    break

                cv2.imshow('Calibration Recorder', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    entries, exits, occ = self.direction_detector.get_counts()
                    n_crossings = len(self.events_log["crossing_events"])
                    print(f"\n=== Recording Stats ===")
                    print(f"  Frame: {self.frame_count}")
                    print(f"  System: IN={entries}, OUT={exits}, Occ={occ}")
                    print(f"  Crossings logged: {n_crossings}")
                    print(f"  Tracks: {len(self.events_log['track_summaries'])} completed, "
                          f"{len(self.active_track_ids)} active")
                    print(f"=======================\n")
        finally:
            self.stop()
            self.save()


def main():
    parser = argparse.ArgumentParser(
        description='Calibration Record Mode - Observe and log Counter_Cam behavior')
    parser.add_argument('--source', type=str, default='0',
                        help='Video source (0=webcam, URL, or file path)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output prefix (creates <prefix>.avi and <prefix>_events.json)')
    parser.add_argument('--model', type=str, default='yolov8s.pt',
                        help='YOLO model path')
    parser.add_argument('--confidence', type=float, default=DETECTION_CONFIDENCE,
                        help='Detection confidence threshold')
    parser.add_argument('--near-threshold', type=float, default=NEAR_EDGE_THRESHOLD,
                        help='Near-edge threshold (0-1)')
    parser.add_argument('--spawn-far', type=float, default=SPAWN_FAR_THRESHOLD,
                        help='Spawn far threshold')
    parser.add_argument('--spawn-near', type=float, default=SPAWN_NEAR_THRESHOLD,
                        help='Spawn near threshold')
    parser.add_argument('--area-growth', type=float, default=AREA_GROWTH_FOR_APPROACH,
                        help='Area growth for approach')
    parser.add_argument('--area-shrink', type=float, default=AREA_SHRINK_FOR_DEPART,
                        help='Area shrink for depart')
    parser.add_argument('--min-duration', type=float, default=MIN_TRACK_DURATION_MS,
                        help='Minimum track duration (ms)')
    parser.add_argument('--hysteresis', type=float, default=CROSSING_HYSTERESIS,
                        help='Crossing hysteresis')
    parser.add_argument('--flip', action='store_true',
                        help='Flip IN/OUT direction')

    args = parser.parse_args()

    recorder = CalibrationRecorder(
        source=args.source,
        output_prefix=args.output,
        model_path=args.model,
        detection_confidence=args.confidence,
        near_edge_threshold=args.near_threshold,
        spawn_far_threshold=args.spawn_far,
        spawn_near_threshold=args.spawn_near,
        area_growth_for_approach=args.area_growth,
        area_shrink_for_depart=args.area_shrink,
        min_track_duration_ms=args.min_duration,
        crossing_hysteresis=args.hysteresis,
        flip_direction=args.flip,
    )

    recorder.run()


if __name__ == '__main__':
    main()

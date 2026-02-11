"""
People Counter: Camera Pipeline (v3.0 - Hybrid Detection)
==========================================================
YOLOv8 + ByteTrack + Hybrid DirectionDetector

Direction detected by:
  1. Threshold crossing (instant, handles people staying in frame)
  2. Evidence scoring (confidence validation)
  3. Termination fallback (catches edge cases)

Usage:
    python people_counter.py                                        # Default
    python people_counter.py --source 1                             # Camera 1
    python people_counter.py --source http://192.168.1.5:4747/video # DroidCam
    python people_counter.py --source video.mp4                     # Video file

Controls:
    Q - Quit
    R - Reset counters
    D - Toggle debug visualization
    F - Flip direction (swap IN/OUT)
    S - Screenshot
"""

import cv2
import numpy as np
import argparse
import time
import sys
from collections import deque

try:
    from ultralytics import YOLO
    import supervision as sv
except ImportError:
    print("Install: pip install ultralytics supervision")
    sys.exit(1)

from direction_detector import DirectionDetector, DirectionDetectorConfig, TrackState


# ============================================================================
#                        QUICK CONFIGURATION
# ============================================================================

# Camera source
CAMERA_SOURCE = "http://192.168.1.5:4747/video"  # DroidCam default

# YOLO
MODEL_PATH = "yolov8s.pt"
CONFIDENCE = 0.3

# Threshold crossing line position (0=top, 1=bottom)
# This is WHERE the count triggers. Adjust based on your camera.
NEAR_EDGE_THRESHOLD = 0.60
CROSSING_HYSTERESIS = 0.03

# Evidence scoring thresholds (for confidence validation + fallback)
SPAWN_FAR_THRESHOLD = 0.35
SPAWN_NEAR_THRESHOLD = 0.65

# Track requirements
MIN_TRACK_DURATION_MS = 200
MIN_TRACK_FRAMES = 4

# Direction flip
FLIP_DIRECTION = False

# ============================================================================


class PeopleCounter:
    def __init__(self, source, model_path=MODEL_PATH, confidence=CONFIDENCE):
        self.source = source
        self.confidence = confidence
        self.show_debug = True
        self.flip_direction = FLIP_DIRECTION
        
        self.recent_events: deque = deque(maxlen=8)
        self.fps_history: deque = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # Open camera
        print(f"Opening: {source}")
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open: {source}")
        
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Cannot read from camera")
        
        self.frame_height, self.frame_width = frame.shape[:2]
        print(f"Frame: {self.frame_width}x{self.frame_height}")
        
        # YOLO
        print(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
        
        # ByteTrack
        self.tracker = sv.ByteTrack(
            track_activation_threshold=confidence,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30,
        )
        
        # Hybrid DirectionDetector
        config = DirectionDetectorConfig(
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            near_edge_threshold=NEAR_EDGE_THRESHOLD,
            crossing_hysteresis=CROSSING_HYSTERESIS,
            spawn_far_threshold=SPAWN_FAR_THRESHOLD,
            spawn_near_threshold=SPAWN_NEAR_THRESHOLD,
            min_track_duration_ms=MIN_TRACK_DURATION_MS,
            min_track_frames=MIN_TRACK_FRAMES,
        )
        self.detector = DirectionDetector(config)
        
        print("=" * 50)
        print("HYBRID DETECTOR READY")
        print(f"  Threshold line: {NEAR_EDGE_THRESHOLD:.0%} from top")
        print(f"  Fallback: evidence scoring on termination")
        print("=" * 50)
        print("Q=Quit  R=Reset  D=Debug  F=Flip  S=Screenshot")
        print()
    
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Lost feed, retrying...")
                time.sleep(1)
                self.cap = cv2.VideoCapture(self.source)
                continue
            
            frame = self._process_frame(frame)
            cv2.imshow("People Counter", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('r') or key == ord('R'):
                self.detector.reset()
                self.recent_events.clear()
                print("[RESET] Counters cleared")
            elif key == ord('d') or key == ord('D'):
                self.show_debug = not self.show_debug
            elif key == ord('f') or key == ord('F'):
                self.flip_direction = not self.flip_direction
                print(f"[FLIP] Direction {'FLIPPED' if self.flip_direction else 'NORMAL'}")
            elif key == ord('s') or key == ord('S'):
                fname = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(fname, frame)
                print(f"[SCREENSHOT] {fname}")
        
        self.cap.release()
        cv2.destroyAllWindows()
        stats = self.detector.get_stats()
        print(f"\nFinal: {stats['entries']} in, {stats['exits']} out, "
              f"occupancy: {stats['current_occupancy']}")
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        # FPS
        now = time.time()
        dt = now - self.last_frame_time
        self.last_frame_time = now
        if dt > 0:
            self.fps_history.append(1.0 / dt)
        
        # Detect
        results = self.model(frame, classes=[0], conf=self.confidence, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)
        
        # Build detection list for DirectionDetector
        det_list = []
        if detections.tracker_id is not None and len(detections.tracker_id) > 0:
            for i, track_id in enumerate(detections.tracker_id):
                if track_id is None:
                    continue
                det_list.append((int(track_id), tuple(detections.xyxy[i])))
        
        # Update detector
        events = self.detector.update(det_list)
        
        # Handle events
        for event in events:
            direction = event["direction"]
            if self.flip_direction:
                direction = "OUT" if direction == "IN" else "IN"
            
            conf = event["confidence"]
            tid = event["track_id"]
            trigger = event["trigger"]
            stats = self.detector.get_stats()
            
            arrow = "-> IN" if direction == "IN" else "<- OUT"
            evt_str = f"{arrow} #{tid} [{conf}] ({trigger[:5]}) | " \
                      f"In:{stats['entries']} Out:{stats['exits']} Occ:{stats['current_occupancy']}"
            self.recent_events.append((time.time(), evt_str, direction))
            
            print(f"[{direction}] Track {tid} | {conf} | {trigger} | "
                  f"Entries:{stats['entries']} Exits:{stats['exits']} "
                  f"Occ:{stats['current_occupancy']}")
        
        # Draw
        frame = self._draw(frame, detections)
        return frame
    
    def _draw(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        h, w = frame.shape[:2]
        
        # === THRESHOLD LINE ===
        threshold_y = int(NEAR_EDGE_THRESHOLD * h)
        hyst_top = int((NEAR_EDGE_THRESHOLD - CROSSING_HYSTERESIS) * h)
        hyst_bot = int((NEAR_EDGE_THRESHOLD + CROSSING_HYSTERESIS) * h)
        
        # Hysteresis zone (subtle fill)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, hyst_top), (w, hyst_bot), (0, 255, 255), -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        
        # Threshold line
        cv2.line(frame, (0, threshold_y), (w, threshold_y), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "THRESHOLD", (w - 130, threshold_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        if self.show_debug:
            # Spawn far line
            far_y = int(SPAWN_FAR_THRESHOLD * h)
            cv2.line(frame, (0, far_y), (w, far_y), (255, 180, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, "spawn_far", (5, far_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 180, 0), 1)
            
            # Spawn near line
            near_y = int(SPAWN_NEAR_THRESHOLD * h)
            cv2.line(frame, (0, near_y), (w, near_y), (0, 180, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "spawn_near", (5, near_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 180, 255), 1)
        
        # === TRACKED PERSONS ===
        if detections.tracker_id is not None and len(detections.tracker_id) > 0:
            for i, track_id in enumerate(detections.tracker_id):
                if track_id is None:
                    continue
                
                track_id = int(track_id)
                bbox = detections.xyxy[i].astype(int)
                x1, y1, x2, y2 = bbox
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                info = self.detector.get_track_info(track_id)
                
                # Color by state
                if info and info.get("has_crossed"):
                    if info["crossed_direction"] == "IN":
                        color = (0, 255, 0)    # Green = counted IN
                    else:
                        color = (0, 0, 255)    # Red = counted OUT
                elif info:
                    state = info["state"]
                    if state == "approaching":
                        color = (0, 200, 100)  # Green-ish
                    elif state == "departing":
                        color = (100, 100, 255) # Red-ish
                    elif state == "stable":
                        color = (255, 255, 0)  # Cyan
                    else:
                        color = (200, 200, 200) # Gray = nascent
                else:
                    color = (200, 200, 200)
                
                # Bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 4, color, -1)
                
                # Label
                label = f"#{track_id}"
                if info:
                    if info["has_crossed"]:
                        label += f" [{info['crossed_direction']}]"
                    elif self.show_debug:
                        label += f" {info['state'][:3]}"
                        ts = info.get("threshold_state", "?")
                        label += f" {ts}"
                
                cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Movement trail (debug)
                if self.show_debug and info and track_id in self.detector.tracks:
                    track = self.detector.tracks[track_id]
                    pts = track.y_history[-10:]
                    for j, y_val in enumerate(pts):
                        ty = int(y_val * h)
                        alpha = j / len(pts)
                        tc = tuple(int(c * alpha) for c in color)
                        cv2.circle(frame, (cx, ty), 2, tc, -1)
        
        # === STATS PANEL ===
        stats = self.detector.get_stats()
        fps = sum(self.fps_history) / max(len(self.fps_history), 1)
        
        panel_h = 130
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (280, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)
        
        y = 25
        cv2.putText(frame, f"ENTRIES: {stats['entries']}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 25
        cv2.putText(frame, f"EXITS:   {stats['exits']}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y += 25
        cv2.putText(frame, f"INSIDE:  {stats['current_occupancy']}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 25
        cv2.putText(frame, f"FPS: {fps:.1f} | Tracks: {stats['active_tracks']}", 
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        y += 20
        flip_str = "FLIPPED" if self.flip_direction else "normal"
        cv2.putText(frame, f"Dir: {flip_str} | Debug: {'ON' if self.show_debug else 'OFF'}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)
        
        # === EVENT LOG ===
        if self.recent_events:
            ev_h = min(len(self.recent_events) * 22 + 10, 200)
            ov2 = frame.copy()
            cv2.rectangle(ov2, (0, h - ev_h), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(ov2, 0.6, frame, 0.4, 0, frame)
            
            for idx, (evt_time, evt_str, evt_dir) in enumerate(reversed(list(self.recent_events))):
                age = time.time() - evt_time
                if age > 15:
                    continue
                alpha = max(0.3, 1.0 - age / 15.0)
                if evt_dir == "IN":
                    ec = (0, int(255 * alpha), 0)
                else:
                    ec = (0, 0, int(255 * alpha))
                yp = h - ev_h + 20 + idx * 22
                if yp < h - 5:
                    cv2.putText(frame, evt_str, (10, yp),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, ec, 1)
        
        return frame


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="People Counter (Hybrid Detection)")
    parser.add_argument("--source", default=CAMERA_SOURCE)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--conf", type=float, default=CONFIDENCE)
    parser.add_argument("--threshold", type=float, default=NEAR_EDGE_THRESHOLD,
                        help="Threshold line position (0=top, 1=bottom)")
    parser.add_argument("--flip", action="store_true", help="Flip IN/OUT direction")
    args = parser.parse_args()
    
    source = args.source
    try:
        source = int(source)
    except ValueError:
        pass
    
    # Override module-level defaults if args provided
    import direction_detector as dd
    counter = PeopleCounter(source=source, model_path=args.model, confidence=args.conf)
    
    # Apply CLI overrides
    if args.threshold != NEAR_EDGE_THRESHOLD:
        counter.detector.config.near_edge_threshold = args.threshold
    if args.flip:
        counter.flip_direction = True
    
    counter.run()


if __name__ == "__main__":
    main()
"""
Quick Video Recorder - Capture Camera Footage for Later Analysis

Simple, lightweight recorder with no YOLO processing.
Record yourself walking in/out, then review with ground_truth_annotator.py

Controls:
    R - Start/Stop recording
    Q - Quit
    S - Take screenshot

Usage:
    # Record from webcam
    python quick_record.py
    
    # Record from specific camera
    python quick_record.py --source 1
    
    # Record from DroidCam
    python quick_record.py --source "http://192.168.1.100:4747/video"
    
    # Then analyze the recording:
    python ground_truth_annotator.py --source recording_20260211_123456.mp4

Author: Ahmad's Mosque Attendance System
"""

import cv2
import numpy as np
import argparse
import time
from datetime import datetime
from pathlib import Path


class QuickRecorder:
    def __init__(self, source='0', output_dir='.', auto_start=False):
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
        print("="*60 + "\n")
    
    def start_recording(self):
        """Start recording video."""
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
        
        self.recording = True
        self.frame_count = 0
        self.start_time = time.time()
        
        print(f"\n🔴 RECORDING STARTED")
        print(f"   File: {self.recording_path}")
        print(f"   Press R to stop, Q to quit\n")
    
    def stop_recording(self):
        """Stop recording video."""
        if not self.recording:
            print("Not recording!")
            return
        
        self.recording = False
        duration = time.time() - self.start_time
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        print(f"\n⏹️  RECORDING STOPPED")
        print(f"   Saved: {self.recording_path}")
        print(f"   Duration: {duration:.1f}s ({self.frame_count} frames)")
        print(f"\n   Next step:")
        print(f"   python ground_truth_annotator.py --source {Path(self.recording_path).name}\n")
    
    def draw_overlay(self, frame):
        """Draw recording indicator and info."""
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
            cv2.putText(frame, f"Frames: {self.frame_count}", (w - 180, 32),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
        else:
            # Ready to record
            cv2.putText(frame, "READY - Press R to record", (20, 32),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Bottom controls
        controls_text = "R=Record/Stop  |  S=Screenshot  |  Q=Quit"
        text_size = cv2.getTextSize(controls_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = (w - text_size[0]) // 2
        
        cv2.rectangle(frame, (0, h - 30), (w, h), (30, 30, 30), -1)
        cv2.putText(frame, controls_text, (text_x, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        
        return frame
    
    def take_screenshot(self, frame):
        """Save a screenshot."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = str(self.output_dir / f"screenshot_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot saved: {filename}")
    
    def run(self):
        """Main loop."""
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
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            
            print("\nRecorder closed.")


def main():
    parser = argparse.ArgumentParser(description='Quick Video Recorder')
    parser.add_argument('--source', type=str, default='0',
                       help='Camera source: index (0, 1, ...) or URL')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory for recordings')
    parser.add_argument('--auto-start', action='store_true',
                       help='Start recording immediately')
    
    args = parser.parse_args()
    
    recorder = QuickRecorder(
        source=args.source,
        output_dir=args.output_dir,
        auto_start=args.auto_start
    )
    recorder.run()


if __name__ == "__main__":
    main()

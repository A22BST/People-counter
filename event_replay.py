"""
Event Replay & Analysis - Compare Different Methods Against Ground Truth

Takes recorded event data and replays it through various configurations
to determine which sensor combination and fusion strategy works best.

Usage:
    # First, edit the JSON to add ground truth:
    #   "ground_truth_in": 150,
    #   "ground_truth_out": 148
    
    # Then analyze:
    python event_replay.py --input friday_prayer.json
    
    # Or compare specific configs:
    python event_replay.py --input friday_prayer.json --config camera_only
    python event_replay.py --input friday_prayer.json --config camera_radar_confirm

Author: Ahmad's Mosque Attendance System
Version: 1.0.0
"""

import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import numpy as np


@dataclass
class AnalysisResult:
    """Result of analyzing with one configuration"""
    config_name: str
    entries: int
    exits: int
    accuracy_in: Optional[float] = None
    accuracy_out: Optional[float] = None
    overall_accuracy: Optional[float] = None
    false_positives: int = 0
    missed_detections: int = 0
    details: Dict = None


class ConfigSimulator:
    """Simulates different counting configurations on recorded data"""
    
    def __init__(self, recording: Dict):
        self.recording = recording
        self.camera_crossings = recording.get("camera_crossings", [])
        self.esp32_events = recording.get("esp32_events", [])
        self.ground_truth_in = recording.get("ground_truth_in")
        self.ground_truth_out = recording.get("ground_truth_out")
        self.ground_truth_entries = recording.get("ground_truth_entries", [])
        self.ground_truth_exits = recording.get("ground_truth_exits", [])
        
        # Event type mappings
        self.RADAR_ENTRY = 1
        self.RADAR_EXIT = 2
        self.TOF_ENTRY = 6
        self.TOF_EXIT = 7
    
    def analyze_camera_only(self) -> AnalysisResult:
        """Config A: Camera only - count all camera crossings"""
        entries = sum(1 for c in self.camera_crossings if c["direction"] == "IN")
        exits = sum(1 for c in self.camera_crossings if c["direction"] == "OUT")
        
        result = self._make_result("A: Camera Only", entries, exits)
        result.details = self._match_timestamps("IN", self.camera_crossings)
        return result
    
    def analyze_camera_high_conf(self) -> AnalysisResult:
        """Config A2: Camera only, HIGH confidence only"""
        filtered = [c for c in self.camera_crossings if c["confidence"] == "HIGH"]
        entries = sum(1 for c in filtered if c["direction"] == "IN")
        exits = sum(1 for c in filtered if c["direction"] == "OUT")
        
        return self._make_result("A2: Camera HIGH only", entries, exits)
    
    def analyze_camera_med_high_conf(self) -> AnalysisResult:
        """Config A3: Camera only, MEDIUM or HIGH confidence"""
        filtered = [c for c in self.camera_crossings if c["confidence"] in ["HIGH", "MEDIUM"]]
        entries = sum(1 for c in filtered if c["direction"] == "IN")
        exits = sum(1 for c in filtered if c["direction"] == "OUT")
        
        return self._make_result("A3: Camera MED+HIGH", entries, exits)
    
    def analyze_camera_radar_confirmation(self, time_window: float = 0.5) -> AnalysisResult:
        """
        Config D: Camera + Radar (confirmation strategy)
        Camera decides, radar confirms/boosts confidence
        """
        entries = 0
        exits = 0
        
        for crossing in self.camera_crossings:
            ts = crossing["timestamp"]
            direction = crossing["direction"]
            cam_conf = crossing["confidence"]
            
            # Look for radar event within time window
            radar_confirms = False
            radar_conflicts = False
            
            for esp in self.esp32_events:
                if abs(esp["timestamp"] - ts) <= time_window:
                    if esp["event_type"] in [self.RADAR_ENTRY, self.RADAR_EXIT]:
                        esp_dir = "IN" if esp["event_type"] == self.RADAR_ENTRY else "OUT"
                        if esp_dir == direction:
                            radar_confirms = True
                        else:
                            radar_conflicts = True
            
            # Decision logic: camera + confirmation boost
            if radar_confirms:
                # Radar confirms - count it
                if direction == "IN":
                    entries += 1
                else:
                    exits += 1
            elif radar_conflicts:
                # Radar conflicts - only count HIGH confidence camera
                if cam_conf == "HIGH":
                    if direction == "IN":
                        entries += 1
                    else:
                        exits += 1
            else:
                # No radar data - use camera if MEDIUM or better
                if cam_conf in ["HIGH", "MEDIUM"]:
                    if direction == "IN":
                        entries += 1
                    else:
                        exits += 1
        
        return self._make_result("D: Camera+Radar (confirm)", entries, exits)
    
    def analyze_camera_radar_tiebreaker(self, time_window: float = 0.5) -> AnalysisResult:
        """
        Config E: Camera + Radar (tiebreaker strategy)
        Use radar only when camera signals are conflicting
        """
        entries = 0
        exits = 0
        
        for crossing in self.camera_crossings:
            ts = crossing["timestamp"]
            direction = crossing["direction"]
            cam_conf = crossing["confidence"]
            area_trend = crossing.get("area_trend_samples", 0)
            y_trend = crossing.get("y_trend_samples", 0)
            
            # Check if camera signals conflict
            # IN should have: area_trend > 0 (growing), y_trend > 0 (down)
            # OUT should have: area_trend < 0 (shrinking), y_trend < 0 (up)
            if direction == "IN":
                signals_agree = (area_trend > 0) == (y_trend > 0)
            else:
                signals_agree = (area_trend < 0) == (y_trend < 0)
            
            if signals_agree or cam_conf == "HIGH":
                # Camera confident, use it
                if direction == "IN":
                    entries += 1
                else:
                    exits += 1
            else:
                # Camera uncertain - use radar as tiebreaker
                for esp in self.esp32_events:
                    if abs(esp["timestamp"] - ts) <= time_window:
                        if esp["event_type"] == self.RADAR_ENTRY:
                            entries += 1
                            break
                        elif esp["event_type"] == self.RADAR_EXIT:
                            exits += 1
                            break
                else:
                    # No radar, use camera anyway (low confidence)
                    if direction == "IN":
                        entries += 1
                    else:
                        exits += 1
        
        return self._make_result("E: Camera+Radar (tiebreaker)", entries, exits)
    
    def analyze_camera_tof_confirmation(self, time_window: float = 0.5) -> AnalysisResult:
        """Config F: Camera + ToF (confirmation strategy)"""
        entries = 0
        exits = 0
        
        for crossing in self.camera_crossings:
            ts = crossing["timestamp"]
            direction = crossing["direction"]
            cam_conf = crossing["confidence"]
            
            tof_confirms = False
            for esp in self.esp32_events:
                if abs(esp["timestamp"] - ts) <= time_window:
                    if esp["event_type"] in [self.TOF_ENTRY, self.TOF_EXIT]:
                        esp_dir = "IN" if esp["event_type"] == self.TOF_ENTRY else "OUT"
                        if esp_dir == direction:
                            tof_confirms = True
            
            if tof_confirms or cam_conf in ["HIGH", "MEDIUM"]:
                if direction == "IN":
                    entries += 1
                else:
                    exits += 1
        
        return self._make_result("F: Camera+ToF (confirm)", entries, exits)
    
    def analyze_all_sensors(self, time_window: float = 0.5) -> AnalysisResult:
        """Config G: Camera + Radar + ToF (all sensors)"""
        entries = 0
        exits = 0
        
        for crossing in self.camera_crossings:
            ts = crossing["timestamp"]
            direction = crossing["direction"]
            cam_conf = crossing["confidence"]
            
            radar_confirms = False
            tof_confirms = False
            
            for esp in self.esp32_events:
                if abs(esp["timestamp"] - ts) <= time_window:
                    if esp["event_type"] == self.RADAR_ENTRY and direction == "IN":
                        radar_confirms = True
                    elif esp["event_type"] == self.RADAR_EXIT and direction == "OUT":
                        radar_confirms = True
                    elif esp["event_type"] == self.TOF_ENTRY and direction == "IN":
                        tof_confirms = True
                    elif esp["event_type"] == self.TOF_EXIT and direction == "OUT":
                        tof_confirms = True
            
            # Count if: HIGH confidence OR any sensor confirms
            if cam_conf == "HIGH" or radar_confirms or tof_confirms:
                if direction == "IN":
                    entries += 1
                else:
                    exits += 1
        
        return self._make_result("G: All Sensors", entries, exits)
    
    def analyze_with_threshold_tuning(self, confidence_filter: str = None,
                                      area_trend_min: int = None,
                                      y_trend_min: int = None) -> AnalysisResult:
        """Tunable camera-only analysis with custom thresholds"""
        entries = 0
        exits = 0
        
        for crossing in self.camera_crossings:
            direction = crossing["direction"]
            conf = crossing["confidence"]
            area_trend = crossing.get("area_trend_samples", 0)
            y_trend = crossing.get("y_trend_samples", 0)
            
            # Apply filters
            if confidence_filter and conf not in confidence_filter.split(","):
                continue
            
            if area_trend_min is not None:
                if direction == "IN" and area_trend < area_trend_min:
                    continue
                if direction == "OUT" and area_trend > -area_trend_min:
                    continue
            
            if y_trend_min is not None:
                if direction == "IN" and y_trend < y_trend_min:
                    continue
                if direction == "OUT" and y_trend > -y_trend_min:
                    continue
            
            if direction == "IN":
                entries += 1
            else:
                exits += 1
        
        name = f"Tuned (conf={confidence_filter}, area>={area_trend_min}, y>={y_trend_min})"
        return self._make_result(name, entries, exits)
    
    def _match_timestamps(self, direction: str, system_crossings: List[Dict], 
                          time_window: float = 2.0) -> Dict:
        """
        Match system crossings to ground truth by timestamp.
        Returns detailed matching info.
        """
        if direction == "IN":
            gt_events = self.ground_truth_entries
        else:
            gt_events = self.ground_truth_exits
        
        sys_events = [c for c in system_crossings if c["direction"] == direction]
        
        matched = 0
        unmatched_sys = []
        unmatched_gt = []
        
        gt_matched = set()
        
        for sys in sys_events:
            sys_ts = sys["timestamp"]
            found_match = False
            
            for i, gt in enumerate(gt_events):
                if i in gt_matched:
                    continue
                gt_ts = gt["timestamp"]
                if abs(sys_ts - gt_ts) <= time_window:
                    matched += 1
                    gt_matched.add(i)
                    found_match = True
                    break
            
            if not found_match:
                unmatched_sys.append(sys_ts)
        
        for i, gt in enumerate(gt_events):
            if i not in gt_matched:
                unmatched_gt.append(gt["timestamp"])
        
        return {
            "matched": matched,
            "false_positives": len(unmatched_sys),
            "missed": len(unmatched_gt),
            "precision": matched / len(sys_events) if sys_events else 0,
            "recall": matched / len(gt_events) if gt_events else 0
        }
    
    def _make_result(self, config_name: str, entries: int, exits: int) -> AnalysisResult:
        """Create result with accuracy calculations"""
        result = AnalysisResult(
            config_name=config_name,
            entries=entries,
            exits=exits,
            details={}
        )
        
        if self.ground_truth_in is not None:
            if self.ground_truth_in > 0:
                result.accuracy_in = min(100.0, entries / self.ground_truth_in * 100)
                result.missed_detections = max(0, self.ground_truth_in - entries)
                result.false_positives = max(0, entries - self.ground_truth_in)
            else:
                result.accuracy_in = 100.0 if entries == 0 else 0.0
        
        if self.ground_truth_out is not None:
            if self.ground_truth_out > 0:
                result.accuracy_out = min(100.0, exits / self.ground_truth_out * 100)
        
        if result.accuracy_in is not None and result.accuracy_out is not None:
            total_truth = (self.ground_truth_in or 0) + (self.ground_truth_out or 0)
            total_detected = entries + exits
            if total_truth > 0:
                result.overall_accuracy = min(100.0, total_detected / total_truth * 100)
        
        return result
    
    def run_all_analyses(self) -> List[AnalysisResult]:
        """Run all configuration analyses"""
        results = [
            self.analyze_camera_only(),
            self.analyze_camera_high_conf(),
            self.analyze_camera_med_high_conf(),
            self.analyze_camera_radar_confirmation(),
            self.analyze_camera_radar_tiebreaker(),
            self.analyze_camera_tof_confirmation(),
            self.analyze_all_sensors(),
        ]
        return results


def print_report(recording: Dict, results: List[AnalysisResult]):
    """Print analysis report"""
    print("\n" + "="*70)
    print("  EVENT ANALYSIS REPORT")
    print("="*70)
    
    # Event info
    print(f"\nEvent: {recording.get('event_name', 'Unknown')}")
    print(f"Date: {recording.get('start_time', 'Unknown')[:10]}")
    print(f"Duration: {recording.get('duration_seconds', 0):.0f} seconds")
    print(f"Frames: {recording.get('total_frames', 0)}")
    print(f"Camera crossings: {recording.get('total_crossings', 0)}")
    print(f"ESP32 events: {recording.get('total_esp_events', 0)}")
    
    # Ground truth
    gt_in = recording.get('ground_truth_in')
    gt_out = recording.get('ground_truth_out')
    print(f"\nGround Truth:")
    if gt_in is not None:
        print(f"  Entries: {gt_in}")
    else:
        print(f"  Entries: NOT SET - edit JSON to add 'ground_truth_in'")
    if gt_out is not None:
        print(f"  Exits: {gt_out}")
    else:
        print(f"  Exits: NOT SET - edit JSON to add 'ground_truth_out'")
    
    # Results table
    print("\n" + "-"*70)
    print(f"{'Configuration':<35} {'IN':>6} {'OUT':>6} {'Acc IN':>8} {'Acc OUT':>8} {'Overall':>8}")
    print("-"*70)
    
    for r in results:
        acc_in = f"{r.accuracy_in:.1f}%" if r.accuracy_in is not None else "N/A"
        acc_out = f"{r.accuracy_out:.1f}%" if r.accuracy_out is not None else "N/A"
        overall = f"{r.overall_accuracy:.1f}%" if r.overall_accuracy is not None else "N/A"
        
        print(f"{r.config_name:<35} {r.entries:>6} {r.exits:>6} {acc_in:>8} {acc_out:>8} {overall:>8}")
    
    print("-"*70)
    
    # Find best config
    if gt_in is not None:
        valid_results = [r for r in results if r.overall_accuracy is not None]
        if valid_results:
            best = max(valid_results, key=lambda r: r.overall_accuracy or 0)
            print(f"\n✓ BEST CONFIG: {best.config_name}")
            print(f"  Overall Accuracy: {best.overall_accuracy:.1f}%")
            if best.overall_accuracy >= 85:
                print(f"  ✓ Meets 85% target!")
            else:
                print(f"  ✗ Below 85% target - needs tuning")
    
    # Crossing confidence breakdown
    print("\n" + "-"*70)
    print("Camera Crossing Confidence Breakdown:")
    crossings = recording.get("camera_crossings", [])
    for conf in ["HIGH", "MEDIUM", "LOW"]:
        in_count = sum(1 for c in crossings if c["direction"] == "IN" and c["confidence"] == conf)
        out_count = sum(1 for c in crossings if c["direction"] == "OUT" and c["confidence"] == conf)
        print(f"  {conf}: IN={in_count}, OUT={out_count}")
    
    # ESP32 event breakdown
    print("\nESP32 Event Breakdown:")
    esp_events = recording.get("esp32_events", [])
    event_names = {1: "RADAR_ENTRY", 2: "RADAR_EXIT", 3: "RADAR_MOTION",
                   4: "TOF_TRIGGERED", 5: "TOF_CLEARED", 6: "TOF_ENTRY", 7: "TOF_EXIT"}
    event_counts = defaultdict(int)
    for e in esp_events:
        event_counts[e["event_type"]] += 1
    for etype, count in sorted(event_counts.items()):
        name = event_names.get(etype, f"TYPE_{etype}")
        print(f"  {name}: {count}")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description='Analyze recorded event data')
    parser.add_argument('--input', type=str, required=True,
                        help='Input JSON file from event_recorder.py')
    parser.add_argument('--config', type=str, default=None,
                        help='Specific config to analyze (or "all" for all)')
    parser.add_argument('--time-window', type=float, default=0.5,
                        help='Time window for sensor correlation (seconds)')
    
    args = parser.parse_args()
    
    # Load recording
    print(f"Loading {args.input}...")
    with open(args.input, 'r') as f:
        recording = json.load(f)
    
    # Run analysis
    simulator = ConfigSimulator(recording)
    results = simulator.run_all_analyses()
    
    # Print report
    print_report(recording, results)


if __name__ == '__main__':
    main()

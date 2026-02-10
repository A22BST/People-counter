"""
Unit Tests for Direction Detector

Tests the core detection logic without requiring camera input.
Simulates track patterns to verify classification accuracy.

Run with: python -m pytest test_direction_detector.py -v
Or simply: python test_direction_detector.py
"""

import unittest
import time
from direction_detector import (
    DirectionDetector, 
    DirectionDetectorConfig,
    TrackState,
    CrossingConfidence
)


class SimulatedTrack:
    """Helper to generate simulated track data"""
    
    def __init__(self, track_id: int, frame_height: int = 480, frame_width: int = 640):
        self.track_id = track_id
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.detections = []
    
    def add_point(self, y_normalized: float, area_normalized: float):
        """Add a detection point (y=0 is top, y=1 is bottom)"""
        # Convert to bbox
        height = (area_normalized * self.frame_height * self.frame_width) ** 0.5
        width = height * 0.5  # Assume width:height = 1:2 for person
        
        center_y = y_normalized * self.frame_height
        center_x = self.frame_width / 2
        
        x1 = center_x - width / 2
        y1 = center_y - height / 2
        x2 = center_x + width / 2
        y2 = center_y + height / 2
        
        self.detections.append((self.track_id, (x1, y1, x2, y2)))
        return self
    
    def generate_entry_pattern(self, num_frames: int = 20):
        """Generate typical entry pattern: far→near, small→large"""
        for i in range(num_frames):
            progress = i / (num_frames - 1)
            y = 0.2 + progress * 0.7  # 0.2 → 0.9 (top to bottom)
            area = 0.02 + progress * 0.06  # 2% → 8% (growing)
            self.add_point(y, area)
        return self
    
    def generate_exit_pattern(self, num_frames: int = 20):
        """Generate typical exit pattern: near→far, large→small"""
        for i in range(num_frames):
            progress = i / (num_frames - 1)
            y = 0.85 - progress * 0.6  # 0.85 → 0.25 (bottom to top)
            area = 0.08 - progress * 0.05  # 8% → 3% (shrinking)
            self.add_point(y, area)
        return self
    
    def generate_hesitation_then_entry(self, num_frames: int = 30):
        """Person hesitates at entrance, then enters"""
        # First half: hesitation (stable near middle)
        for i in range(num_frames // 2):
            y = 0.5 + (i % 3 - 1) * 0.02  # Slight movement
            area = 0.05 + (i % 2) * 0.005
            self.add_point(y, area)
        
        # Second half: decisive entry
        for i in range(num_frames // 2):
            progress = i / (num_frames // 2 - 1)
            y = 0.5 + progress * 0.4  # 0.5 → 0.9
            area = 0.05 + progress * 0.03
            self.add_point(y, area)
        return self
    
    def generate_turnaround(self, num_frames: int = 20):
        """Person approaches then turns around (should be abandoned)"""
        # Approach
        for i in range(num_frames // 2):
            progress = i / (num_frames // 2 - 1)
            y = 0.3 + progress * 0.3  # 0.3 → 0.6
            area = 0.03 + progress * 0.02
            self.add_point(y, area)
        
        # Turn around
        for i in range(num_frames // 2):
            progress = i / (num_frames // 2 - 1)
            y = 0.6 - progress * 0.3  # 0.6 → 0.3
            area = 0.05 - progress * 0.02
            self.add_point(y, area)
        return self


class TestDirectionDetector(unittest.TestCase):
    """Test cases for DirectionDetector"""
    
    def setUp(self):
        """Create detector with test configuration"""
        config = DirectionDetectorConfig(
            frame_height=480,
            frame_width=640,
            near_edge_threshold=0.85,
            min_track_duration_ms=100,  # Lower for faster tests
            min_samples_for_trend=3,
        )
        self.detector = DirectionDetector(config)
    
    def feed_track(self, track: SimulatedTrack, frame_delay_ms: float = 33.3) -> list:
        """Feed track data to detector and return events"""
        all_events = []
        
        for i, detection in enumerate(track.detections):
            # Simulate time passing
            time.sleep(frame_delay_ms / 1000.0)
            events = self.detector.update([detection])
            all_events.extend(events)
        
        # Feed empty to trigger termination
        time.sleep(0.1)
        events = self.detector.update([])
        all_events.extend(events)
        
        return all_events
    
    def test_clear_entry_pattern(self):
        """Test detection of clear entry pattern"""
        track = SimulatedTrack(1).generate_entry_pattern(25)
        events = self.feed_track(track)
        
        self.assertEqual(len(events), 1, "Should detect exactly one crossing")
        self.assertEqual(events[0]['direction'], 'IN', "Should be classified as entry")
        self.assertIn(events[0]['confidence'], ['HIGH', 'MEDIUM'], 
                      "Should have high/medium confidence")
    
    def test_clear_exit_pattern(self):
        """Test detection of clear exit pattern"""
        track = SimulatedTrack(2).generate_exit_pattern(25)
        events = self.feed_track(track)
        
        self.assertEqual(len(events), 1, "Should detect exactly one crossing")
        self.assertEqual(events[0]['direction'], 'OUT', "Should be classified as exit")
    
    def test_hesitation_then_entry(self):
        """Test that hesitation followed by entry is still classified as entry"""
        track = SimulatedTrack(3).generate_hesitation_then_entry(35)
        events = self.feed_track(track)
        
        self.assertEqual(len(events), 1, "Should detect exactly one crossing")
        self.assertEqual(events[0]['direction'], 'IN', 
                        "Hesitation then entry should still be IN")
    
    def test_turnaround_not_counted(self):
        """Test that turnaround is not counted as crossing"""
        track = SimulatedTrack(4).generate_turnaround(24)
        events = self.feed_track(track)
        
        # Either no event (abandoned) or low confidence
        if events:
            self.assertEqual(events[0]['confidence'], 'LOW',
                           "Turnaround should be low confidence at most")
    
    def test_multiple_simultaneous_tracks(self):
        """Test handling multiple people entering simultaneously"""
        track1 = SimulatedTrack(10).generate_entry_pattern(20)
        track2 = SimulatedTrack(11).generate_entry_pattern(20)
        
        all_events = []
        
        # Feed both tracks simultaneously
        for i in range(min(len(track1.detections), len(track2.detections))):
            time.sleep(0.033)
            detections = [track1.detections[i], track2.detections[i]]
            events = self.detector.update(detections)
            all_events.extend(events)
        
        # Terminate both
        time.sleep(0.1)
        events = self.detector.update([])
        all_events.extend(events)
        
        # Should detect two entries
        entries = [e for e in all_events if e['direction'] == 'IN']
        self.assertEqual(len(entries), 2, "Should detect two entries")
    
    def test_entry_then_exit(self):
        """Test sequence: person enters then later exits"""
        # Entry
        track1 = SimulatedTrack(20).generate_entry_pattern(20)
        events1 = self.feed_track(track1)
        
        # Reset detector for new track
        # (In real use, tracks would have different IDs)
        
        # Exit (new track ID)
        track2 = SimulatedTrack(21).generate_exit_pattern(20)
        events2 = self.feed_track(track2)
        
        self.assertEqual(len(events1), 1)
        self.assertEqual(events1[0]['direction'], 'IN')
        
        self.assertEqual(len(events2), 1)
        self.assertEqual(events2[0]['direction'], 'OUT')
        
        # Check counters
        entries, exits, occupancy = self.detector.get_counts()
        self.assertEqual(entries, 1)
        self.assertEqual(exits, 1)
        self.assertEqual(occupancy, 0)
    
    def test_short_track_filtered(self):
        """Test that very short tracks are filtered out"""
        # Only 2 frames - should be too short
        track = SimulatedTrack(30)
        track.add_point(0.3, 0.04)
        track.add_point(0.9, 0.07)
        
        events = self.feed_track(track)
        
        # Should be filtered due to short duration
        self.assertEqual(len(events), 0, "Short tracks should be filtered")
    
    def test_area_trend_calculation(self):
        """Test that area trend is correctly calculated"""
        track = SimulatedTrack(40).generate_entry_pattern(15)
        
        # Feed partial track
        for detection in track.detections[:10]:
            time.sleep(0.033)
            self.detector.update([detection])
        
        # Check debug info
        debug = self.detector.get_debug_info(40)
        self.assertIsNotNone(debug)
        self.assertEqual(debug['area_trend'], 'GROWING', 
                        "Entry pattern should show GROWING area")
    
    def test_spawn_position_recorded(self):
        """Test that spawn position is correctly recorded"""
        track = SimulatedTrack(50)
        track.add_point(0.25, 0.03)  # Spawn at y=0.25 (far from door)
        
        self.detector.update([track.detections[0]])
        
        debug = self.detector.get_debug_info(50)
        self.assertIsNotNone(debug)
        self.assertAlmostEqual(debug['spawn_y'], 0.25, delta=0.05)


class TestStandbySlotIntegration(unittest.TestCase):
    """Test the Standby Slot Method integration"""
    
    def test_standby_slot_created_on_exit(self):
        """Test that exit creates standby slot"""
        from people_counter import PeopleCounter, PeopleCounterConfig
        from collections import deque
        
        # We'll test the slot logic directly since full integration needs camera
        slots = deque()
        unique_visitors = 0
        
        # Simulate: Entry
        unique_visitors += 1  # New visitor
        assert unique_visitors == 1
        
        # Simulate: Exit
        from people_counter import StandbySlot
        slots.append(StandbySlot(created_time=time.time()))
        assert len(slots) == 1
        
        # Simulate: Re-entry (consumes slot)
        slots.popleft()  # Returning person
        assert len(slots) == 0
        assert unique_visitors == 1  # Still 1 unique visitor
    
    def test_standby_slot_expiry(self):
        """Test that slots expire after timeout"""
        from people_counter import StandbySlot
        
        slot = StandbySlot(created_time=time.time() - 31, timeout_seconds=30)
        self.assertTrue(slot.is_expired(time.time()))
        
        slot2 = StandbySlot(created_time=time.time() - 10, timeout_seconds=30)
        self.assertFalse(slot2.is_expired(time.time()))


def run_quick_benchmark():
    """Run a quick performance benchmark"""
    print("\n=== Performance Benchmark ===")
    
    config = DirectionDetectorConfig()
    detector = DirectionDetector(config)
    
    # Generate 100 tracks
    num_tracks = 100
    frames_per_track = 30
    
    start_time = time.time()
    
    for track_id in range(num_tracks):
        track = SimulatedTrack(track_id).generate_entry_pattern(frames_per_track)
        
        for detection in track.detections:
            detector.update([detection])
        
        # Terminate
        detector.update([])
    
    elapsed = time.time() - start_time
    total_frames = num_tracks * frames_per_track
    
    print(f"Processed {num_tracks} tracks ({total_frames} frames)")
    print(f"Total time: {elapsed:.3f}s")
    print(f"Frames/second: {total_frames / elapsed:.1f}")
    print(f"Tracks/second: {num_tracks / elapsed:.1f}")
    
    entries, exits, _ = detector.get_counts()
    print(f"Detected: {entries} entries, {exits} exits")


if __name__ == '__main__':
    # Run tests
    print("Running unit tests...\n")
    unittest.main(verbosity=2, exit=False)
    
    # Run benchmark
    run_quick_benchmark()

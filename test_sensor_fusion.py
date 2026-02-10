"""
Unit Tests for Sensor Fusion

Tests fusion logic without requiring camera or ESP32 hardware.
Simulates camera events and ESP32 sensor events to verify fusion accuracy.

Run with: python -m pytest test_sensor_fusion.py -v
Or simply: python test_sensor_fusion.py
"""

import unittest
import time
from sensor_fusion import (
    SensorFusionEngine,
    CameraEvent,
    ESPEvent,
    SensorEvent,
    FusedDecision,
    CameraWithConfirmation,
    TieBreakerStrategy,
    CrossingConfirmationStrategy,
    CONFIDENCE_MAP,
    create_camera_event_from_detector_event
)


# ============================================================================
# MOCK CLASSES (test without hardware)
# ============================================================================

class MockESP32Reader:
    """Mock ESP32 reader that returns pre-configured events"""
    
    def __init__(self):
        self.events = []
        self.on_event = None
        self._is_running = False
    
    def start(self):
        self._is_running = True
    
    def stop(self):
        self._is_running = False
    
    def connect(self):
        return True
    
    def add_event(self, event_type: SensorEvent, confidence: float = 0.8, 
                  source: str = "test", extra: str = ""):
        """Add a simulated sensor event"""
        event = ESPEvent(
            event_type=event_type,
            confidence=confidence,
            source=source,
            extra=extra,
            timestamp=time.time()
        )
        self.events.append(event)
        if self.on_event:
            self.on_event(event)
    
    def get_recent_events(self, window_ms: float = 500):
        cutoff = time.time() - (window_ms / 1000)
        return [e for e in self.events if e.timestamp > cutoff]
    
    def clear_events(self):
        self.events.clear()
    
    def send_command(self, cmd: str):
        pass


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_in_event(track_id: int = 1, confidence: str = "HIGH") -> CameraEvent:
    """Create simulated camera IN event (matches DirectionDetector output)"""
    return CameraEvent(
        track_id=track_id,
        direction='IN',
        confidence=confidence,
        duration_ms=500,
        spawn_y=0.2,
        termination_y=0.9,
        area_change_ratio=1.5,
        area_trend=5,
        y_trend=5
    )


def create_out_event(track_id: int = 1, confidence: str = "HIGH") -> CameraEvent:
    """Create simulated camera OUT event"""
    return CameraEvent(
        track_id=track_id,
        direction='OUT',
        confidence=confidence,
        duration_ms=500,
        spawn_y=0.85,
        termination_y=0.25,
        area_change_ratio=0.6,
        area_trend=-5,
        y_trend=-5
    )


def create_conflict_event(track_id: int = 1) -> CameraEvent:
    """Create camera event with conflicting signals"""
    return CameraEvent(
        track_id=track_id,
        direction='IN',
        confidence='LOW',
        duration_ms=300,
        spawn_y=0.5,
        termination_y=0.6,
        area_change_ratio=1.1,
        area_trend=2,
        y_trend=-1  # Conflict: area growing but Y going wrong way
    )


# ============================================================================
# TEST CASES
# ============================================================================

class TestCameraWithConfirmation(unittest.TestCase):
    """Test the default confirmation strategy"""
    
    def setUp(self):
        self.strategy = CameraWithConfirmation()
    
    def test_camera_only_in(self):
        """Camera detects IN with no sensor data"""
        event = create_in_event()
        decision = self.strategy.fuse(event, [])
        
        self.assertEqual(decision.direction, 'IN')
        self.assertEqual(decision.source, 'camera_only')
        self.assertLess(decision.confidence, event.confidence_float)
    
    def test_camera_only_out(self):
        """Camera detects OUT with no sensor data"""
        event = create_out_event()
        decision = self.strategy.fuse(event, [])
        
        self.assertEqual(decision.direction, 'OUT')
        self.assertEqual(decision.source, 'camera_only')
    
    def test_sensor_confirms_in(self):
        """Sensor confirms camera's IN detection"""
        camera_event = create_in_event()
        sensor_events = [
            ESPEvent(SensorEvent.RADAR_ENTRY, 0.8, "radar_tripwire", "")
        ]
        
        decision = self.strategy.fuse(camera_event, sensor_events)
        
        self.assertEqual(decision.direction, 'IN')
        self.assertEqual(decision.source, 'sensor_confirmed')
        self.assertGreater(decision.confidence, camera_event.confidence_float)
    
    def test_sensor_confirms_out(self):
        """Sensor confirms camera's OUT detection"""
        camera_event = create_out_event()
        sensor_events = [
            ESPEvent(SensorEvent.TOF_EXIT, 0.9, "tof_zones", "")
        ]
        
        decision = self.strategy.fuse(camera_event, sensor_events)
        
        self.assertEqual(decision.direction, 'OUT')
        self.assertEqual(decision.source, 'sensor_confirmed')
    
    def test_sensor_conflicts(self):
        """Sensor says OUT but camera says IN"""
        camera_event = create_in_event()
        sensor_events = [
            ESPEvent(SensorEvent.RADAR_EXIT, 0.7, "radar_tripwire", "")
        ]
        
        decision = self.strategy.fuse(camera_event, sensor_events)
        
        self.assertEqual(decision.direction, 'IN')  # Still trusts camera
        self.assertEqual(decision.source, 'sensor_conflict')
        self.assertLess(decision.confidence, camera_event.confidence_float)
    
    def test_multiple_sensor_events(self):
        """Multiple sensor events - uses highest confidence"""
        camera_event = create_in_event()
        sensor_events = [
            ESPEvent(SensorEvent.TOF_TRIGGERED, 0.5, "tof_tripwire", ""),
            ESPEvent(SensorEvent.RADAR_MOTION, 0.6, "radar_tripwire", ""),
            ESPEvent(SensorEvent.RADAR_ENTRY, 0.85, "radar_tripwire", ""),
        ]
        
        decision = self.strategy.fuse(camera_event, sensor_events)
        
        self.assertEqual(decision.source, 'sensor_confirmed')
        self.assertEqual(decision.sensor_confidence, 0.85)


class TestTieBreakerStrategy(unittest.TestCase):
    """Test the tiebreaker strategy"""
    
    def setUp(self):
        self.strategy = TieBreakerStrategy()
    
    def test_camera_confident_ignores_sensor(self):
        """High-confidence camera ignores conflicting sensor"""
        camera_event = create_in_event(confidence="HIGH")
        sensor_events = [
            ESPEvent(SensorEvent.RADAR_EXIT, 0.9, "radar_tripwire", "")
        ]
        
        decision = self.strategy.fuse(camera_event, sensor_events)
        
        self.assertEqual(decision.direction, 'IN')
        self.assertEqual(decision.source, 'camera_confident')
    
    def test_camera_conflict_uses_sensor(self):
        """Conflicting camera signals → sensor tiebreaks"""
        camera_event = create_conflict_event()
        sensor_events = [
            ESPEvent(SensorEvent.RADAR_EXIT, 0.75, "radar_tripwire", "")
        ]
        
        decision = self.strategy.fuse(camera_event, sensor_events)
        
        self.assertEqual(decision.direction, 'OUT')  # Sensor wins
        self.assertEqual(decision.source, 'sensor_tiebreaker')
    
    def test_camera_conflict_no_sensor(self):
        """Conflicting camera but no sensor → uncertain"""
        camera_event = create_conflict_event()
        
        decision = self.strategy.fuse(camera_event, [])
        
        self.assertEqual(decision.direction, 'IN')
        self.assertEqual(decision.source, 'camera_uncertain')
        self.assertLess(decision.confidence, 0.4)


class TestCrossingConfirmationStrategy(unittest.TestCase):
    """Test the crossing confirmation strategy"""
    
    def setUp(self):
        self.strategy = CrossingConfirmationStrategy()
    
    def test_no_crossing_returns_none(self):
        """No crossing event → returns None"""
        camera_event = create_in_event()
        sensor_events = [
            ESPEvent(SensorEvent.RADAR_MOTION, 0.5, "radar_tripwire", "")
        ]
        
        decision = self.strategy.fuse(camera_event, sensor_events)
        
        self.assertIsNone(decision)
    
    def test_tof_triggered_confirms(self):
        """ToF trigger confirms crossing"""
        camera_event = create_in_event()
        sensor_events = [
            ESPEvent(SensorEvent.TOF_TRIGGERED, 0.8, "tof_tripwire", "")
        ]
        
        decision = self.strategy.fuse(camera_event, sensor_events)
        
        self.assertIsNotNone(decision)
        self.assertEqual(decision.direction, 'IN')
        self.assertEqual(decision.source, 'crossing_confirmed')
    
    def test_radar_entry_confirms(self):
        """Radar entry confirms crossing"""
        camera_event = create_out_event()
        sensor_events = [
            ESPEvent(SensorEvent.RADAR_EXIT, 0.7, "radar_tripwire", "")
        ]
        
        decision = self.strategy.fuse(camera_event, sensor_events)
        
        self.assertIsNotNone(decision)
        self.assertEqual(decision.direction, 'OUT')


class TestSensorFusionEngine(unittest.TestCase):
    """Test the main fusion engine"""
    
    def setUp(self):
        self.engine = SensorFusionEngine(port="MOCK", strategy="confirmation")
        self.engine.esp32 = MockESP32Reader()
        self.engine.esp32.start()
    
    def tearDown(self):
        self.engine.stop()
    
    def test_count_tracking(self):
        """Verify IN/OUT counts tracked correctly"""
        # IN
        self.engine.esp32.add_event(SensorEvent.RADAR_ENTRY, 0.8)
        self.engine.process_camera_event(create_in_event(track_id=1))
        
        self.engine.esp32.clear_events()
        
        # OUT
        self.engine.esp32.add_event(SensorEvent.RADAR_EXIT, 0.8)
        self.engine.process_camera_event(create_out_event(track_id=2))
        
        counts = self.engine.get_counts()
        self.assertEqual(counts['entries'], 1)
        self.assertEqual(counts['exits'], 1)
        self.assertEqual(counts['occupancy'], 0)
    
    def test_strategy_switching(self):
        """Verify strategy can be switched"""
        self.assertIsInstance(self.engine.strategy, CameraWithConfirmation)
        
        self.engine.set_strategy("tiebreaker")
        self.assertIsInstance(self.engine.strategy, TieBreakerStrategy)
        
        self.engine.set_strategy("crossing")
        self.assertIsInstance(self.engine.strategy, CrossingConfirmationStrategy)
    
    def test_decision_callback(self):
        """Verify callback invoked"""
        decisions = []
        self.engine.on_decision = lambda d: decisions.append(d)
        
        self.engine.process_camera_event(create_in_event())
        
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].direction, 'IN')


class TestIntegrationHelpers(unittest.TestCase):
    """Test integration with DirectionDetector"""
    
    def test_create_from_detector_event(self):
        """Test creating CameraEvent from DirectionDetector output"""
        detector_event = {
            'track_id': 42,
            'direction': 'IN',
            'confidence': 'HIGH',
            'duration_ms': 750.5,
            'spawn_y': 0.25,
            'termination_y': 0.88,
            'area_change_ratio': 1.45
        }
        
        camera_event = create_camera_event_from_detector_event(detector_event)
        
        self.assertEqual(camera_event.track_id, 42)
        self.assertEqual(camera_event.direction, 'IN')
        self.assertEqual(camera_event.confidence, 'HIGH')
        self.assertAlmostEqual(camera_event.confidence_float, 0.9)
        self.assertAlmostEqual(camera_event.spawn_y, 0.25)
        self.assertAlmostEqual(camera_event.termination_y, 0.88)
    
    def test_confidence_mapping(self):
        """Verify confidence string to float mapping"""
        self.assertEqual(CONFIDENCE_MAP['HIGH'], 0.9)
        self.assertEqual(CONFIDENCE_MAP['MEDIUM'], 0.7)
        self.assertEqual(CONFIDENCE_MAP['LOW'], 0.5)


# ============================================================================
# SCENARIO TESTS
# ============================================================================

class TestRealWorldScenarios(unittest.TestCase):
    """Test realistic usage scenarios"""
    
    def setUp(self):
        self.engine = SensorFusionEngine(port="MOCK", strategy="confirmation")
        self.engine.esp32 = MockESP32Reader()
        self.engine.esp32.start()
    
    def tearDown(self):
        self.engine.stop()
    
    def test_group_entry(self):
        """3 people enter together - camera counts, sensor confirms"""
        for i in range(3):
            self.engine.esp32.add_event(SensorEvent.TOF_TRIGGERED, 0.9)
            self.engine.process_camera_event(create_in_event(track_id=i))
            time.sleep(0.01)
        
        counts = self.engine.get_counts()
        self.assertEqual(counts['entries'], 3)
        self.assertEqual(counts['occupancy'], 3)
    
    def test_turnback_with_crossing_strategy(self):
        """Person approaches then turns back (crossing strategy)"""
        self.engine.set_strategy("crossing")
        
        # Camera sees IN but no sensor crossing
        decision = self.engine.process_camera_event(create_in_event())
        
        self.assertIsNone(decision)
        self.assertEqual(self.engine.get_counts()['entries'], 0)


# ============================================================================
# BENCHMARK
# ============================================================================

def run_benchmark():
    """Quick performance test"""
    print("\n=== Fusion Benchmark ===")
    
    engine = SensorFusionEngine(port="MOCK", strategy="confirmation")
    engine.esp32 = MockESP32Reader()
    engine.esp32.start()
    
    num_events = 1000
    start = time.time()
    
    for i in range(num_events):
        if i % 2 == 0:
            engine.esp32.add_event(SensorEvent.RADAR_ENTRY, 0.8)
            engine.process_camera_event(create_in_event(track_id=i))
        else:
            engine.esp32.add_event(SensorEvent.RADAR_EXIT, 0.8)
            engine.process_camera_event(create_out_event(track_id=i))
        engine.esp32.clear_events()
    
    elapsed = time.time() - start
    
    print(f"Processed {num_events} events in {elapsed:.3f}s")
    print(f"Rate: {num_events/elapsed:.1f} events/sec")
    print(f"Final counts: {engine.get_counts()}")
    
    engine.stop()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  Sensor Fusion Unit Tests")
    print("="*60 + "\n")
    
    unittest.main(verbosity=2, exit=False)
    run_benchmark()

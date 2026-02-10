"""
Sensor Fusion Module - Camera + ESP32 (Radar/ToF)

Integrates with existing:
  - direction_detector.py (DirectionDetector, TrackHistory, CrossingConfidence)
  - people_counter.py (PeopleCounter, StandbySlot)

Architecture:
  Camera (YOLOv8/ByteTrack) → DirectionDetector → events → Fusion
  ESP32 (Serial) → sensor events → Fusion
                                      ↓
                              Final count decision

ESP32 Event Format: EVT,<type>,<confidence>,<source>,<extra>
Event Types:
  1 = RADAR_ENTRY
  2 = RADAR_EXIT
  3 = RADAR_MOTION
  4 = TOF_TRIGGERED
  5 = TOF_CLEARED
  6 = TOF_ENTRY
  7 = TOF_EXIT
  8 = CROSSING_CONFIRMED

Version: 1.0.0 - Aligned with people_counter.py v1.1.0
"""

import serial
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Deque
from collections import deque
from enum import IntEnum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SensorFusion")


# ============================================================================
# EVENT DEFINITIONS (match Arduino direction_sensor_modular.ino)
# ============================================================================

class SensorEvent(IntEnum):
    NONE = 0
    RADAR_ENTRY = 1
    RADAR_EXIT = 2
    RADAR_MOTION = 3
    TOF_TRIGGERED = 4
    TOF_CLEARED = 5
    TOF_ENTRY = 6
    TOF_EXIT = 7
    CROSSING_CONFIRMED = 8


@dataclass
class ESPEvent:
    """Parsed event from ESP32"""
    event_type: SensorEvent
    confidence: float
    source: str
    extra: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class CameraEvent:
    """
    Event from camera system.
    
    Matches DirectionDetector.update() output format:
    {
        "track_id": int,
        "direction": "IN" or "OUT",
        "confidence": "HIGH"/"MEDIUM"/"LOW" (string),
        "duration_ms": float,
        "spawn_y": float,
        "termination_y": float,
        "area_change_ratio": float
    }
    """
    track_id: int
    direction: str  # "IN" or "OUT" (matches DirectionDetector)
    confidence: str  # "HIGH", "MEDIUM", "LOW" (string, not enum)
    duration_ms: float = 0.0
    spawn_y: float = 0.5  # 0=top (far), 1=bottom (near door)
    termination_y: float = 0.5
    area_change_ratio: float = 1.0
    
    # Derived trend indicators (from TrackHistory if available)
    area_trend: int = 0  # positive = growing, negative = shrinking
    y_trend: int = 0     # positive = moving down (toward door)
    
    timestamp: float = field(default_factory=time.time)
    
    @property
    def confidence_float(self) -> float:
        """Convert string confidence to float"""
        return CONFIDENCE_MAP.get(self.confidence, 0.5)


@dataclass
class FusedDecision:
    """Final fused counting decision"""
    direction: str  # "IN" or "OUT" (matches DirectionDetector format)
    count: int
    confidence: float
    camera_confidence: float
    sensor_confidence: float
    source: str  # 'camera_only', 'sensor_confirmed', 'sensor_conflict', etc.
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# CONFIDENCE MAPPING (matches DirectionDetector CrossingConfidence)
# ============================================================================

CONFIDENCE_MAP = {
    'HIGH': 0.9,
    'MEDIUM': 0.7,
    'LOW': 0.5,
}

# ============================================================================
# DIRECTION FLIP HELPER
# ============================================================================

def flip_direction(direction: str) -> str:
    """Swap IN <-> OUT"""
    if direction == "IN":
        return "OUT"
    elif direction == "OUT":
        return "IN"
    return direction


# ============================================================================
# ESP32 SERIAL READER
# ============================================================================

class ESP32Reader:
    """Reads and parses events from ESP32 via Serial"""
    
    def __init__(self, port: str = "COM3", baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.serial: Optional[serial.Serial] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Event buffer (last N events)
        self.events: Deque[ESPEvent] = deque(maxlen=100)
        
        # Callbacks
        self.on_event: Optional[Callable[[ESPEvent], None]] = None
        
    def connect(self) -> bool:
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.1
            )
            logger.info(f"Connected to ESP32 on {self.port}")
            return True
        except serial.SerialException as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    def start(self):
        if not self.serial:
            if not self.connect():
                return
        
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        logger.info("ESP32 reader started")
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.serial:
            self.serial.close()
        logger.info("ESP32 reader stopped")
    
    def _read_loop(self):
        while self.running:
            try:
                if self.serial and self.serial.in_waiting:
                    line = self.serial.readline().decode('utf-8').strip()
                    if line.startswith("EVT,"):
                        event = self._parse_event(line)
                        if event:
                            self.events.append(event)
                            if self.on_event:
                                self.on_event(event)
                    elif line.startswith("RAW,") or line.startswith("CORRELATION,"):
                        logger.debug(f"ESP32: {line}")
            except Exception as e:
                logger.error(f"Read error: {e}")
                time.sleep(0.1)
    
    def _parse_event(self, line: str) -> Optional[ESPEvent]:
        """Parse: EVT,<type>,<confidence>,<source>,<extra>"""
        try:
            parts = line.split(",")
            if len(parts) >= 4:
                return ESPEvent(
                    event_type=SensorEvent(int(parts[1])),
                    confidence=float(parts[2]),
                    source=parts[3],
                    extra=parts[4] if len(parts) > 4 else ""
                )
        except (ValueError, IndexError) as e:
            logger.warning(f"Parse error for '{line}': {e}")
        return None
    
    def send_command(self, cmd: str):
        """Send command to ESP32"""
        if self.serial:
            self.serial.write(f"{cmd}\n".encode())
            logger.debug(f"Sent: {cmd}")
    
    def get_recent_events(self, window_ms: float = 500) -> List[ESPEvent]:
        """Get events within time window"""
        cutoff = time.time() - (window_ms / 1000)
        return [e for e in self.events if e.timestamp > cutoff]
    
    def clear_events(self):
        """Clear event buffer"""
        self.events.clear()


# ============================================================================
# FUSION STRATEGIES
# ============================================================================

class FusionStrategy:
    """Base class for fusion strategies"""
    
    def fuse(self, camera_event: CameraEvent, 
             sensor_events: List[ESPEvent]) -> FusedDecision:
        raise NotImplementedError


class CameraWithConfirmation(FusionStrategy):
    """
    Camera-primary: Camera makes decision, sensors confirm/boost confidence.
    
    - Camera + sensor agree → HIGH confidence
    - Camera only → MEDIUM confidence (slight penalty)
    - Camera + sensor disagree → LOW confidence (flag for review)
    """
    
    def __init__(self, agreement_boost: float = 0.15, 
                 disagreement_penalty: float = 0.3,
                 no_sensor_penalty: float = 0.1):
        self.agreement_boost = agreement_boost
        self.disagreement_penalty = disagreement_penalty
        self.no_sensor_penalty = no_sensor_penalty
    
    def fuse(self, camera_event: CameraEvent,
             sensor_events: List[ESPEvent]) -> FusedDecision:
        
        sensor_dir = self._get_sensor_direction(sensor_events)
        sensor_conf = self._get_sensor_confidence(sensor_events)
        
        direction = camera_event.direction
        base_conf = camera_event.confidence_float
        
        if sensor_dir is None:
            return FusedDecision(
                direction=direction,
                count=1,
                confidence=base_conf - self.no_sensor_penalty,
                camera_confidence=base_conf,
                sensor_confidence=0.0,
                source='camera_only'
            )
        
        # Map sensor direction to camera format ("IN"/"OUT")
        sensor_dir_mapped = "IN" if sensor_dir == "entry" else "OUT"
        
        if sensor_dir_mapped == direction:
            final_conf = min(1.0, base_conf + self.agreement_boost * sensor_conf)
            return FusedDecision(
                direction=direction,
                count=1,
                confidence=final_conf,
                camera_confidence=base_conf,
                sensor_confidence=sensor_conf,
                source='sensor_confirmed'
            )
        else:
            final_conf = max(0.1, base_conf - self.disagreement_penalty)
            return FusedDecision(
                direction=direction,
                count=1,
                confidence=final_conf,
                camera_confidence=base_conf,
                sensor_confidence=sensor_conf,
                source='sensor_conflict'
            )
    
    def _get_sensor_direction(self, events: List[ESPEvent]) -> Optional[str]:
        for e in reversed(events):
            if e.event_type in (SensorEvent.RADAR_ENTRY, SensorEvent.TOF_ENTRY):
                return 'entry'
            if e.event_type in (SensorEvent.RADAR_EXIT, SensorEvent.TOF_EXIT):
                return 'exit'
        return None
    
    def _get_sensor_confidence(self, events: List[ESPEvent]) -> float:
        relevant = [e for e in events if e.event_type in 
                   (SensorEvent.RADAR_ENTRY, SensorEvent.RADAR_EXIT,
                    SensorEvent.TOF_ENTRY, SensorEvent.TOF_EXIT)]
        return max((e.confidence for e in relevant), default=0.0)


class TieBreakerStrategy(FusionStrategy):
    """
    Camera-primary: Only invoke sensors when camera signals conflict.
    """
    
    def __init__(self, camera_confident_threshold: float = 0.7):
        self.camera_confident_threshold = camera_confident_threshold
    
    def fuse(self, camera_event: CameraEvent,
             sensor_events: List[ESPEvent]) -> FusedDecision:
        
        area_says_entry = camera_event.area_trend > 0
        y_says_entry = camera_event.y_trend > 0
        signals_agree = (area_says_entry == y_says_entry)
        camera_conf = camera_event.confidence_float
        
        if signals_agree or camera_conf >= self.camera_confident_threshold:
            return FusedDecision(
                direction=camera_event.direction,
                count=1,
                confidence=camera_conf,
                camera_confidence=camera_conf,
                sensor_confidence=0.0,
                source='camera_confident'
            )
        
        sensor_dir = self._get_sensor_direction(sensor_events)
        sensor_conf = self._get_sensor_confidence(sensor_events)
        
        if sensor_dir:
            direction = "IN" if sensor_dir == 'entry' else "OUT"
            return FusedDecision(
                direction=direction,
                count=1,
                confidence=sensor_conf,
                camera_confidence=camera_conf,
                sensor_confidence=sensor_conf,
                source='sensor_tiebreaker'
            )
        
        return FusedDecision(
            direction=camera_event.direction,
            count=1,
            confidence=camera_conf * 0.5,
            camera_confidence=camera_conf,
            sensor_confidence=0.0,
            source='camera_uncertain'
        )
    
    def _get_sensor_direction(self, events: List[ESPEvent]) -> Optional[str]:
        for e in reversed(events):
            if e.event_type in (SensorEvent.RADAR_ENTRY, SensorEvent.TOF_ENTRY):
                return 'entry'
            if e.event_type in (SensorEvent.RADAR_EXIT, SensorEvent.TOF_EXIT):
                return 'exit'
        return None
    
    def _get_sensor_confidence(self, events: List[ESPEvent]) -> float:
        relevant = [e for e in events if e.event_type in 
                   (SensorEvent.RADAR_ENTRY, SensorEvent.RADAR_EXIT,
                    SensorEvent.TOF_ENTRY, SensorEvent.TOF_EXIT)]
        return max((e.confidence for e in relevant), default=0.0)


class CrossingConfirmationStrategy(FusionStrategy):
    """
    Wait for ToF/radar crossing detection before committing count.
    Returns None if no crossing confirmed yet.
    """
    
    def __init__(self, require_crossing: bool = True):
        self.require_crossing = require_crossing
    
    def fuse(self, camera_event: CameraEvent,
             sensor_events: List[ESPEvent]) -> Optional[FusedDecision]:
        
        crossing_detected = any(
            e.event_type in (SensorEvent.TOF_TRIGGERED, SensorEvent.TOF_CLEARED,
                            SensorEvent.TOF_ENTRY, SensorEvent.TOF_EXIT,
                            SensorEvent.RADAR_ENTRY, SensorEvent.RADAR_EXIT)
            for e in sensor_events
        )
        
        if not crossing_detected and self.require_crossing:
            return None
        
        camera_conf = camera_event.confidence_float
        sensor_conf = max((e.confidence for e in sensor_events), default=0.5)
        
        return FusedDecision(
            direction=camera_event.direction,
            count=1,
            confidence=min(1.0, camera_conf + 0.15),
            camera_confidence=camera_conf,
            sensor_confidence=sensor_conf,
            source='crossing_confirmed'
        )


# ============================================================================
# MAIN FUSION ENGINE
# ============================================================================

class SensorFusionEngine:
    """
    Main fusion engine coordinating camera and ESP32 sensors.
    """
    
    STRATEGIES = {
        'confirmation': CameraWithConfirmation,
        'tiebreaker': TieBreakerStrategy,
        'crossing': CrossingConfirmationStrategy,
    }
    
    def __init__(self, port: str = "COM3", strategy: str = 'confirmation',
                 sensor_window_ms: float = 500, flip_direction: bool = False):
        self.esp32 = ESP32Reader(port=port)
        self.strategy = self.STRATEGIES[strategy]()
        self.sensor_window_ms = sensor_window_ms
        self.flip_direction = flip_direction  # Swap IN<->OUT
        
        self.decisions: List[FusedDecision] = []
        self.total_entries = 0
        self.total_exits = 0
        
        self.on_decision: Optional[Callable[[FusedDecision], None]] = None
    
    def start(self):
        self.esp32.start()
        logger.info(f"Fusion engine started with strategy: {type(self.strategy).__name__}")
    
    def stop(self):
        self.esp32.stop()
    
    def set_strategy(self, strategy_name: str):
        if strategy_name in self.STRATEGIES:
            self.strategy = self.STRATEGIES[strategy_name]()
            logger.info(f"Switched to strategy: {strategy_name}")
    
    def set_flip(self, flip: bool):
        """Set direction flip (also sends command to ESP32)"""
        self.flip_direction = flip
        self.esp32.send_command(f"flip {'on' if flip else 'off'}")
        logger.info(f"Direction flip: {'ON' if flip else 'OFF'}")
    
    def process_camera_event(self, camera_event: CameraEvent) -> Optional[FusedDecision]:
        # Apply direction flip to camera event if enabled
        if self.flip_direction:
            camera_event.direction = flip_direction(camera_event.direction)
        
        sensor_events = self.esp32.get_recent_events(self.sensor_window_ms)
        decision = self.strategy.fuse(camera_event, sensor_events)
        
        if decision:
            self.decisions.append(decision)
            
            if decision.direction == "IN":
                self.total_entries += decision.count
            else:
                self.total_exits += decision.count
            
            if self.on_decision:
                self.on_decision(decision)
            
            logger.info(f"FUSION: {decision.direction} "
                       f"(conf={decision.confidence:.2f}, src={decision.source})")
        
        return decision
    
    def get_counts(self) -> Dict[str, int]:
        return {
            'entries': self.total_entries,
            'exits': self.total_exits,
            'occupancy': self.total_entries - self.total_exits
        }
    
    def send_esp_command(self, cmd: str):
        self.esp32.send_command(cmd)
    
    def tell_expected_count(self, count: int):
        self.esp32.send_command(f"count {count}")
    
    def start_calibration(self):
        """Enter calibration mode on ESP32 - shows raw coordinate data"""
        self.esp32.send_command("calibrate on")
        logger.info("Calibration mode started - watch for CAL,... messages")
    
    def stop_calibration(self):
        """Exit calibration mode"""
        self.esp32.send_command("calibrate off")
        logger.info("Calibration mode stopped")


# ============================================================================
# INTEGRATION HELPERS FOR people_counter.py
# ============================================================================

def create_camera_event_from_detector_event(event: dict, 
                                            track_history=None) -> CameraEvent:
    """
    Convert DirectionDetector event dict to CameraEvent.
    
    Args:
        event: Dict from DirectionDetector.update():
               {"track_id", "direction", "confidence", "duration_ms", 
                "spawn_y", "termination_y", "area_change_ratio"}
        track_history: Optional TrackHistory for trend data
    """
    area_trend = 0
    y_trend = 0
    
    if track_history is not None:
        area_trend = getattr(track_history, 'area_trend_samples', 0)
        y_trend = getattr(track_history, 'y_trend_samples', 0)
    
    return CameraEvent(
        track_id=event['track_id'],
        direction=event['direction'],
        confidence=event['confidence'],
        duration_ms=event.get('duration_ms', 0.0),
        spawn_y=event.get('spawn_y', 0.5),
        termination_y=event.get('termination_y', 0.5),
        area_change_ratio=event.get('area_change_ratio', 1.0),
        area_trend=area_trend,
        y_trend=y_trend
    )


def get_track_history_for_event(direction_detector, track_id: int):
    """Get TrackHistory from DirectionDetector's completed_crossings."""
    for track in reversed(direction_detector.completed_crossings):
        if track.track_id == track_id:
            return track
    return None


# ============================================================================
# TEST / DEMO
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sensor Fusion Test")
    parser.add_argument("--port", default="COM3", help="ESP32 serial port")
    parser.add_argument("--strategy", default="confirmation", 
                       choices=['confirmation', 'tiebreaker', 'crossing'])
    args = parser.parse_args()
    
    print(f"\n{'='*50}")
    print("  Sensor Fusion Engine - Test Mode")
    print(f"{'='*50}")
    print(f"Port: {args.port}")
    print(f"Strategy: {args.strategy}")
    print("\nCommands:")
    print("  in       - Simulate camera IN event (HIGH)")
    print("  in_low   - Simulate camera IN event (LOW)")
    print("  out      - Simulate camera OUT event")
    print("  status   - Show counts")
    print("  switch <strategy>")
    print("  esp <cmd>")
    print("  quit")
    print(f"{'='*50}\n")
    
    engine = SensorFusionEngine(port=args.port, strategy=args.strategy)
    engine.on_decision = lambda d: print(f"  → {d.direction} conf={d.confidence:.2f} src={d.source}")
    engine.start()
    
    try:
        while True:
            cmd = input("> ").strip().lower()
            
            if cmd == "quit":
                break
            elif cmd == "in":
                engine.process_camera_event(CameraEvent(1, 'IN', 'HIGH', area_trend=5, y_trend=5))
            elif cmd == "in_low":
                engine.process_camera_event(CameraEvent(2, 'IN', 'LOW', area_trend=2, y_trend=-1))
            elif cmd == "out":
                engine.process_camera_event(CameraEvent(3, 'OUT', 'HIGH', area_trend=-5, y_trend=-5))
            elif cmd == "status":
                print(f"  {engine.get_counts()}")
            elif cmd.startswith("switch "):
                engine.set_strategy(cmd.split()[1])
            elif cmd.startswith("esp "):
                engine.send_esp_command(cmd[4:])
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        print(f"\nFinal: {engine.get_counts()}")

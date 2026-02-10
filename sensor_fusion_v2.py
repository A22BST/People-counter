"""
Sensor Fusion Module v2.1 - Camera + ToF (VL53L5CX)

Integrates:
  - direction_detector.py (DirectionDetector, TrackHistory, CrossingConfidence)
  - people_counter.py (PeopleCounter, StandbySlot)
  - ESP32 ToF sensor (dual-zone direction detection)

Architecture:
  Camera (YOLOv8/ByteTrack) → DirectionDetector → crossing events
  ESP32 ToF (Serial)        → zone sequence events
                                    ↓
                            SensorFusionEngine
                                    ↓
                            FusedDecision (count + confidence)

ToF Integration Methods:
  1. CONFIRMATION - ToF boosts/penalizes camera confidence
  2. DUAL_ZONE - ToF provides independent direction signal
  3. HYBRID - Combines both approaches

ESP32 Event Format: EVT,<type>,<confidence>,tof,<extra>
Event Types:
  4 = TOF_TRIGGERED (presence detected)
  5 = TOF_CLEARED (presence ended)
  6 = TOF_ENTRY (outside→inside zone sequence)
  7 = TOF_EXIT (inside→outside zone sequence)

Version: 2.1.0 - ToF dual-zone integration
"""

import serial
import threading
import time
import re
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Deque, Tuple
from collections import deque
from enum import IntEnum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SensorFusion")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class FusionConfig:
    """Configuration for sensor fusion"""
    
    # Serial connection
    port: str = "COM3"
    baudrate: int = 115200
    
    # Timing windows
    sensor_window_ms: float = 800    # Time window to match sensor events to camera
    tof_direction_timeout_ms: float = 1500  # Max age of ToF direction event
    
    # Confidence adjustments
    tof_confirm_boost: float = 0.15   # Boost when ToF confirms camera direction
    tof_conflict_penalty: float = 0.2  # Penalty when ToF conflicts with camera
    
    # Strategy
    strategy: str = "dual_zone"  # "confirmation", "dual_zone", "hybrid"
    
    # Tiebreaker settings
    camera_low_confidence_threshold: float = 0.6  # Below this, prefer ToF
    tof_high_confidence_threshold: float = 0.75   # Above this, trust ToF
    
    # Direction flip
    flip_direction: bool = False


# ============================================================================
# EVENT DEFINITIONS
# ============================================================================

class SensorEvent(IntEnum):
    NONE = 0
    RADAR_ENTRY = 1     # Not used (radar disabled)
    RADAR_EXIT = 2      # Not used
    RADAR_MOTION = 3    # Not used
    TOF_TRIGGERED = 4   # Presence detected
    TOF_CLEARED = 5     # Presence ended
    TOF_ENTRY = 6       # Outside→Inside sequence
    TOF_EXIT = 7        # Inside→Outside sequence
    CROSSING_CONFIRMED = 8


@dataclass
class ToFEvent:
    """Parsed event from ESP32 ToF sensor"""
    event_type: SensorEvent
    confidence: float
    source: str  # "tof"
    extra: str   # "ENTRY", "EXIT", "triggered", "cleared"
    timestamp: float = field(default_factory=time.time)
    
    @property
    def is_direction_event(self) -> bool:
        return self.event_type in (SensorEvent.TOF_ENTRY, SensorEvent.TOF_EXIT)
    
    @property
    def direction(self) -> Optional[str]:
        """Get direction from event type"""
        if self.event_type == SensorEvent.TOF_ENTRY:
            return "IN"
        elif self.event_type == SensorEvent.TOF_EXIT:
            return "OUT"
        return None


@dataclass
class CalibrationData:
    """Parsed calibration data from ESP32"""
    triggered_count: int = 0
    min_distance: int = 0
    column_counts: List[int] = field(default_factory=lambda: [0]*8)
    outside_triggered: int = 0
    inside_triggered: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class CameraEvent:
    """Event from camera direction detector"""
    track_id: int
    direction: str  # "IN" or "OUT"
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    duration_ms: float = 0.0
    spawn_y: float = 0.5
    termination_y: float = 0.5
    area_change_ratio: float = 1.0
    trigger: str = "threshold_crossing"  # or "track_termination"
    timestamp: float = field(default_factory=time.time)
    
    @property
    def confidence_float(self) -> float:
        """Convert string confidence to float"""
        return {"HIGH": 0.9, "MEDIUM": 0.7, "LOW": 0.5}.get(self.confidence, 0.5)


@dataclass
class FusedDecision:
    """Final fused counting decision"""
    direction: str  # "IN" or "OUT"
    count: int = 1
    confidence: float = 0.5
    camera_confidence: float = 0.0
    tof_confidence: float = 0.0
    source: str = "camera_only"  # See sources below
    track_id: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    
    # Possible sources:
    # - camera_only: No ToF data available
    # - tof_confirmed: ToF confirms camera direction
    # - tof_conflict: ToF disagrees (penalized)
    # - tof_tiebreaker: Camera uncertain, ToF decided
    # - tof_only: Camera missed, ToF detected


# ============================================================================
# ESP32 SERIAL READER
# ============================================================================

class ESP32ToFReader:
    """Reads and parses events from ESP32 VL53L5CX sensor"""
    
    # Regex patterns for parsing
    EVT_PATTERN = re.compile(r'^EVT,(\d+),([\d.]+),(\w+),(.*)$')
    CAL_PATTERN = re.compile(
        r'^CAL,TOF,triggered=(\d+),min=(\d+),cols=\[([\d,]+)\],zones=(\d+)\|(\d+)$'
    )
    
    def __init__(self, config: FusionConfig):
        self.config = config
        self.serial: Optional[serial.Serial] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Event buffers
        self.events: Deque[ToFEvent] = deque(maxlen=100)
        self.calibration_data: Optional[CalibrationData] = None
        
        # Callbacks
        self.on_event: Optional[Callable[[ToFEvent], None]] = None
        self.on_calibration: Optional[Callable[[CalibrationData], None]] = None
        
        # Connection state
        self.connected = False
        self.last_error: Optional[str] = None
    
    def connect(self) -> bool:
        """Establish serial connection"""
        try:
            self.serial = serial.Serial(
                port=self.config.port,
                baudrate=self.config.baudrate,
                timeout=0.1
            )
            self.connected = True
            self.last_error = None
            logger.info(f"Connected to ESP32 ToF on {self.config.port}")
            return True
        except serial.SerialException as e:
            self.connected = False
            self.last_error = str(e)
            logger.error(f"Failed to connect: {e}")
            return False
    
    def start(self):
        """Start background reading thread"""
        if not self.serial:
            if not self.connect():
                return False
        
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        logger.info("ESP32 ToF reader started")
        return True
    
    def stop(self):
        """Stop reading and close connection"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.serial:
            self.serial.close()
            self.serial = None
        self.connected = False
        logger.info("ESP32 ToF reader stopped")
    
    def _read_loop(self):
        """Background thread reading serial data"""
        while self.running:
            try:
                if self.serial and self.serial.in_waiting:
                    line = self.serial.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        self._parse_line(line)
            except Exception as e:
                logger.error(f"Read error: {e}")
                time.sleep(0.1)
    
    def _parse_line(self, line: str):
        """Parse a line from ESP32"""
        # Try event format
        match = self.EVT_PATTERN.match(line)
        if match:
            event = ToFEvent(
                event_type=SensorEvent(int(match.group(1))),
                confidence=float(match.group(2)),
                source=match.group(3),
                extra=match.group(4)
            )
            self.events.append(event)
            if self.on_event:
                self.on_event(event)
            return
        
        # Try calibration format
        match = self.CAL_PATTERN.match(line)
        if match:
            cols = [int(x) for x in match.group(3).split(',')]
            self.calibration_data = CalibrationData(
                triggered_count=int(match.group(1)),
                min_distance=int(match.group(2)),
                column_counts=cols,
                outside_triggered=int(match.group(4)),
                inside_triggered=int(match.group(5))
            )
            if self.on_calibration:
                self.on_calibration(self.calibration_data)
            return
        
        # Log other messages
        if line.startswith("OK:") or line.startswith("---"):
            logger.debug(f"ESP32: {line}")
    
    def send_command(self, cmd: str):
        """Send command to ESP32"""
        if self.serial and self.connected:
            self.serial.write(f"{cmd}\n".encode())
            logger.debug(f"Sent: {cmd}")
    
    def get_recent_events(self, window_ms: float = None) -> List[ToFEvent]:
        """Get events within time window"""
        window = window_ms or self.config.sensor_window_ms
        cutoff = time.time() - (window / 1000)
        return [e for e in self.events if e.timestamp > cutoff]
    
    def get_recent_direction(self, window_ms: float = None) -> Optional[ToFEvent]:
        """Get most recent direction event within window"""
        window = window_ms or self.config.tof_direction_timeout_ms
        cutoff = time.time() - (window / 1000)
        
        for event in reversed(self.events):
            if event.timestamp < cutoff:
                break
            if event.is_direction_event:
                return event
        return None
    
    def clear_events(self):
        """Clear event buffer"""
        self.events.clear()


# ============================================================================
# FUSION STRATEGIES
# ============================================================================

class FusionStrategy:
    """Base class for fusion strategies"""
    
    def fuse(self, camera_event: CameraEvent,
             tof_events: List[ToFEvent],
             config: FusionConfig) -> FusedDecision:
        raise NotImplementedError


class ConfirmationStrategy(FusionStrategy):
    """
    ToF confirms or conflicts with camera direction.
    Camera is primary, ToF adjusts confidence.
    """
    
    def fuse(self, camera_event: CameraEvent,
             tof_events: List[ToFEvent],
             config: FusionConfig) -> FusedDecision:
        
        camera_conf = camera_event.confidence_float
        
        # Find matching ToF direction event
        tof_direction_event = None
        for event in reversed(tof_events):
            if event.is_direction_event:
                tof_direction_event = event
                break
        
        if tof_direction_event is None:
            # No ToF direction data - camera only
            return FusedDecision(
                direction=camera_event.direction,
                confidence=camera_conf,
                camera_confidence=camera_conf,
                tof_confidence=0.0,
                source="camera_only",
                track_id=camera_event.track_id
            )
        
        tof_direction = tof_direction_event.direction
        tof_conf = tof_direction_event.confidence
        
        if tof_direction == camera_event.direction:
            # Agreement - boost confidence
            final_conf = min(1.0, camera_conf + config.tof_confirm_boost)
            return FusedDecision(
                direction=camera_event.direction,
                confidence=final_conf,
                camera_confidence=camera_conf,
                tof_confidence=tof_conf,
                source="tof_confirmed",
                track_id=camera_event.track_id
            )
        else:
            # Conflict - penalize or override
            final_conf = max(0.3, camera_conf - config.tof_conflict_penalty)
            return FusedDecision(
                direction=camera_event.direction,  # Still use camera direction
                confidence=final_conf,
                camera_confidence=camera_conf,
                tof_confidence=tof_conf,
                source="tof_conflict",
                track_id=camera_event.track_id
            )


class DualZoneStrategy(FusionStrategy):
    """
    ToF provides independent direction signal via zone sequence.
    Can break ties when camera is uncertain.
    """
    
    def fuse(self, camera_event: CameraEvent,
             tof_events: List[ToFEvent],
             config: FusionConfig) -> FusedDecision:
        
        camera_conf = camera_event.confidence_float
        
        # Find ToF direction event
        tof_direction_event = None
        for event in reversed(tof_events):
            if event.is_direction_event:
                tof_direction_event = event
                break
        
        if tof_direction_event is None:
            # No ToF direction - use camera only
            return FusedDecision(
                direction=camera_event.direction,
                confidence=camera_conf,
                camera_confidence=camera_conf,
                tof_confidence=0.0,
                source="camera_only",
                track_id=camera_event.track_id
            )
        
        tof_direction = tof_direction_event.direction
        tof_conf = tof_direction_event.confidence
        
        # Decision logic based on confidence levels
        if tof_direction == camera_event.direction:
            # Agreement - combine confidences
            final_conf = min(1.0, (camera_conf + tof_conf) / 2 + 0.1)
            return FusedDecision(
                direction=camera_event.direction,
                confidence=final_conf,
                camera_confidence=camera_conf,
                tof_confidence=tof_conf,
                source="tof_confirmed",
                track_id=camera_event.track_id
            )
        
        # Conflict - use tiebreaker logic
        if camera_conf < config.camera_low_confidence_threshold:
            if tof_conf >= config.tof_high_confidence_threshold:
                # Camera uncertain, ToF confident - trust ToF
                return FusedDecision(
                    direction=tof_direction,
                    confidence=tof_conf * 0.9,  # Slight reduction for override
                    camera_confidence=camera_conf,
                    tof_confidence=tof_conf,
                    source="tof_tiebreaker",
                    track_id=camera_event.track_id
                )
        
        # Default: trust camera but note conflict
        return FusedDecision(
            direction=camera_event.direction,
            confidence=max(0.4, camera_conf - config.tof_conflict_penalty),
            camera_confidence=camera_conf,
            tof_confidence=tof_conf,
            source="tof_conflict",
            track_id=camera_event.track_id
        )


class HybridStrategy(FusionStrategy):
    """
    Combines confirmation and dual-zone approaches.
    Uses ToF presence for timing, direction for validation.
    """
    
    def fuse(self, camera_event: CameraEvent,
             tof_events: List[ToFEvent],
             config: FusionConfig) -> FusedDecision:
        
        camera_conf = camera_event.confidence_float
        
        # Check for presence (ToF_TRIGGERED) and direction events
        has_presence = any(e.event_type == SensorEvent.TOF_TRIGGERED for e in tof_events)
        
        tof_direction_event = None
        for event in reversed(tof_events):
            if event.is_direction_event:
                tof_direction_event = event
                break
        
        if not has_presence and tof_direction_event is None:
            # No ToF data at all
            return FusedDecision(
                direction=camera_event.direction,
                confidence=camera_conf,
                camera_confidence=camera_conf,
                tof_confidence=0.0,
                source="camera_only",
                track_id=camera_event.track_id
            )
        
        if has_presence and tof_direction_event is None:
            # Presence detected but no direction - small boost for timing correlation
            return FusedDecision(
                direction=camera_event.direction,
                confidence=min(1.0, camera_conf + 0.05),
                camera_confidence=camera_conf,
                tof_confidence=0.5,
                source="tof_presence",
                track_id=camera_event.track_id
            )
        
        # Have direction event - use dual zone logic
        tof_direction = tof_direction_event.direction
        tof_conf = tof_direction_event.confidence
        
        if tof_direction == camera_event.direction:
            final_conf = min(1.0, camera_conf + config.tof_confirm_boost + 0.05)
            return FusedDecision(
                direction=camera_event.direction,
                confidence=final_conf,
                camera_confidence=camera_conf,
                tof_confidence=tof_conf,
                source="tof_confirmed",
                track_id=camera_event.track_id
            )
        
        # Conflict with both presence and direction
        if camera_conf < config.camera_low_confidence_threshold \
           and tof_conf >= config.tof_high_confidence_threshold:
            return FusedDecision(
                direction=tof_direction,
                confidence=tof_conf * 0.85,
                camera_confidence=camera_conf,
                tof_confidence=tof_conf,
                source="tof_tiebreaker",
                track_id=camera_event.track_id
            )
        
        return FusedDecision(
            direction=camera_event.direction,
            confidence=max(0.35, camera_conf - config.tof_conflict_penalty),
            camera_confidence=camera_conf,
            tof_confidence=tof_conf,
            source="tof_conflict",
            track_id=camera_event.track_id
        )


# ============================================================================
# MAIN FUSION ENGINE
# ============================================================================

class SensorFusionEngine:
    """Main fusion engine coordinating camera and ToF sensor"""
    
    STRATEGIES = {
        'confirmation': ConfirmationStrategy,
        'dual_zone': DualZoneStrategy,
        'hybrid': HybridStrategy,
    }
    
    def __init__(self, config: FusionConfig = None):
        self.config = config or FusionConfig()
        self.tof_reader = ESP32ToFReader(self.config)
        self.strategy = self.STRATEGIES[self.config.strategy]()
        
        # Counters
        self.decisions: Deque[FusedDecision] = deque(maxlen=1000)
        self.total_entries = 0
        self.total_exits = 0
        
        # Callbacks
        self.on_decision: Optional[Callable[[FusedDecision], None]] = None
        self.on_tof_event: Optional[Callable[[ToFEvent], None]] = None
        
        # Wire up ToF callbacks
        self.tof_reader.on_event = self._on_tof_event
    
    def _on_tof_event(self, event: ToFEvent):
        """Handle incoming ToF event"""
        if self.on_tof_event:
            self.on_tof_event(event)
        
        # Log direction events
        if event.is_direction_event:
            logger.info(f"ToF: {event.direction} (conf={event.confidence:.2f})")
    
    def start(self) -> bool:
        """Start the fusion engine"""
        success = self.tof_reader.start()
        if success:
            logger.info(f"Fusion engine started: strategy={self.config.strategy}")
        return success
    
    def stop(self):
        """Stop the fusion engine"""
        self.tof_reader.stop()
    
    @property
    def connected(self) -> bool:
        """Check if ToF sensor is connected"""
        return self.tof_reader.connected
    
    def set_strategy(self, strategy_name: str):
        """Change fusion strategy"""
        if strategy_name in self.STRATEGIES:
            self.strategy = self.STRATEGIES[strategy_name]()
            self.config.strategy = strategy_name
            logger.info(f"Strategy changed to: {strategy_name}")
    
    def set_flip(self, flip: bool):
        """Set direction flip (also notifies ESP32)"""
        self.config.flip_direction = flip
        self.tof_reader.send_command(f"flip {'on' if flip else 'off'}")
        logger.info(f"Direction flip: {'ON' if flip else 'OFF'}")
    
    def process_camera_event(self, camera_event: CameraEvent) -> FusedDecision:
        """
        Process a camera crossing event and fuse with ToF data.
        
        Args:
            camera_event: Event from DirectionDetector
            
        Returns:
            FusedDecision with final direction and confidence
        """
        # Apply direction flip if enabled
        if self.config.flip_direction:
            camera_event.direction = "OUT" if camera_event.direction == "IN" else "IN"
        
        # Get recent ToF events
        tof_events = self.tof_reader.get_recent_events()
        
        # Fuse using selected strategy
        decision = self.strategy.fuse(camera_event, tof_events, self.config)
        
        # Update counters
        self.decisions.append(decision)
        if decision.direction == "IN":
            self.total_entries += decision.count
        else:
            self.total_exits += decision.count
        
        # Clear used ToF events to prevent double-counting
        self.tof_reader.clear_events()
        
        # Notify callback
        if self.on_decision:
            self.on_decision(decision)
        
        logger.info(f"FUSION: {decision.direction} (conf={decision.confidence:.2f}, "
                   f"src={decision.source})")
        
        return decision
    
    def check_tof_only_events(self) -> Optional[FusedDecision]:
        """
        Check for ToF direction events without camera event.
        Call this periodically to catch crossings camera missed.
        
        Returns:
            FusedDecision if ToF detected crossing, None otherwise
        """
        tof_event = self.tof_reader.get_recent_direction(window_ms=500)
        
        if tof_event and tof_event.confidence >= self.config.tof_high_confidence_threshold:
            # ToF saw something camera missed
            direction = tof_event.direction
            if self.config.flip_direction:
                direction = "OUT" if direction == "IN" else "IN"
            
            decision = FusedDecision(
                direction=direction,
                confidence=tof_event.confidence * 0.8,  # Reduce for no camera confirmation
                camera_confidence=0.0,
                tof_confidence=tof_event.confidence,
                source="tof_only"
            )
            
            self.decisions.append(decision)
            if decision.direction == "IN":
                self.total_entries += 1
            else:
                self.total_exits += 1
            
            self.tof_reader.clear_events()
            
            if self.on_decision:
                self.on_decision(decision)
            
            logger.warning(f"TOF-ONLY: {decision.direction} (camera missed)")
            return decision
        
        return None
    
    def get_counts(self) -> Dict[str, int]:
        """Get current counts"""
        return {
            'entries': self.total_entries,
            'exits': self.total_exits,
            'occupancy': max(0, self.total_entries - self.total_exits)
        }
    
    def get_stats(self) -> Dict[str, any]:
        """Get fusion statistics"""
        if not self.decisions:
            return {'total': 0}
        
        sources = {}
        for d in self.decisions:
            sources[d.source] = sources.get(d.source, 0) + 1
        
        avg_conf = sum(d.confidence for d in self.decisions) / len(self.decisions)
        
        return {
            'total': len(self.decisions),
            'avg_confidence': avg_conf,
            'sources': sources,
            'entries': self.total_entries,
            'exits': self.total_exits
        }
    
    def reset_counts(self):
        """Reset all counters"""
        self.total_entries = 0
        self.total_exits = 0
        self.decisions.clear()
        self.tof_reader.clear_events()
    
    def send_command(self, cmd: str):
        """Send raw command to ESP32"""
        self.tof_reader.send_command(cmd)
    
    def start_calibration(self):
        """Start ToF calibration mode"""
        self.tof_reader.send_command("cal on")
        logger.info("Calibration mode ON - watch for CAL,TOF,... messages")
    
    def stop_calibration(self):
        """Stop calibration mode"""
        self.tof_reader.send_command("cal off")


# ============================================================================
# HELPER FUNCTIONS FOR INTEGRATION
# ============================================================================

def create_camera_event(event_dict: dict) -> CameraEvent:
    """
    Convert DirectionDetector event dict to CameraEvent.
    
    Args:
        event_dict: Dict from DirectionDetector.update():
            {"track_id", "direction", "confidence", "trigger", ...}
    """
    return CameraEvent(
        track_id=event_dict.get('track_id', 0),
        direction=event_dict.get('direction', 'IN'),
        confidence=event_dict.get('confidence', 'MEDIUM'),
        duration_ms=event_dict.get('duration_ms', 0.0),
        spawn_y=event_dict.get('spawn_y', 0.5),
        termination_y=event_dict.get('termination_y', 0.5),
        area_change_ratio=event_dict.get('area_change_ratio', 1.0),
        trigger=event_dict.get('trigger', 'threshold_crossing')
    )


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sensor Fusion Test")
    parser.add_argument("--port", default="COM3", help="ESP32 serial port")
    parser.add_argument("--strategy", default="dual_zone",
                       choices=['confirmation', 'dual_zone', 'hybrid'])
    parser.add_argument("--calibrate", action="store_true",
                       help="Start in calibration mode")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("  Sensor Fusion Engine v2.1 - Test Mode")
    print(f"{'='*60}")
    print(f"Port: {args.port}")
    print(f"Strategy: {args.strategy}")
    print("\nCommands:")
    print("  in [high/med/low] - Simulate camera IN event")
    print("  out [high/med/low] - Simulate camera OUT event")
    print("  status            - Show counts and stats")
    print("  switch <strategy> - Change strategy")
    print("  flip              - Toggle direction flip")
    print("  cal               - Toggle calibration mode")
    print("  reset             - Reset counters")
    print("  quit              - Exit")
    print(f"{'='*60}\n")
    
    config = FusionConfig(port=args.port, strategy=args.strategy)
    engine = SensorFusionEngine(config)
    
    def on_decision(d: FusedDecision):
        print(f"  → {d.direction} conf={d.confidence:.2f} src={d.source}")
    
    def on_tof(e: ToFEvent):
        if e.is_direction_event:
            print(f"  [ToF] {e.direction} conf={e.confidence:.2f}")
        elif e.event_type == SensorEvent.TOF_TRIGGERED:
            print(f"  [ToF] Presence detected")
    
    engine.on_decision = on_decision
    engine.on_tof_event = on_tof
    
    if not engine.start():
        print("Failed to connect to ESP32. Running in camera-only mode.")
    
    if args.calibrate:
        engine.start_calibration()
    
    calibrating = args.calibrate
    flip_enabled = False
    track_counter = 1
    
    try:
        while True:
            cmd = input("> ").strip().lower()
            
            if cmd == "quit":
                break
            
            elif cmd.startswith("in"):
                parts = cmd.split()
                conf = "HIGH" if len(parts) < 2 else parts[1].upper()
                event = CameraEvent(track_counter, "IN", conf)
                track_counter += 1
                engine.process_camera_event(event)
            
            elif cmd.startswith("out"):
                parts = cmd.split()
                conf = "HIGH" if len(parts) < 2 else parts[1].upper()
                event = CameraEvent(track_counter, "OUT", conf)
                track_counter += 1
                engine.process_camera_event(event)
            
            elif cmd == "status":
                counts = engine.get_counts()
                stats = engine.get_stats()
                print(f"\n  Counts: {counts}")
                print(f"  Stats: {stats}\n")
            
            elif cmd.startswith("switch"):
                parts = cmd.split()
                if len(parts) > 1:
                    engine.set_strategy(parts[1])
            
            elif cmd == "flip":
                flip_enabled = not flip_enabled
                engine.set_flip(flip_enabled)
            
            elif cmd == "cal":
                calibrating = not calibrating
                if calibrating:
                    engine.start_calibration()
                else:
                    engine.stop_calibration()
            
            elif cmd == "reset":
                engine.reset_counts()
                print("  Counters reset")
            
            elif cmd == "check":
                # Check for ToF-only events
                engine.check_tof_only_events()
    
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        print(f"\nFinal: {engine.get_counts()}")
        print(f"Stats: {engine.get_stats()}")

"""
Direction Detector: Track Lifecycle Based Entry/Exit Detection

Version: 2.0.0 - Threshold Crossing Update

Core Principle:
- Camera mounted on door frame, looking OUTSIDE
- ENTRY: Person crosses threshold line going DOWN (into near zone)
- EXIT: Person crosses threshold line going UP (out of near zone)

Key Change from v1:
- v1: Counted only when track terminated (person left frame)
- v2: Counts immediately when person CROSSES the threshold line
- This fixes: hallway scenario (stays in frame), sitting scenario (never leaves)

Author: Ahmad's Mosque Attendance System
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
import time
import numpy as np


class TrackState(Enum):
    """State machine for track lifecycle"""
    NASCENT = auto()      # Just appeared, insufficient data
    APPROACHING = auto()  # Moving toward camera (bbox growing)
    DEPARTING = auto()    # Moving away from camera (bbox shrinking)
    STABLE = auto()       # Not moving significantly (hesitating/standing)
    CROSSED_IN = auto()   # Confirmed entry (crossed threshold going in)
    CROSSED_OUT = auto()  # Confirmed exit (crossed threshold going out)
    ABANDONED = auto()    # Track lost without crossing (e.g., turned around)


class CrossingConfidence(Enum):
    """Confidence level of crossing detection"""
    LOW = auto()       # Single signal, possible false positive
    MEDIUM = auto()    # Multiple correlated signals
    HIGH = auto()      # All signals agree, high certainty


@dataclass
class TrackHistory:
    """Stores historical data for a single track"""
    track_id: int
    
    # Bounding box history: (timestamp, x1, y1, x2, y2)
    bbox_history: Deque[Tuple[float, float, float, float, float]] = field(
        default_factory=lambda: deque(maxlen=60)  # ~2 seconds at 30fps
    )
    
    # Derived metrics history
    area_history: Deque[Tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=60)
    )
    centroid_y_history: Deque[Tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=60)
    )
    
    # State tracking
    state: TrackState = TrackState.NASCENT
    state_history: List[Tuple[float, TrackState]] = field(default_factory=list)
    
    # Spawn/termination info
    spawn_time: float = 0.0
    spawn_y_normalized: float = 0.0  # 0=top, 1=bottom (near door)
    spawn_area: float = 0.0
    last_update_time: float = 0.0
    termination_y_normalized: Optional[float] = None
    termination_area: Optional[float] = None
    
    # Accumulated evidence
    area_trend_samples: int = 0     # Positive = growing, negative = shrinking
    y_trend_samples: int = 0        # Positive = moving down (toward door)
    
    # ===== NEW: Threshold crossing tracking =====
    has_crossed_threshold: bool = False      # Has this track crossed the threshold?
    crossed_direction: Optional[str] = None  # "IN" or "OUT" when crossed
    crossed_time: Optional[float] = None     # When the crossing happened
    crossed_y: Optional[float] = None        # Y position when crossed
    last_y_normalized: Optional[float] = None  # Previous Y for crossing detection
    
    # State-based crossing detection (above/below threshold with hysteresis)
    threshold_state: Optional[str] = None    # "above", "below", or None (in zone)
    
    # Final determination (may be set at crossing OR termination)
    crossing_direction: Optional[str] = None  # "IN" or "OUT"
    crossing_confidence: Optional[CrossingConfidence] = None
    crossing_time: Optional[float] = None


@dataclass 
class DirectionDetectorConfig:
    """Configuration parameters - tune these for your setup"""
    
    # Frame geometry (camera looking outside, mounted on door frame)
    frame_height: int = 480
    frame_width: int = 640
    
    # Near-edge threshold: bottom portion of frame is "near door"
    # If camera is angled down at doorway, people disappear at bottom
    near_edge_threshold: float = 0.85  # Bottom 15% of frame
    
    # ===== NEW: Threshold crossing settings =====
    count_on_crossing: bool = True       # Count when crossing threshold (vs termination)
    crossing_hysteresis: float = 0.03    # Buffer zone to prevent flickering (3% of frame)
    min_samples_before_crossing: int = 3 # Need at least 3 samples before counting crossing
    
    # Minimum track requirements
    min_track_duration_ms: int = 300      # Must exist for 300ms minimum
    min_samples_for_trend: int = 5        # Need 5+ samples to determine trend
    min_trend_consistency: float = 0.6    # 60% of samples must agree on direction
    
    # Area change thresholds (normalized by initial area)
    significant_area_change: float = 0.15  # 15% change is significant
    area_growth_for_approach: float = 1.2  # 20% larger = approaching
    area_shrink_for_depart: float = 0.8    # 20% smaller = departing
    
    # Y-position movement thresholds (normalized 0-1)
    significant_y_movement: float = 0.1    # 10% of frame height
    
    # Timing thresholds
    track_timeout_ms: int = 500           # Track considered lost after 500ms no update
    hesitation_timeout_ms: int = 2000     # Person standing still for 2s = uncertain
    
    # Spawn position thresholds for initial classification
    spawn_near_threshold: float = 0.7      # Spawning in bottom 30% = likely exiting
    spawn_far_threshold: float = 0.4       # Spawning in top 40% = likely entering
    
    # Confidence thresholds
    high_confidence_trend_ratio: float = 0.8   # 80% trend agreement = high confidence
    
    # Minimum area for valid detection (filter tiny false positives)
    min_bbox_area_ratio: float = 0.01     # At least 1% of frame


class DirectionDetector:
    """
    Detects entry/exit direction based on track lifecycle analysis.
    
    Version 2.0: Counts on threshold crossing instead of track termination.
    
    Usage:
        detector = DirectionDetector(config)
        
        # Each frame, call update with detections
        events = detector.update(detections)
        
        # events is a list of crossing events that happened THIS frame
        # (either threshold crossings or track terminations)
    """
    
    def __init__(self, config: DirectionDetectorConfig = None):
        self.config = config or DirectionDetectorConfig()
        self.tracks: Dict[int, TrackHistory] = {}
        self.completed_crossings: List[TrackHistory] = []
        
        # Counters
        self.entry_count = 0
        self.exit_count = 0
        
        # Frame area for normalization
        self.frame_area = self.config.frame_width * self.config.frame_height
    
    def update(self, detections: List[Tuple[int, Tuple[float, float, float, float]]]) -> List[dict]:
        """
        Process new detections and return any crossing events.
        
        Args:
            detections: List of (track_id, (x1, y1, x2, y2)) tuples
            
        Returns:
            List of crossing events (may be empty)
            Each event: {
                "track_id": int,
                "direction": "IN" or "OUT",
                "confidence": "HIGH"/"MEDIUM"/"LOW",
                "trigger": "threshold_crossing" or "track_termination",
                ...
            }
        """
        timestamp = time.time()
        events = []
        
        current_track_ids = set()
        
        # Process each detection
        for track_id, bbox in detections:
            current_track_ids.add(track_id)
            
            if track_id not in self.tracks:
                self._init_track(track_id, bbox, timestamp)
            else:
                # Update track and check for threshold crossing
                crossing_event = self._update_track(track_id, bbox, timestamp)
                if crossing_event:
                    events.append(crossing_event)
        
        # Process stale/terminated tracks
        termination_events = self._process_stale_tracks(current_track_ids, timestamp)
        events.extend(termination_events)
        
        return events
    
    def _init_track(self, track_id: int, bbox: Tuple[float, float, float, float], timestamp: float):
        """Initialize a new track with spawn information"""
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        centroid_y = (y1 + y2) / 2
        centroid_y_normalized = centroid_y / self.config.frame_height
        area_normalized = area / self.frame_area
        
        # Filter tiny detections
        if area_normalized < self.config.min_bbox_area_ratio:
            return
        
        track = TrackHistory(
            track_id=track_id,
            spawn_time=timestamp,
            spawn_y_normalized=centroid_y_normalized,
            spawn_area=area,
            last_update_time=timestamp,
            last_y_normalized=centroid_y_normalized  # Initialize for crossing detection
        )
        
        track.bbox_history.append((timestamp, x1, y1, x2, y2))
        track.area_history.append((timestamp, area))
        track.centroid_y_history.append((timestamp, centroid_y_normalized))
        track.state_history.append((timestamp, TrackState.NASCENT))
        
        # Initialize threshold state for crossing detection
        threshold = self.config.near_edge_threshold
        hysteresis = self.config.crossing_hysteresis
        if centroid_y_normalized < threshold - hysteresis:
            track.threshold_state = "above"
        elif centroid_y_normalized > threshold + hysteresis:
            track.threshold_state = "below"
        # else: in the zone, state stays None
        
        self.tracks[track_id] = track
    
    def _update_track(self, track_id: int, bbox: Tuple[float, float, float, float], 
                      timestamp: float) -> Optional[dict]:
        """
        Update existing track with new detection.
        Returns crossing event if threshold was crossed.
        """
        track = self.tracks[track_id]
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        centroid_y = (y1 + y2) / 2
        centroid_y_normalized = centroid_y / self.config.frame_height
        
        # Store history
        track.bbox_history.append((timestamp, x1, y1, x2, y2))
        track.area_history.append((timestamp, area))
        track.centroid_y_history.append((timestamp, centroid_y_normalized))
        track.last_update_time = timestamp
        
        # Update state based on accumulated evidence
        self._update_track_state(track, timestamp)
        
        # Check for threshold crossing (if enabled)
        # Note: double-counting prevention is handled inside _check_threshold_crossing
        crossing_event = None
        if self.config.count_on_crossing:
            crossing_event = self._check_threshold_crossing(track, centroid_y_normalized, timestamp)
        
        # Update last Y for next crossing check
        track.last_y_normalized = centroid_y_normalized
        
        return crossing_event
    
    def _check_threshold_crossing(self, track: TrackHistory, current_y: float, 
                                   timestamp: float) -> Optional[dict]:
        """
        Check if track just crossed the threshold line using state-based detection.
        
        State transitions:
        - "above" (y < threshold - hysteresis) → clearly above threshold
        - "below" (y > threshold + hysteresis) → clearly below threshold  
        - None (in hysteresis zone) → transitioning
        
        Crossing detected when state changes from "above" to "below" or vice versa.
        """
        threshold = self.config.near_edge_threshold
        hysteresis = self.config.crossing_hysteresis
        
        # Determine current state based on position
        if current_y < threshold - hysteresis:
            current_state = "above"
        elif current_y > threshold + hysteresis:
            current_state = "below"
        else:
            current_state = None  # In the hysteresis zone
        
        # ALWAYS try to initialize state if we have a clear position and no state yet
        if track.threshold_state is None and current_state is not None:
            track.threshold_state = current_state
            track.last_y_normalized = current_y
            # Don't return yet - continue to check samples requirement
        
        # Need minimum samples before we count a crossing
        if len(track.centroid_y_history) < self.config.min_samples_before_crossing:
            track.last_y_normalized = current_y
            return None
        
        # Check for state transition (crossing)
        crossing_detected = False
        direction = None
        
        if track.threshold_state == "above" and current_state == "below":
            # Crossed from above to below = ENTRY (into near zone)
            crossing_detected = True
            direction = "IN"
        elif track.threshold_state == "below" and current_state == "above":
            # Crossed from below to above = EXIT (out of near zone)
            crossing_detected = True
            direction = "OUT"
        
        # Prevent double-counting in same direction (but allow re-crossing in opposite direction)
        if crossing_detected and track.crossed_direction == direction:
            # Already crossed in this direction, skip
            crossing_detected = False
        
        # Update state (only when we have a clear position, not in zone)
        if current_state is not None:
            track.threshold_state = current_state
        track.last_y_normalized = current_y
        
        if not crossing_detected:
            return None
        
        # Determine confidence based on supporting evidence
        confidence = self._get_crossing_confidence(track, direction)
        
        # Mark track as crossed
        track.has_crossed_threshold = True
        track.crossed_direction = direction
        track.crossed_time = timestamp
        track.crossed_y = current_y
        
        # Also set the final crossing info
        track.crossing_direction = direction
        track.crossing_confidence = confidence
        track.crossing_time = timestamp
        track.state = TrackState.CROSSED_IN if direction == "IN" else TrackState.CROSSED_OUT
        
        # Update counts
        if direction == "IN":
            self.entry_count += 1
        else:
            self.exit_count += 1
        
        # Calculate duration
        duration_ms = (timestamp - track.spawn_time) * 1000
        
        # Get current area ratio
        current_area = track.area_history[-1][1] if track.area_history else track.spawn_area
        area_ratio = current_area / track.spawn_area if track.spawn_area > 0 else 1.0
        
        return {
            "track_id": track.track_id,
            "direction": direction,
            "confidence": confidence.name,
            "trigger": "threshold_crossing",
            "duration_ms": duration_ms,
            "spawn_y": track.spawn_y_normalized,
            "crossing_y": current_y,
            "area_change_ratio": area_ratio,
            "area_trend_samples": track.area_trend_samples,
            "y_trend_samples": track.y_trend_samples
        }
    
    def _get_crossing_confidence(self, track: TrackHistory, direction: str) -> CrossingConfidence:
        """
        Determine confidence level based on supporting evidence.
        """
        signals_supporting = 0
        signals_total = 0
        
        # Signal 1: Area trend matches direction
        if direction == "IN":
            # Entry should have growing area (approaching)
            if track.area_trend_samples > 0:
                signals_supporting += 1
            signals_total += 1
        else:
            # Exit should have shrinking area (departing)
            if track.area_trend_samples < 0:
                signals_supporting += 1
            signals_total += 1
        
        # Signal 2: Y movement matches direction
        if direction == "IN":
            # Entry should be moving down (positive y_trend)
            if track.y_trend_samples > 0:
                signals_supporting += 1
            signals_total += 1
        else:
            # Exit should be moving up (negative y_trend)
            if track.y_trend_samples < 0:
                signals_supporting += 1
            signals_total += 1
        
        # Signal 3: Spawn position consistent with direction
        if direction == "IN":
            # Entry should spawn far (top of frame)
            if track.spawn_y_normalized < self.config.spawn_far_threshold:
                signals_supporting += 1
            signals_total += 1
        else:
            # Exit should spawn near (bottom of frame)
            if track.spawn_y_normalized > self.config.spawn_near_threshold:
                signals_supporting += 1
            signals_total += 1
        
        # Signal 4: State history supports direction
        approaching_count = sum(1 for _, s in track.state_history if s == TrackState.APPROACHING)
        departing_count = sum(1 for _, s in track.state_history if s == TrackState.DEPARTING)
        
        if direction == "IN" and approaching_count > departing_count:
            signals_supporting += 1
        elif direction == "OUT" and departing_count > approaching_count:
            signals_supporting += 1
        signals_total += 1
        
        # Determine confidence
        if signals_total == 0:
            return CrossingConfidence.LOW
        
        ratio = signals_supporting / signals_total
        
        if ratio >= 0.75:
            return CrossingConfidence.HIGH
        elif ratio >= 0.5:
            return CrossingConfidence.MEDIUM
        else:
            return CrossingConfidence.LOW
    
    def _update_track_state(self, track: TrackHistory, timestamp: float):
        """Analyze track history and update state machine"""
        if len(track.area_history) < self.config.min_samples_for_trend:
            return  # Not enough data yet
        
        # Calculate area trend
        areas = [a for _, a in track.area_history]
        area_changes = np.diff(areas)
        growing_samples = np.sum(area_changes > (track.spawn_area * 0.02))  # 2% threshold
        shrinking_samples = np.sum(area_changes < -(track.spawn_area * 0.02))
        
        # Calculate Y movement trend
        y_positions = [y for _, y in track.centroid_y_history]
        y_changes = np.diff(y_positions)
        moving_down_samples = np.sum(y_changes > 0.01)  # Moving toward bottom
        moving_up_samples = np.sum(y_changes < -0.01)   # Moving toward top
        
        total_changes = len(area_changes)
        
        # Accumulate evidence
        if growing_samples > shrinking_samples:
            track.area_trend_samples = int(growing_samples)
        else:
            track.area_trend_samples = -int(shrinking_samples)
            
        if moving_down_samples > moving_up_samples:
            track.y_trend_samples = int(moving_down_samples)
        else:
            track.y_trend_samples = -int(moving_up_samples)
        
        # Don't update state if already crossed
        if track.has_crossed_threshold:
            return
        
        # Determine state based on evidence
        old_state = track.state
        
        # Check for approaching (growing + moving down)
        if (growing_samples / max(total_changes, 1) > self.config.min_trend_consistency and
            moving_down_samples / max(total_changes, 1) > self.config.min_trend_consistency * 0.5):
            track.state = TrackState.APPROACHING
            
        # Check for departing (shrinking + moving up)
        elif (shrinking_samples / max(total_changes, 1) > self.config.min_trend_consistency and
              moving_up_samples / max(total_changes, 1) > self.config.min_trend_consistency * 0.5):
            track.state = TrackState.DEPARTING
            
        # Check for stable (not much movement)
        elif (growing_samples + shrinking_samples) / max(total_changes, 1) < 0.3:
            track.state = TrackState.STABLE
        
        # Record state change
        if track.state != old_state:
            track.state_history.append((timestamp, track.state))
    
    def _process_stale_tracks(self, current_ids: set, timestamp: float) -> List[dict]:
        """
        Handle tracks that are no longer being detected.
        
        For tracks that already crossed threshold: just clean up
        For tracks that never crossed: try to classify on termination (fallback)
        """
        events = []
        stale_ids = []
        
        for track_id, track in self.tracks.items():
            if track_id not in current_ids:
                time_since_update = (timestamp - track.last_update_time) * 1000
                
                if time_since_update > self.config.track_timeout_ms:
                    stale_ids.append(track_id)
                    
                    # If track already crossed threshold, just complete it
                    if track.has_crossed_threshold:
                        self.completed_crossings.append(track)
                    else:
                        # Fallback: try to classify on termination (legacy behavior)
                        if not self.config.count_on_crossing:
                            event = self._terminate_track(track_id, timestamp, timed_out=True)
                            if event:
                                events.append(event)
                        else:
                            # With crossing mode, uncrossed tracks are abandoned
                            track.state = TrackState.ABANDONED
                            self.completed_crossings.append(track)
        
        # Remove stale tracks
        for track_id in stale_ids:
            del self.tracks[track_id]
        
        return events
    
    def _terminate_track(self, track_id: int, timestamp: float, timed_out: bool = False) -> Optional[dict]:
        """
        Terminate a track and determine crossing direction (legacy/fallback mode).
        
        Only used when count_on_crossing is False.
        """
        track = self.tracks.get(track_id)
        if not track:
            return None
        
        # Skip if already crossed (shouldn't happen in normal flow)
        if track.has_crossed_threshold:
            return None
        
        # Check minimum duration
        duration_ms = (timestamp - track.spawn_time) * 1000
        if duration_ms < self.config.min_track_duration_ms:
            track.state = TrackState.ABANDONED
            return None
        
        # Check minimum samples
        if len(track.bbox_history) < self.config.min_samples_for_trend:
            track.state = TrackState.ABANDONED
            return None
        
        # Get termination metrics
        if track.centroid_y_history:
            _, final_y = track.centroid_y_history[-1]
            track.termination_y_normalized = final_y
        if track.area_history:
            _, final_area = track.area_history[-1]
            track.termination_area = final_area
        
        # Determine crossing direction with confidence
        direction, confidence = self._classify_crossing_legacy(track, timed_out)
        
        if direction is None:
            track.state = TrackState.ABANDONED
            self.completed_crossings.append(track)
            return None
        
        # Valid crossing detected
        track.crossing_direction = direction
        track.crossing_confidence = confidence
        track.crossing_time = timestamp
        track.state = TrackState.CROSSED_IN if direction == "IN" else TrackState.CROSSED_OUT
        
        if direction == "IN":
            self.entry_count += 1
        else:
            self.exit_count += 1
        
        self.completed_crossings.append(track)
        
        return {
            "track_id": track_id,
            "direction": direction,
            "confidence": confidence.name,
            "trigger": "track_termination",
            "duration_ms": duration_ms,
            "spawn_y": track.spawn_y_normalized,
            "termination_y": track.termination_y_normalized,
            "area_change_ratio": (track.termination_area / track.spawn_area) if track.spawn_area > 0 else 1.0,
            "area_trend_samples": track.area_trend_samples,
            "y_trend_samples": track.y_trend_samples
        }
    
    def _classify_crossing_legacy(self, track: TrackHistory, timed_out: bool) -> Tuple[Optional[str], Optional[CrossingConfidence]]:
        """
        Legacy classification based on spawn/termination positions (fallback mode).
        """
        signals = {
            "spawn_near": False,
            "spawn_far": False,
            "terminated_near": False,
            "terminated_far": False,
            "was_approaching": False,
            "was_departing": False,
            "area_grew": False,
            "area_shrank": False,
            "ended_near_spawn": False,
        }
        
        # Signal 1 & 2: Spawn and termination positions
        signals["spawn_near"] = track.spawn_y_normalized > self.config.spawn_near_threshold
        signals["spawn_far"] = track.spawn_y_normalized < self.config.spawn_far_threshold
        
        if track.termination_y_normalized is not None:
            signals["terminated_near"] = track.termination_y_normalized > self.config.near_edge_threshold
            signals["terminated_far"] = track.termination_y_normalized < self.config.spawn_far_threshold
            
            y_travel = abs(track.termination_y_normalized - track.spawn_y_normalized)
            signals["ended_near_spawn"] = y_travel < 0.2
        
        # Signal 3 & 4: State history
        approaching_time = sum(1 for _, state in track.state_history if state == TrackState.APPROACHING)
        departing_time = sum(1 for _, state in track.state_history if state == TrackState.DEPARTING)
        
        signals["was_approaching"] = approaching_time > departing_time
        signals["was_departing"] = departing_time > approaching_time
        
        # Signal 5: Area change
        if track.termination_area and track.spawn_area > 0:
            area_ratio = track.termination_area / track.spawn_area
            signals["area_grew"] = area_ratio > self.config.area_growth_for_approach
            signals["area_shrank"] = area_ratio < self.config.area_shrink_for_depart
        
        # Decision logic
        entry_signals = (
            (signals["spawn_far"] and signals["terminated_near"]) or
            (signals["was_approaching"] and signals["terminated_near"]) or
            (signals["area_grew"] and signals["terminated_near"])
        )
        
        exit_signals = (
            (signals["spawn_near"] and signals["terminated_far"]) or
            (signals["was_departing"] and signals["terminated_far"]) or
            (signals["area_shrank"] and signals["terminated_far"])
        )
        
        if entry_signals and not exit_signals:
            confidence = CrossingConfidence.HIGH if signals["was_approaching"] else CrossingConfidence.MEDIUM
            return "IN", confidence
        elif exit_signals and not entry_signals:
            confidence = CrossingConfidence.HIGH if signals["was_departing"] else CrossingConfidence.MEDIUM
            return "OUT", confidence
        elif signals["ended_near_spawn"]:
            return None, None  # Turnaround
        elif entry_signals and exit_signals:
            # Conflicting signals - use area trend as tiebreaker
            if signals["was_approaching"]:
                return "IN", CrossingConfidence.LOW
            elif signals["was_departing"]:
                return "OUT", CrossingConfidence.LOW
            return None, None
        else:
            return None, None
    
    def get_counts(self) -> Tuple[int, int, int]:
        """Get current counts: (entries, exits, occupancy)"""
        return self.entry_count, self.exit_count, self.entry_count - self.exit_count
    
    def reset_counts(self):
        """Reset all counters"""
        self.entry_count = 0
        self.exit_count = 0
        self.completed_crossings.clear()
    
    def get_active_tracks(self) -> Dict[int, TrackHistory]:
        """Get currently active tracks"""
        return self.tracks.copy()
    
    def get_track_state(self, track_id: int) -> Optional[TrackState]:
        """Get state of a specific track"""
        track = self.tracks.get(track_id)
        return track.state if track else None
    
    def get_completed_crossings(self) -> List[TrackHistory]:
        """Get list of completed track histories"""
        return self.completed_crossings.copy()
    
    def get_diagnostics(self) -> dict:
        """Get diagnostic information"""
        return {
            "active_tracks": len(self.tracks),
            "completed_crossings": len(self.completed_crossings),
            "entry_count": self.entry_count,
            "exit_count": self.exit_count,
            "occupancy": self.entry_count - self.exit_count,
            "config": {
                "near_edge_threshold": self.config.near_edge_threshold,
                "count_on_crossing": self.config.count_on_crossing,
                "crossing_hysteresis": self.config.crossing_hysteresis,
                "min_track_duration_ms": self.config.min_track_duration_ms,
            }
        }

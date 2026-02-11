"""
Direction Detector: Hybrid Threshold Crossing + Evidence Scoring
================================================================
Version: 3.0.0

How it works:
  1. THRESHOLD CROSSING triggers the count instantly when a person
     crosses the near_edge line (handles people staying in frame)
  2. EVIDENCE SCORING validates the direction using spawn position,
     area trends, and movement patterns (reduces false positives)
  3. TERMINATION FALLBACK catches anyone the threshold missed
     (e.g., person walked through at edge of frame)

Camera Setup:
    - Mounted on door frame, angled to see outside
    - Top of frame = far from door (outside/hallway)
    - Bottom of frame = near door threshold
    
    ENTRY: Person appears at top -> crosses threshold going DOWN
    EXIT:  Person near bottom -> crosses threshold going UP

Author: Ahmad's Mosque Attendance System
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
import time


# ============================================================================
# ENUMS
# ============================================================================

class TrackState(Enum):
    NASCENT = auto()
    APPROACHING = auto()
    DEPARTING = auto()
    STABLE = auto()
    CROSSED_IN = auto()
    CROSSED_OUT = auto()
    ABANDONED = auto()


class CrossingConfidence(Enum):
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()
    UNCERTAIN = auto()


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DirectionDetectorConfig:
    """All tunable parameters. Edit QUICK CONFIG in people_counter.py instead."""
    
    # Frame geometry
    frame_height: int = 480
    frame_width: int = 640
    
    # === THRESHOLD CROSSING (primary trigger) ===
    near_edge_threshold: float = 0.60
    crossing_hysteresis: float = 0.03
    min_samples_before_crossing: int = 3
    
    # === EVIDENCE SCORING (confidence validation) ===
    spawn_far_threshold: float = 0.35
    spawn_near_threshold: float = 0.65
    area_growth_for_approach: float = 1.15
    area_shrink_for_depart: float = 0.85
    
    # === TRACK MANAGEMENT ===
    min_track_duration_ms: float = 200
    min_track_frames: int = 4
    track_timeout_ms: float = 500
    
    # === CONFIDENCE THRESHOLDS ===
    high_evidence_threshold: int = 5
    medium_evidence_threshold: int = 3
    
    # === TERMINATION FALLBACK ===
    enable_termination_fallback: bool = True
    termination_min_y_travel: float = 0.20
    
    # === FILTERING ===
    min_bbox_area_ratio: float = 0.005


# ============================================================================
# TRACK HISTORY
# ============================================================================

@dataclass
class TrackHistory:
    """Full lifetime data for a single tracked person."""
    track_id: int
    
    spawn_time: float = 0.0
    spawn_y_normalized: float = 0.0
    spawn_area: float = 0.0
    
    y_history: list = field(default_factory=list)
    area_history: list = field(default_factory=list)
    state_history: list = field(default_factory=list)
    
    state: TrackState = TrackState.NASCENT
    last_update_time: float = 0.0
    frame_count: int = 0
    
    # Threshold crossing state machine
    threshold_state: Optional[str] = None  # "above" | "below" | None
    has_crossed: bool = False
    crossed_direction: Optional[str] = None
    crossed_time: Optional[float] = None
    
    # Termination info
    termination_y: Optional[float] = None
    termination_area: Optional[float] = None
    
    def get_current_state(self) -> TrackState:
        """Determine movement state from recent history."""
        if len(self.area_history) < 3 or len(self.y_history) < 3:
            return TrackState.NASCENT
        
        recent_areas = self.area_history[-5:]
        recent_ys = self.y_history[-5:]
        
        if len(recent_areas) < 2:
            return TrackState.NASCENT
        
        area_ratio = recent_areas[-1] / max(recent_areas[0], 1e-8)
        y_delta = recent_ys[-1] - recent_ys[0]
        
        if area_ratio > 1.05 and y_delta > 0.02:
            return TrackState.APPROACHING
        elif area_ratio < 0.95 and y_delta < -0.02:
            return TrackState.DEPARTING
        else:
            return TrackState.STABLE


# ============================================================================
# DIRECTION DETECTOR
# ============================================================================

class DirectionDetector:
    """
    Hybrid detector: threshold crossing + evidence scoring.
    
    Primary:  Counts when person crosses the threshold line.
    Validate: Evidence scoring determines confidence level.
    Fallback: If crossing missed, classify on track termination.
    """
    
    def __init__(self, config: Optional[DirectionDetectorConfig] = None):
        self.config = config or DirectionDetectorConfig()
        self.tracks: Dict[int, TrackHistory] = {}
        self.completed_crossings: List[dict] = []
        
        self.entry_count: int = 0
        self.exit_count: int = 0
        self.frame_area = self.config.frame_width * self.config.frame_height
    
    # ----------------------------------------------------------------
    # PUBLIC API
    # ----------------------------------------------------------------
    
    def update(self, detections: List[Tuple[int, Tuple[float, float, float, float]]]) -> List[dict]:
        """
        Process detections. Returns crossing events this frame.
        
        Args:
            detections: [(track_id, (x1, y1, x2, y2)), ...]
        Returns:
            [{"track_id", "direction", "confidence", "trigger", ...}, ...]
        """
        timestamp = time.time()
        events = []
        seen_ids = set()
        
        for track_id, bbox in detections:
            seen_ids.add(track_id)
            if track_id not in self.tracks:
                self._init_track(track_id, bbox, timestamp)
            else:
                event = self._update_track(track_id, bbox, timestamp)
                if event:
                    events.append(event)
        
        # Terminated tracks
        terminated = set(self.tracks.keys()) - seen_ids
        for tid in terminated:
            event = self._handle_termination(tid, timestamp)
            if event:
                events.append(event)
        
        # Timed-out tracks
        timeout_thresh = timestamp - (self.config.track_timeout_ms / 1000.0)
        timed_out = [
            tid for tid, t in self.tracks.items()
            if t.last_update_time < timeout_thresh and tid not in terminated
        ]
        for tid in timed_out:
            event = self._handle_termination(tid, timestamp, timed_out=True)
            if event:
                events.append(event)
        
        return events
    
    def get_track_info(self, track_id: int) -> Optional[dict]:
        """Get info for an active track (for visualization)."""
        track = self.tracks.get(track_id)
        if not track:
            return None
        state = track.get_current_state()
        return {
            "track_id": track_id,
            "state": state.name.lower(),
            "frame_count": track.frame_count,
            "spawn_y": track.spawn_y_normalized,
            "current_y": track.y_history[-1] if track.y_history else 0,
            "current_area": track.area_history[-1] if track.area_history else 0,
            "spawn_area": track.spawn_area,
            "has_crossed": track.has_crossed,
            "crossed_direction": track.crossed_direction,
            "threshold_state": track.threshold_state,
        }
    
    def get_debug_info(self, track_id: int) -> Optional[dict]:
        """Backward-compatible alias for get_track_info."""
        return self.get_track_info(track_id)
    
    def get_stats(self) -> dict:
        return {
            "entries": self.entry_count,
            "exits": self.exit_count,
            "current_occupancy": max(0, self.entry_count - self.exit_count),
            "active_tracks": len(self.tracks),
        }
    
    def reset(self):
        self.tracks.clear()
        self.completed_crossings.clear()
        self.entry_count = 0
        self.exit_count = 0
    
    # ----------------------------------------------------------------
    # TRACK LIFECYCLE
    # ----------------------------------------------------------------
    
    def _init_track(self, track_id: int, bbox: Tuple, timestamp: float):
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        y_norm = ((y1 + y2) / 2) / self.config.frame_height
        area_norm = area / self.frame_area
        
        if area_norm < self.config.min_bbox_area_ratio:
            return
        
        self.tracks[track_id] = TrackHistory(
            track_id=track_id,
            spawn_time=timestamp,
            spawn_y_normalized=y_norm,
            spawn_area=area_norm,
            last_update_time=timestamp,
            frame_count=1,
            y_history=[y_norm],
            area_history=[area_norm],
            state_history=[(timestamp, TrackState.NASCENT)],
        )
    
    def _update_track(self, track_id: int, bbox: Tuple, timestamp: float) -> Optional[dict]:
        track = self.tracks.get(track_id)
        if not track:
            return None
        
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        y_norm = ((y1 + y2) / 2) / self.config.frame_height
        area_norm = area / self.frame_area
        
        track.last_update_time = timestamp
        track.frame_count += 1
        track.y_history.append(y_norm)
        track.area_history.append(area_norm)
        
        # Bound history
        if len(track.y_history) > 300:
            track.y_history = track.y_history[-200:]
        if len(track.area_history) > 300:
            track.area_history = track.area_history[-200:]
        
        # Update movement state
        state = track.get_current_state()
        track.state = state
        track.state_history.append((timestamp, state))
        
        # Check threshold crossing
        return self._check_threshold_crossing(track, y_norm, timestamp)
    
    def _handle_termination(self, track_id: int, timestamp: float,
                            timed_out: bool = False) -> Optional[dict]:
        """Termination fallback: evidence-score tracks that never crossed."""
        if track_id not in self.tracks:
            return None
        
        track = self.tracks.pop(track_id)
        track.termination_y = track.y_history[-1] if track.y_history else None
        track.termination_area = track.area_history[-1] if track.area_history else None
        
        # Already counted via threshold crossing
        if track.has_crossed:
            return None
        
        if not self.config.enable_termination_fallback:
            return None
        
        duration_ms = (timestamp - track.spawn_time) * 1000
        if duration_ms < self.config.min_track_duration_ms:
            return None
        if track.frame_count < self.config.min_track_frames:
            return None
        
        # Must have traveled enough
        if track.termination_y is not None:
            y_travel = abs(track.termination_y - track.spawn_y_normalized)
            if y_travel < self.config.termination_min_y_travel:
                return None
        
        direction, confidence, signals = self._evidence_classify(track)
        if direction is None:
            return None
        
        if direction == "IN":
            self.entry_count += 1
        else:
            self.exit_count += 1
        
        event = {
            "track_id": track_id,
            "direction": direction,
            "confidence": confidence.name,
            "trigger": "termination_fallback",
            "timed_out": timed_out,
            "frame_count": track.frame_count,
            "duration_ms": duration_ms,
            "spawn_y": track.spawn_y_normalized,
            "termination_y": track.termination_y,
            "signals": signals,
        }
        self.completed_crossings.append(event)
        return event
    
    # ----------------------------------------------------------------
    # THRESHOLD CROSSING (primary trigger)
    # ----------------------------------------------------------------
    
    def _check_threshold_crossing(self, track: TrackHistory, current_y: float,
                                   timestamp: float) -> Optional[dict]:
        """
        State machine: above/below threshold with hysteresis.
        Crossing = state transition above->below (IN) or below->above (OUT).
        """
        threshold = self.config.near_edge_threshold
        hyst = self.config.crossing_hysteresis
        
        if current_y < threshold - hyst:
            current_state = "above"
        elif current_y > threshold + hyst:
            current_state = "below"
        else:
            current_state = None
        
        # Initialize on first clear position
        if track.threshold_state is None and current_state is not None:
            track.threshold_state = current_state
            return None
        
        # Minimum samples
        if track.frame_count < self.config.min_samples_before_crossing:
            return None
        
        # Minimum duration
        duration_ms = (timestamp - track.spawn_time) * 1000
        if duration_ms < self.config.min_track_duration_ms:
            return None
        
        # Detect crossing
        crossing_direction = None
        if track.threshold_state == "above" and current_state == "below":
            crossing_direction = "IN"
        elif track.threshold_state == "below" and current_state == "above":
            crossing_direction = "OUT"
        
        # Update state
        if current_state is not None:
            track.threshold_state = current_state
        
        if crossing_direction is None:
            return None
        
        # Prevent double-count same direction
        if track.crossed_direction == crossing_direction:
            return None
        
        # Validate with evidence scoring
        confidence, signals = self._get_crossing_confidence(track, crossing_direction)
        
        # Mark as crossed
        track.has_crossed = True
        track.crossed_direction = crossing_direction
        track.crossed_time = timestamp
        track.state = TrackState.CROSSED_IN if crossing_direction == "IN" else TrackState.CROSSED_OUT
        
        if crossing_direction == "IN":
            self.entry_count += 1
        else:
            self.exit_count += 1
        
        area_ratio = 1.0
        if track.spawn_area > 0 and track.area_history:
            area_ratio = track.area_history[-1] / track.spawn_area
        
        event = {
            "track_id": track.track_id,
            "direction": crossing_direction,
            "confidence": confidence.name,
            "trigger": "threshold_crossing",
            "frame_count": track.frame_count,
            "duration_ms": duration_ms,
            "spawn_y": track.spawn_y_normalized,
            "crossing_y": current_y,
            "area_ratio": area_ratio,
            "signals": signals,
        }
        self.completed_crossings.append(event)
        return event
    
    # ----------------------------------------------------------------
    # EVIDENCE SCORING
    # ----------------------------------------------------------------
    
    def _get_crossing_confidence(self, track: TrackHistory,
                                  direction: str) -> Tuple[CrossingConfidence, dict]:
        """Validate a threshold crossing with evidence scoring."""
        signals = self._compute_signals(track)
        score = self._score_evidence(signals, direction)
        
        if score >= self.config.high_evidence_threshold:
            confidence = CrossingConfidence.HIGH
        elif score >= self.config.medium_evidence_threshold:
            confidence = CrossingConfidence.MEDIUM
        else:
            confidence = CrossingConfidence.LOW
        
        return confidence, signals
    
    def _evidence_classify(self, track: TrackHistory) -> Tuple[Optional[str], Optional[CrossingConfidence], dict]:
        """Full evidence classification (termination fallback)."""
        signals = self._compute_signals(track)
        
        entry_score = self._score_evidence(signals, "IN")
        exit_score = self._score_evidence(signals, "OUT")
        
        if entry_score < 2 and exit_score < 2:
            return None, None, signals
        
        if entry_score > exit_score:
            direction, score = "IN", entry_score
        elif exit_score > entry_score:
            direction, score = "OUT", exit_score
        else:
            return None, None, signals
        
        if score >= self.config.high_evidence_threshold:
            confidence = CrossingConfidence.HIGH
        elif score >= self.config.medium_evidence_threshold:
            confidence = CrossingConfidence.MEDIUM
        else:
            confidence = CrossingConfidence.LOW
        
        return direction, confidence, signals
    
    def _compute_signals(self, track: TrackHistory) -> dict:
        """Compute all evidence signals from track history."""
        signals = {
            "spawn_far": False, "spawn_near": False,
            "terminated_near": False, "terminated_far": False,
            "was_approaching": False, "was_departing": False,
            "area_grew": False, "area_shrank": False,
            "ended_near_spawn": False,
        }
        
        signals["spawn_far"] = track.spawn_y_normalized < self.config.spawn_far_threshold
        signals["spawn_near"] = track.spawn_y_normalized > self.config.spawn_near_threshold
        
        last_y = track.termination_y or (track.y_history[-1] if track.y_history else None)
        if last_y is not None:
            signals["terminated_near"] = last_y > self.config.near_edge_threshold
            signals["terminated_far"] = last_y < self.config.spawn_far_threshold
            signals["ended_near_spawn"] = abs(last_y - track.spawn_y_normalized) < 0.15
        
        approaching = sum(1 for _, s in track.state_history if s == TrackState.APPROACHING)
        departing = sum(1 for _, s in track.state_history if s == TrackState.DEPARTING)
        signals["was_approaching"] = approaching > departing
        signals["was_departing"] = departing > approaching
        
        last_area = track.termination_area or (track.area_history[-1] if track.area_history else None)
        if last_area and track.spawn_area > 0:
            ratio = last_area / track.spawn_area
            signals["area_grew"] = ratio > self.config.area_growth_for_approach
            signals["area_shrank"] = ratio < self.config.area_shrink_for_depart
        
        return signals
    
    def _score_evidence(self, signals: dict, direction: str) -> int:
        """
        Score evidence for a direction.
        Weights: spawn=2, termination=3, movement=2, area=1, near-spawn=-2
        """
        score = 0
        if direction == "IN":
            if signals["spawn_far"]:        score += 2
            if signals["terminated_near"]:  score += 3
            if signals["was_approaching"]:  score += 2
            if signals["area_grew"]:        score += 1
        else:
            if signals["spawn_near"]:       score += 2
            if signals["terminated_far"]:   score += 3
            if signals["was_departing"]:    score += 2
            if signals["area_shrank"]:      score += 1
        
        if signals["ended_near_spawn"]:
            score -= 2
        
        return score
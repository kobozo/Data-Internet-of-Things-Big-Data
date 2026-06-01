"""
Tiny centroid tracker.

Good enough to give each person a stable ID for a few seconds, which
is all we need to measure dwell time and average speed.  No external
dependencies (deep_sort etc. would be overkill for our use case).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from .detector import Detection


@dataclass
class Track:
    track_id: int
    first_seen_ts: float
    last_seen_ts: float
    last_detection: Detection
    centroid_history: Deque[Tuple[float, float, float]] = field(
        default_factory=lambda: deque(maxlen=60)
    )           # (ts, cx, cy)
    disappeared: int = 0

    # cumulative per-person evidence (filled in by classifier)
    has_suitcase: bool = False
    has_backpack: bool = False
    has_handbag: bool = False
    holding_phone_high: bool = False

    @property
    def lifetime(self) -> float:
        return self.last_seen_ts - self.first_seen_ts

    @property
    def avg_speed_px(self) -> float:
        """Mean per-step centroid displacement."""
        if len(self.centroid_history) < 2:
            return 0.0
        total = 0.0
        prev = None
        for ts, cx, cy in self.centroid_history:
            if prev is not None:
                total += math.hypot(cx - prev[0], cy - prev[1])
            prev = (cx, cy)
        return total / (len(self.centroid_history) - 1)

    @property
    def dwell_seconds(self) -> float:
        """How long the centroid has stayed within ~25 px."""
        if len(self.centroid_history) < 2:
            return 0.0
        last_ts, last_cx, last_cy = self.centroid_history[-1]
        dwell = 0.0
        for ts, cx, cy in reversed(self.centroid_history):
            if math.hypot(cx - last_cx, cy - last_cy) <= 25:
                dwell = last_ts - ts
            else:
                break
        return dwell


class CentroidTracker:
    def __init__(self, max_disappeared: int = 15, max_distance_px: float = 80):
        self.max_disappeared = max_disappeared
        self.max_distance_px = max_distance_px
        self._next_id = 1
        self.tracks: Dict[int, Track] = {}
        self._just_finalised: List[Track] = []

    # ----- public API ------------------------------------------------
    def update(self, person_dets: List[Detection], ts: float) -> Dict[int, Track]:
        """Match person detections to existing tracks.  Returns the
        currently *active* tracks indexed by id."""
        # clear the drain queue from the previous update
        self._just_finalised.clear()
        # 1. Build a list of detection centroids
        det_centroids = [(d.cx, d.cy) for d in person_dets]

        # 2. If no existing tracks, every detection becomes a new one
        if not self.tracks:
            for d in person_dets:
                self._spawn(d, ts)
            return self.tracks

        # 3. Build cost matrix between tracks and detections
        track_ids = list(self.tracks.keys())
        track_centroids = [
            (self.tracks[tid].last_detection.cx,
             self.tracks[tid].last_detection.cy)
            for tid in track_ids
        ]

        # greedy nearest-neighbour assignment - O(N*M), N,M <= 30 in practice
        unmatched_dets = set(range(len(person_dets)))
        unmatched_tracks = set(range(len(track_ids)))

        pairs: List[Tuple[float, int, int]] = []
        for ti, (tx, ty) in enumerate(track_centroids):
            for di, (dx, dy) in enumerate(det_centroids):
                dist = math.hypot(tx - dx, ty - dy)
                if dist <= self.max_distance_px:
                    pairs.append((dist, ti, di))
        pairs.sort()

        for dist, ti, di in pairs:
            if ti not in unmatched_tracks or di not in unmatched_dets:
                continue
            tid = track_ids[ti]
            tr = self.tracks[tid]
            det = person_dets[di]
            tr.last_detection = det
            tr.last_seen_ts = ts
            tr.disappeared = 0
            tr.centroid_history.append((ts, det.cx, det.cy))
            unmatched_tracks.discard(ti)
            unmatched_dets.discard(di)

        # 4. Detections without a partner -> new tracks
        for di in unmatched_dets:
            self._spawn(person_dets[di], ts)

        # 5. Tracks without a partner -> increment disappear counter, drop if too old
        dead = []
        for ti in unmatched_tracks:
            tid = track_ids[ti]
            tr = self.tracks[tid]
            tr.disappeared += 1
            if tr.disappeared > self.max_disappeared:
                dead.append(tid)
        for tid in dead:
            self._just_finalised.append(self.tracks[tid])
            del self.tracks[tid]

        return self.tracks

    def drain_finalised(self) -> List[Track]:
        """Pop tracks that died on the most recent update."""
        out = list(self._just_finalised)
        self._just_finalised.clear()
        return out

    def all_active(self) -> List[Track]:
        return list(self.tracks.values())

    def _spawn(self, det: Detection, ts: float) -> None:
        tr = Track(
            track_id=self._next_id,
            first_seen_ts=ts,
            last_seen_ts=ts,
            last_detection=det,
        )
        tr.centroid_history.append((ts, det.cx, det.cy))
        self.tracks[self._next_id] = tr
        self._next_id += 1

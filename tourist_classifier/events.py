"""
Event generator.

Consumes per-frame metrics and emits debounced 'high-level' events.
The downstream system (MQTT subscriber, dashboard, alerting) reacts to
these events rather than to the firehose of frame-level rows.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from .classifier import Score
from .tracker import Track


@dataclass
class Event:
    ts: float
    type: str
    payload: Dict


@dataclass
class FrameMetrics:
    ts: float
    n_people: int
    n_tourists: int
    n_locals: int
    n_uncertain: int
    avg_dwell: float
    avg_speed: float

    @property
    def tourist_ratio(self) -> float:
        return self.n_tourists / self.n_people if self.n_people else 0.0


class EventEngine:
    def __init__(self, cfg: dict):
        self.window_seconds = float(cfg["window_seconds"])
        self.rules = cfg["rules"]
        self.cooldowns: Dict[str, float] = {}
        self.window: Deque[FrameMetrics] = deque()

    # ------------------------------------------------------------------
    def ingest(self,
               ts: float,
               tracks: Dict[int, Track],
               scores: Dict[int, Score]) -> Tuple[FrameMetrics, List[Event]]:
        m = self._metrics(ts, tracks, scores)
        self.window.append(m)
        # drop frames older than window
        cutoff = ts - self.window_seconds
        while self.window and self.window[0].ts < cutoff:
            self.window.popleft()

        events = self._evaluate_rules(ts, tracks, scores)
        return m, events

    # ------------------------------------------------------------------
    def _metrics(self, ts, tracks, scores) -> FrameMetrics:
        n_people = len(tracks)
        n_t = sum(1 for s in scores.values() if s.label == "TOURIST")
        n_l = sum(1 for s in scores.values() if s.label == "LOCAL")
        n_u = n_people - n_t - n_l
        avg_dwell = (sum(tr.dwell_seconds for tr in tracks.values())
                     / n_people) if n_people else 0.0
        avg_speed = (sum(tr.avg_speed_px for tr in tracks.values())
                     / n_people) if n_people else 0.0
        return FrameMetrics(
            ts=ts,
            n_people=n_people,
            n_tourists=n_t,
            n_locals=n_l,
            n_uncertain=n_u,
            avg_dwell=round(avg_dwell, 2),
            avg_speed=round(avg_speed, 2),
        )

    # ------------------------------------------------------------------
    def _on_cooldown(self, name: str, ts: float) -> bool:
        last = self.cooldowns.get(name, 0.0)
        cd = self.rules[name].get("cooldown_seconds", 30)
        return (ts - last) < cd

    def _fire(self, name: str, ts: float, payload: dict) -> Event:
        self.cooldowns[name] = ts
        return Event(ts=ts, type=name, payload=payload)

    # ------------------------------------------------------------------
    def _evaluate_rules(self, ts, tracks, scores) -> List[Event]:
        events: List[Event] = []

        # ---- tourist_hotspot ---------------------------------------
        rule = self.rules.get("tourist_hotspot")
        if rule and not self._on_cooldown("tourist_hotspot", ts):
            tourists_in_window = sum(m.n_tourists for m in self.window)
            people_in_window = sum(m.n_people for m in self.window) or 1
            ratio = tourists_in_window / people_in_window
            if (tourists_in_window >= rule["min_tourists_in_window"]
                    and ratio >= rule["min_tourist_ratio"]):
                events.append(self._fire("tourist_hotspot", ts, {
                    "tourists_in_window": tourists_in_window,
                    "tourist_ratio": round(ratio, 2),
                    "window_seconds": self.window_seconds,
                }))

        # ---- tourist_group_arrived ---------------------------------
        rule = self.rules.get("tourist_group_arrived")
        if rule and not self._on_cooldown("tourist_group_arrived", ts):
            tourist_tracks = [tracks[tid] for tid, s in scores.items()
                              if s.label == "TOURIST" and tid in tracks]
            cluster = self._largest_spatial_cluster(
                tourist_tracks, rule["max_pairwise_distance_px"])
            if cluster >= rule["min_simultaneous"]:
                events.append(self._fire("tourist_group_arrived", ts, {
                    "group_size": cluster,
                }))

        # ---- density_spike -----------------------------------------
        rule = self.rules.get("density_spike")
        if rule and not self._on_cooldown("density_spike", ts):
            if len(self.window) >= 5:
                # baseline = mean of the first 80% of the window
                cut = max(1, int(len(self.window) * 0.8))
                older = list(self.window)[:cut]
                baseline = (sum(m.n_people for m in older) / len(older)) or 1
                current = self.window[-1].n_people
                if (current >= rule["min_people"] and
                        current >= rule["multiplier_vs_baseline"] * baseline):
                    events.append(self._fire("density_spike", ts, {
                        "current": current,
                        "baseline": round(baseline, 2),
                    }))

        # ---- quiet_period ------------------------------------------
        rule = self.rules.get("quiet_period")
        if rule and not self._on_cooldown("quiet_period", ts):
            duration = rule["duration_seconds"]
            recent = [m for m in self.window if ts - m.ts <= duration]
            if recent and all(m.n_people <= rule["max_people"] for m in recent):
                # require enough samples in the period
                if recent[-1].ts - recent[0].ts >= duration * 0.8:
                    events.append(self._fire("quiet_period", ts, {
                        "max_people": rule["max_people"],
                        "duration_seconds": duration,
                    }))

        return events

    @staticmethod
    def _largest_spatial_cluster(tracks: List[Track], max_dist: float) -> int:
        """Union-find sized clusters by pairwise distance."""
        n = len(tracks)
        if n == 0:
            return 0
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(n):
            for j in range(i + 1, n):
                a, b = tracks[i].last_detection, tracks[j].last_detection
                if math.hypot(a.cx - b.cx, a.cy - b.cy) <= max_dist:
                    parent[find(i)] = find(j)
        sizes: Dict[int, int] = {}
        for i in range(n):
            r = find(i)
            sizes[r] = sizes.get(r, 0) + 1
        return max(sizes.values()) if sizes else 0

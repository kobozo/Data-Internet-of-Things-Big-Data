"""
Tourist-vs-Local scoring.

We look at each PERSON track and combine evidence from:

  * luggage/accessory detections that spatially overlap the person bbox
  * 'photo pose': a cell-phone bbox sitting in the UPPER half of the
    person bbox (a local checks their messages with the phone low,
    a tourist raises it to shoot the scenery)
  * dwell time: tourists stop more often
  * walking speed: tourists walk slower
  * group membership: another high-scoring person within R px

The output is one of TOURIST / LOCAL / UNCERTAIN per track with the
underlying numeric score so we can plot / aggregate later.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .detector import Detection
from .tracker import Track


# ----- IoU / containment helpers ------------------------------------
def _bbox_overlap_ratio(inner: Detection, outer: Detection) -> float:
    """Fraction of `inner` that sits inside `outer`'s bbox."""
    ix1 = max(inner.x1, outer.x1)
    iy1 = max(inner.y1, outer.y1)
    ix2 = min(inner.x2, outer.x2)
    iy2 = min(inner.y2, outer.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    inner_area = max(1.0, inner.w * inner.h)
    return inter / inner_area


def _phone_high_on_person(phone: Detection, person: Detection) -> bool:
    """True when the phone bbox is in the upper third of the person bbox."""
    if _bbox_overlap_ratio(phone, person) < 0.3:
        return False
    phone_cy = phone.cy
    person_top = person.y1
    person_bot = person.y2
    third = person_top + (person_bot - person_top) / 3.0
    return phone_cy < third


# ----- main scoring -------------------------------------------------
@dataclass
class Score:
    label: str       # TOURIST / LOCAL / UNCERTAIN
    score: float
    reasons: List[str]


class TouristClassifier:
    def __init__(self, cfg: dict):
        self.weights = cfg["weights"]
        self.dwell_seconds = float(cfg["dwell_seconds"])
        self.slow_speed_px = float(cfg["slow_speed_px"])
        self.group_radius_px = float(cfg["group_radius_px"])
        self.thr = cfg["thresholds"]

    # ----- per-frame update of the track's cumulative evidence ------
    def attach_accessories(self,
                           tracks: Dict[int, Track],
                           detections: List[Detection]) -> None:
        """For each non-person detection (luggage, phone), find the
        track whose person bbox most contains it and update its flags."""
        # snapshot person tracks
        person_tracks = list(tracks.values())
        for det in detections:
            if det.cls_name == "person":
                continue
            best_tr = None
            best_overlap = 0.0
            for tr in person_tracks:
                overlap = _bbox_overlap_ratio(det, tr.last_detection)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_tr = tr
            if best_tr is None or best_overlap < 0.3:
                continue

            if det.cls_name == "suitcase":
                best_tr.has_suitcase = True
            elif det.cls_name == "backpack":
                best_tr.has_backpack = True
            elif det.cls_name == "handbag":
                best_tr.has_handbag = True
            elif det.cls_name == "cell_phone":
                if _phone_high_on_person(det, best_tr.last_detection):
                    best_tr.holding_phone_high = True

    # ----- compute label per track ----------------------------------
    def score_tracks(self, tracks: Dict[int, Track]) -> Dict[int, Score]:
        scores: Dict[int, Score] = {}

        # first pass: individual score without group bonus
        raw: Dict[int, Tuple[float, List[str]]] = {}
        for tid, tr in tracks.items():
            s = 0.0
            reasons: List[str] = []
            if tr.has_suitcase:
                s += self.weights["has_suitcase"]; reasons.append("suitcase")
            if tr.has_backpack:
                s += self.weights["has_backpack"]; reasons.append("backpack")
            if tr.has_handbag:
                s += self.weights["has_handbag"]; reasons.append("handbag")
            if tr.holding_phone_high:
                s += self.weights["holding_phone_high"]; reasons.append("photo_pose")
            if tr.dwell_seconds >= self.dwell_seconds:
                s += self.weights["dwell_long"]; reasons.append(
                    f"dwell_{tr.dwell_seconds:.1f}s")
            if 0.1 < tr.avg_speed_px < self.slow_speed_px:
                s += self.weights["slow_walker"]; reasons.append("slow_walker")
            raw[tid] = (s, reasons)

        # second pass: group bonus from neighbouring high-scoring tracks
        high = {tid: tr for tid, tr in tracks.items()
                if raw[tid][0] >= self.thr["local"]}
        for tid, tr in tracks.items():
            s, reasons = raw[tid]
            cx, cy = tr.last_detection.cx, tr.last_detection.cy
            for other_id, other in high.items():
                if other_id == tid:
                    continue
                d = math.hypot(cx - other.last_detection.cx,
                               cy - other.last_detection.cy)
                if d <= self.group_radius_px:
                    s += self.weights["group_member"]
                    reasons.append(f"group_with_{other_id}")
                    break  # one group bonus is enough

            if s >= self.thr["tourist"]:
                label = "TOURIST"
            elif s < self.thr["local"]:
                label = "LOCAL"
            else:
                label = "UNCERTAIN"
            scores[tid] = Score(label=label, score=round(s, 2), reasons=reasons)
        return scores

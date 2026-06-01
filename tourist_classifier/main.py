"""
Entry point.

Usage:
    python -m tourist_classifier.main --config config.yaml
    python -m tourist_classifier.main --config config.yaml --source 0       # webcam
    python -m tourist_classifier.main --config config.yaml --headless       # no window
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from typing import Dict, Optional

import cv2
import numpy as np
import yaml

from .classifier import Score, TouristClassifier
from .detector import Detector
from .events import EventEngine
from .storage import StorageBundle
from .stream import Livestream
from .tracker import CentroidTracker, Track

log = logging.getLogger("tourist_classifier")


# ---------------------------------------------------------------------
# Visual overlay
# ---------------------------------------------------------------------
COLORS = {
    "TOURIST":   (0,  64, 255),   # red-orange (BGR)
    "LOCAL":     (0, 200,  64),   # green
    "UNCERTAIN": (200, 200, 0),   # yellow
}


def draw_overlay(frame, tracks: Dict[int, Track],
                 scores: Dict[int, Score], metrics) -> None:
    h, w = frame.shape[:2]
    for tid, tr in tracks.items():
        sc = scores.get(tid)
        if sc is None:
            continue
        col = COLORS.get(sc.label, (255, 255, 255))
        d = tr.last_detection
        cv2.rectangle(frame,
                      (int(d.x1), int(d.y1)),
                      (int(d.x2), int(d.y2)),
                      col, 2)
        label = f"#{tid} {sc.label} {sc.score:.1f}"
        cv2.putText(frame, label,
                    (int(d.x1), max(15, int(d.y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    # banner
    banner = (f"people:{metrics.n_people}  "
              f"tourists:{metrics.n_tourists}  "
              f"locals:{metrics.n_locals}  "
              f"uncertain:{metrics.n_uncertain}  "
              f"ratio:{metrics.tourist_ratio:.0%}")
    cv2.rectangle(frame, (0, 0), (w, 26), (0, 0, 0), -1)
    cv2.putText(frame, banner, (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------
_stop = False


def _handle_sigint(signum, frame):  # noqa: ARG001
    global _stop
    _stop = True
    log.info("Shutdown requested...")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Tourist-vs-Local livestream analyser")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--source",
                   help="Override stream.url (URL or webcam index)")
    p.add_argument("--headless", action="store_true",
                   help="Disable the OpenCV display window")
    p.add_argument("--max-seconds", type=float, default=0,
                   help="Stop after N seconds (0 = run forever)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = load_config(args.config)

    signal.signal(signal.SIGINT, _handle_sigint)

    # ---- wiring ----------------------------------------------------
    stream_cfg = cfg["stream"]
    source = args.source or stream_cfg["url"]
    show_window = stream_cfg.get("show_window", True) and not args.headless

    stream = Livestream(
        url=source,
        process_fps=stream_cfg.get("process_fps", 2),
        reconnect_after_seconds=stream_cfg.get("reconnect_after_seconds", 10),
    )

    det_cfg = cfg["detector"]
    detector = Detector(
        model_path=det_cfg["model"],
        conf_threshold=det_cfg["conf_threshold"],
        class_map=det_cfg["classes"],
        device=det_cfg.get("device", "cpu"),
    )

    tracker = CentroidTracker(
        max_disappeared=cfg["tracker"]["max_disappeared"],
        max_distance_px=cfg["tracker"]["max_distance_px"],
    )

    classifier = TouristClassifier(cfg["classifier"])
    event_engine = EventEngine(cfg["events"])
    storage = StorageBundle(cfg["storage"])

    # track-finalisation: keep last scores so we can flush on disappear
    last_scores: Dict[int, Score] = {}

    # scene-cut detection state
    scene_cfg = cfg.get("scene", {})
    cut_threshold = float(scene_cfg.get("cut_threshold", 0))
    prev_small: Optional[np.ndarray] = None
    scene_id = 0
    n_cuts = 0
    # timing diagnostics
    inference_times = []

    started = time.time()
    try:
        with stream:
            for frm in stream.frames():
                if _stop:
                    break
                if args.max_seconds and (time.time() - started) > args.max_seconds:
                    break

                # ---- scene-cut detection ---------------------------
                # On multi-camera composite streams (EarthCam Crossroads
                # etc.) the source cuts to a different physical camera
                # every 1-2 s.  Each such cut LOOKS LIKE every pedestrian
                # teleporting; in reality the previous view is simply
                # gone.  We detect it via mean-abs-diff on a 160x90
                # grayscale thumbnail.
                cut_detected = False
                if cut_threshold > 0:
                    cur_small = cv2.resize(
                        cv2.cvtColor(frm.image, cv2.COLOR_BGR2GRAY),
                        (160, 90))
                    if prev_small is not None:
                        mad = float(np.mean(
                            np.abs(cur_small.astype(np.int16)
                                   - prev_small.astype(np.int16))))
                        if mad > cut_threshold:
                            cut_detected = True
                            n_cuts += 1
                            scene_id += 1
                            log.info("scene cut (mad=%.1f) -> reset tracker, "
                                     "scene_id=%d", mad, scene_id)
                            # drop everything: tracks + their queued scores.
                            # We do NOT clear the event-engine's rolling window
                            # because the headcount metrics are still valid
                            # observations even across cuts.
                            tracker.tracks.clear()
                            tracker._just_finalised.clear()
                            last_scores.clear()
                    prev_small = cur_small

                # ---- detection -------------------------------------
                t_inf = time.time()
                detections = detector.infer(frm.image)
                inference_times.append(time.time() - t_inf)
                persons = [d for d in detections if d.cls_name == "person"]
                accessories = [d for d in detections if d.cls_name != "person"]

                # ---- tracking --------------------------------------
                active_tracks = tracker.update(persons, frm.ts)

                # ---- attach evidence + classify --------------------
                classifier.attach_accessories(active_tracks, accessories)
                scores = classifier.score_tracks(active_tracks)
                last_scores.update(scores)

                # ---- finalise tracks that just disappeared ---------
                for tr in tracker.drain_finalised():
                    sc = last_scores.pop(tr.track_id,
                                         Score("UNCERTAIN", 0.0, []))
                    # only log tracks long enough to be informative
                    if tr.lifetime >= 0.5:
                        storage.write_track(tr, sc, frm.ts)
                        log.debug("Track %s finalised: %s (score %.2f)",
                                  tr.track_id, sc.label, sc.score)

                # ---- events / metrics ------------------------------
                metrics, events = event_engine.ingest(
                    frm.ts, active_tracks, scores)
                storage.write_frame(metrics)
                for ev in events:
                    log.info("EVENT %s %s", ev.type, ev.payload)
                    storage.write_event(ev)

                # ---- visualise -------------------------------------
                if show_window:
                    draw_overlay(frm.image, active_tracks, scores, metrics)
                    cv2.imshow("Tourist vs Local", frm.image)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):  # 27=ESC
                        break

                # ---- per-frame log ---------------------------------
                if frm.index % 10 == 0:
                    age_ms = (time.time() - frm.ts) * 1000
                    recent_inf = inference_times[-10:]
                    avg_inf_ms = (sum(recent_inf) / len(recent_inf) * 1000
                                  ) if recent_inf else 0
                    log.info(
                        "frame=%d people=%d T=%d L=%d U=%d  "
                        "scene=%d cuts=%d  inf=%.0fms age=%.0fms",
                        frm.index, metrics.n_people, metrics.n_tourists,
                        metrics.n_locals, metrics.n_uncertain,
                        scene_id, n_cuts, avg_inf_ms, age_ms,
                    )

    finally:
        # flush remaining tracks as 'closed' rows
        for tid, tr in tracker.tracks.items():
            sc = last_scores.get(tid, Score("UNCERTAIN", 0.0, []))
            storage.write_track(tr, sc, time.time())
        storage.close()
        if show_window:
            cv2.destroyAllWindows()
        log.info("Bye.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

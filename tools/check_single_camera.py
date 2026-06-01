"""
Check whether a YouTube livestream is a SINGLE FIXED CAMERA
(good) or a multi-camera compilation (bad — the tracker can't
accumulate dwell time across cuts).

Pulls ~8 frames at 1 fps and measures the mean absolute pixel
difference between consecutive low-res grayscale frames.  A
camera cut shows up as a huge spike in that diff.

Usage:
    python tools/check_single_camera.py <url>

Example:
    python tools/check_single_camera.py https://www.youtube.com/watch?v=j39vIidsIJI
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# allow `python tools/check_single_camera.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tourist_classifier.stream import Livestream  # noqa: E402


# anything above this MAD between successive frames is almost certainly
# a camera cut and not just pedestrian/light motion
CUT_THRESHOLD = 25.0


def analyse(url: str, n_frames: int = 8, fps: float = 1.0) -> int:
    print(f"Connecting to {url} ...", flush=True)
    ls = Livestream(url, process_fps=fps)
    ls.open()
    diffs: list[float] = []
    prev = None
    t0 = time.time()
    n_read = 0
    for i, frm in enumerate(ls.frames()):
        small = cv2.resize(
            cv2.cvtColor(frm.image, cv2.COLOR_BGR2GRAY), (160, 90)
        )
        if prev is not None:
            diffs.append(
                float(np.mean(np.abs(small.astype(int) - prev.astype(int))))
            )
        prev = small
        n_read += 1
        if i + 1 >= n_frames or time.time() - t0 > 25:
            break
    ls.close()

    if not diffs:
        print("FAILED: could not pull enough frames")
        return 2

    print(f"\nframe-to-frame diffs (lower = more stable): {[round(d,1) for d in diffs]}")
    max_d = max(diffs)
    switches = sum(1 for d in diffs if d > CUT_THRESHOLD)

    if switches == 0:
        print(f"max_diff={max_d:.1f}, no cuts detected")
        print("VERDICT: SINGLE-CAMERA -> safe to use for the demo.")
        return 0
    else:
        print(f"max_diff={max_d:.1f}, {switches} of {len(diffs)} transitions look like camera cuts")
        print("VERDICT: MULTI-CAMERA -> the tracker will reset on every cut, "
              "no pedestrian will accumulate enough dwell time to be classified.  "
              "Pick a different URL.")
        return 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("url")
    p.add_argument("--frames", type=int, default=8)
    args = p.parse_args()
    try:
        return analyse(args.url, n_frames=args.frames)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

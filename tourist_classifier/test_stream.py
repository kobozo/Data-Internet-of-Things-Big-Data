"""Unit tests for tourist_classifier.stream.Livestream.

These exercise the decimation/clock contract without touching the network:
cv2.VideoCapture is replaced by a deterministic FakeCapture and the URL
resolvers are patched to no-ops.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from . import stream as stream_mod
from .stream import Livestream


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class FakeCapture:
    """Deterministic stand-in for cv2.VideoCapture.

    Yields ``n_frames`` synthetic frames via grab()/retrieve() then reports
    grab() == False forever (end of stream).  ``fps`` is what get(CAP_PROP_FPS)
    returns.  Each retrieved image carries a unique fill value so callers can
    assert ordering.
    """

    def __init__(self, fps=30.0, n_frames=1000, opened=True, frame_shape=(4, 4, 3)):
        self._fps = fps
        self._n_frames = n_frames
        self._opened = opened
        self._frame_shape = frame_shape
        self._pos = 0            # index of next frame to grab
        self._grabbed_value = -1  # fill value of the most recently grabbed frame
        self.set_calls = []
        self.released = False

    def isOpened(self):
        return self._opened

    def get(self, prop):
        # We only care about CAP_PROP_FPS in these tests.
        return float(self._fps)

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        return True

    def grab(self):
        if self._pos >= self._n_frames:
            return False
        # Encode the absolute frame index into the frame for ordering checks.
        self._grabbed_value = self._pos
        self._pos += 1
        return True

    def retrieve(self):
        if self._grabbed_value < 0:
            return False, None
        img = np.full(self._frame_shape, self._grabbed_value % 256, dtype=np.uint8)
        return True, img

    def release(self):
        self.released = True


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Make resolve_source / _resolve_youtube inert for every test."""
    monkeypatch.setattr(stream_mod, "resolve_source", lambda src: src)
    monkeypatch.setattr(stream_mod, "_resolve_youtube", lambda url: url)


def _install_capture(monkeypatch, *caps):
    """Patch cv2.VideoCapture to hand out the given FakeCapture(s) in order.

    Returns a list that records the captures actually constructed.
    """
    handed_out = []
    cap_iter = iter(caps)

    def _factory(_resolved):
        cap = next(cap_iter)
        handed_out.append(cap)
        return cap

    monkeypatch.setattr(stream_mod.cv2, "VideoCapture", _factory)
    return handed_out


def _drain(ls, count, timeout=5.0):
    """Pull ``count`` Frame objects from a started Livestream.

    Polls the hand-off queue via frames() with a bounded overall timeout so a
    stuck reader never hangs the suite.
    """
    out = []
    gen = ls.frames()
    deadline = time.time() + timeout
    while len(out) < count:
        if time.time() > deadline:
            pytest.fail(f"timed out after collecting {len(out)}/{count} frames")
        out.append(next(gen))
    return out


# --------------------------------------------------------------------------- #
# 1. Stride computation in open()
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "src_fps, process_fps, expected_stride, expected_src_fps",
    [
        (30.0, 2.0, 15, 30.0),               # 30/2 -> 15
        (25.0, 2.0, 12, 25.0),               # round(12.5) == 12 (banker's rounding)
        (0.0, 2.0, 12, 25.0),                # fps<=0 -> fallback 25 -> 25/2 -> 12
        (float("nan"), 2.0, 12, 25.0),       # NaN -> fallback 25
        (float("inf"), 2.0, 12, 25.0),       # inf -> fallback 25
        (240.0, 2.0, 12, 25.0),              # >120 -> fallback 25
        (60.0, 0.0, 1, 60.0),                # process_fps<=0 -> stride 1
        (60.0, -5.0, 1, 60.0),               # process_fps<0 -> stride 1
        (10.0, 100.0, 1, 10.0),              # round(0.1)=0 -> max(1, .) == 1
    ],
)
def test_open_computes_stride(
    monkeypatch, src_fps, process_fps, expected_stride, expected_src_fps
):
    cap = FakeCapture(fps=src_fps, n_frames=0)  # 0 frames -> reader idles immediately
    _install_capture(monkeypatch, cap)

    ls = Livestream("fake://url", process_fps=process_fps)
    ls.open()
    try:
        assert ls._stride == expected_stride
        assert ls._src_fps == expected_src_fps
    finally:
        ls.close()


def test_open_raises_when_not_opened(monkeypatch):
    cap = FakeCapture(opened=False)
    _install_capture(monkeypatch, cap)
    ls = Livestream("fake://url", process_fps=2.0)
    with pytest.raises(RuntimeError, match="Could not open stream"):
        ls.open()


# --------------------------------------------------------------------------- #
# 2. Synthetic video clock: consecutive ts exactly 1/process_fps apart
# --------------------------------------------------------------------------- #
def test_video_clock_spacing(monkeypatch):
    cap = FakeCapture(fps=30.0, n_frames=10_000)
    _install_capture(monkeypatch, cap)

    ls = Livestream("fake://url", process_fps=2.0)
    ls.open()
    try:
        frames = _drain(ls, 5)
    finally:
        ls.close()

    ts = [f.ts for f in frames]
    # Monotonically increasing.
    assert all(b > a for a, b in zip(ts, ts[1:])), ts
    # Each step is exactly 1/process_fps == 0.5s, independent of wall-clock.
    step = 1.0 / ls.process_fps
    for a, b in zip(ts, ts[1:]):
        assert math.isclose(b - a, step, abs_tol=1e-9), (a, b)
    # ts is anchored at the stream-start origin.
    assert math.isclose(ts[0], ls._stream_start, abs_tol=1e-9)
    # Frame.index increments 1..N.
    assert [f.index for f in frames] == [1, 2, 3, 4, 5]


# --------------------------------------------------------------------------- #
# 3. Monotonic ts across a simulated reconnect
# --------------------------------------------------------------------------- #
def test_ts_monotonic_across_reconnect(monkeypatch):
    cap1 = FakeCapture(fps=30.0, n_frames=10_000)
    cap2 = FakeCapture(fps=30.0, n_frames=10_000)
    _install_capture(monkeypatch, cap1, cap2)

    ls = Livestream("fake://url", process_fps=2.0)
    ls.open()
    try:
        first = _drain(ls, 3)
        start_origin = ls._stream_start
        emit_after_first = ls._emit_index

        # Simulate the reconnect that frames() performs on a stall: close the
        # current reader, sleep, open again.  _stream_start must NOT reset and
        # _emit_index must NOT reset.
        ls.close()
        ls.open()  # hands out cap2

        assert ls._stream_start == start_origin, "origin reset on reconnect"
        assert ls._emit_index == emit_after_first, "emit index reset on reconnect"

        second = _drain(ls, 3)
    finally:
        ls.close()

    all_ts = [f.ts for f in first] + [f.ts for f in second]
    # ts never goes backwards across the reconnect boundary.
    assert all(b > a for a, b in zip(all_ts, all_ts[1:])), all_ts
    # The spacing is preserved across the boundary (no reset to origin).
    step = 1.0 / ls.process_fps
    assert math.isclose(second[0].ts - first[-1].ts, step, abs_tol=1e-9)


# --------------------------------------------------------------------------- #
# 4. Bounded queue evicts the oldest item when full
# --------------------------------------------------------------------------- #
def test_reader_loop_evicts_oldest_when_full(monkeypatch):
    """Drive _reader_loop directly with stride=1 and never drain the queue.

    With _QUEUE_MAXSIZE == 8 and more than 8 emitted frames, the queue must
    hold the 8 NEWEST frames (oldest evicted), and never raise.
    """
    n = stream_mod.Livestream._QUEUE_MAXSIZE  # 8
    extra = 5
    cap = FakeCapture(fps=2.0, n_frames=n + extra)  # 2fps@2fps -> stride 1
    _install_capture(monkeypatch, cap)

    ls = Livestream("fake://url", process_fps=2.0)
    ls.open()
    # Stop the auto-started reader thread; we run the loop ourselves so the
    # queue is never drained concurrently.
    ls.close()
    # close() released cap; re-attach the fake and reset the producer so the
    # loop has frames again, then run the loop to exhaustion in this thread.
    cap._pos = 0
    cap.released = False
    ls._cap = cap
    ls._stop_event.clear()
    ls._emit_index = 0
    ls._reader_loop()  # returns when grab() == False (finite fake)

    q = ls._queue
    assert q.maxsize == n
    assert q.qsize() == n, "queue should be saturated at maxsize, not blocked"

    fill_values = []
    while not q.empty():
        img, _ts = q.get_nowait()
        fill_values.append(int(img.flat[0]))

    # n + extra frames were produced (fill values 0 .. n+extra-1); only the
    # last n survive -> the oldest `extra` were evicted.
    expected = list(range((n + extra) - n, n + extra))
    assert fill_values == expected, fill_values


def test_close_releases_capture_and_stops_thread(monkeypatch):
    cap = FakeCapture(fps=30.0, n_frames=10_000)
    _install_capture(monkeypatch, cap)
    ls = Livestream("fake://url", process_fps=2.0)
    ls.open()
    thread = ls._reader_thread
    ls.close()
    assert cap.released is True
    assert ls._cap is None
    assert thread is not None and not thread.is_alive()

"""
Livestream reader.

Resolves any public livestream URL (YouTube, m3u8, direct file, or a
webcam index) into an OpenCV VideoCapture.  Includes auto-reconnect
and a frame-rate throttle so the rest of the pipeline only sees frames
at `process_fps` instead of the source's native rate.
"""
from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


def _resolve_youtube(url: str) -> str:
    """Return a direct stream URL (HLS / DASH) that OpenCV can open."""
    try:
        import yt_dlp  # local import so the module is importable without it
    except ImportError as exc:  # pragma: no cover - explicit user-facing error
        raise RuntimeError(
            "yt-dlp is required for YouTube URLs.  Install with "
            "`pip install yt-dlp`."
        ) from exc

    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        # Prefer 720p or lower so the laptop CPU survives
        "format": "best[height<=720]/best",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info["url"]


def _try_streamlink(url: str) -> Optional[str]:
    """Fallback that asks streamlink (CLI) for the HLS URL."""
    try:
        out = subprocess.check_output(
            ["streamlink", "--stream-url", url, "best"],
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        return out.decode().strip() or None
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def resolve_source(source: str | int) -> str | int:
    """Translate a user-supplied source into something cv2.VideoCapture
    can open."""
    if isinstance(source, int):
        return source
    s = source.strip()
    if s.isdigit():  # webcam index passed as string
        return int(s)
    if "youtube.com" in s or "youtu.be" in s:
        try:
            return _resolve_youtube(s)
        except Exception as exc:
            log.warning("yt-dlp failed (%s); trying streamlink", exc)
            sl = _try_streamlink(s)
            if sl:
                return sl
            raise
    # Everything else (http(s)://..../stream.m3u8, rtsp://..., file path)
    return s


@dataclass
class Frame:
    """A processed frame with metadata."""
    image: "cv2.Mat"
    ts: float       # wall clock timestamp (seconds since epoch)
    index: int      # monotonically increasing per session


class Livestream:
    """Iterator-style reader with throttling and reconnection.

    Decoding happens in a background thread that drains the FFmpeg
    buffer as fast as the source allows.  Only the MOST RECENT
    decoded frame is kept; the main thread therefore always sees
    near-live material instead of stale buffered frames.

    Without this, a slow per-frame consumer (e.g. YOLO at 0.4 s per
    inference) lets the buffer grow unboundedly and the pipeline
    drifts further and further behind real time — visually it looks
    like the video is jumping forward several seconds at a time.
    """

    def __init__(
        self,
        url: str | int,
        process_fps: float = 2.0,
        reconnect_after_seconds: float = 10.0,
    ):
        self.url = url
        self.target_dt = 1.0 / float(process_fps) if process_fps > 0 else 0.0
        self.reconnect_after = reconnect_after_seconds
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_index = 0

        # Latest-frame slot drained by the background reader.
        self._latest: Optional[Tuple[np.ndarray, float]] = None
        self._latest_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._reader_thread: Optional[threading.Thread] = None
        self._last_decoded_ts = 0.0

    # ----- lifecycle -------------------------------------------------
    def open(self) -> None:
        resolved = resolve_source(self.url)
        log.info("Opening stream: %s", resolved if isinstance(resolved, int)
                 else "<resolved url>")
        self._cap = cv2.VideoCapture(resolved)
        # smaller internal buffer => closer to "live" (best-effort: not
        # honoured by every backend, hence the reader thread below)
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:
            pass
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open stream: {self.url}")

        # start the background drainer
        with self._latest_lock:
            self._latest = None
        self._last_decoded_ts = time.time()
        self._stop_event.clear()
        self._reader_thread = threading.Thread(
            target=self._reader_loop, name="LivestreamReader", daemon=True
        )
        self._reader_thread.start()

    def close(self) -> None:
        self._stop_event.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Livestream":
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()

    # ----- background reader -----------------------------------------
    def _reader_loop(self) -> None:
        """Drain the decoder as fast as the source allows; keep only
        the latest decoded frame.

        We alternate between many cheap `grab()` calls (which only
        advance the decoder position without doing the YUV->BGR
        conversion) and a single `retrieve()` to materialise the most
        recent grabbed frame.  This keeps us at the live edge even on
        a slow CPU and avoids the FFmpeg buffer from filling up.
        """
        SKIP = 3  # how many grabs per decode; tune higher for slow CPUs
        while not self._stop_event.is_set():
            cap = self._cap
            if cap is None:
                time.sleep(0.05)
                continue

            # rapid-grab loop: drop SKIP-1 frames, then decode the next one
            ok = True
            for _ in range(SKIP):
                ok = cap.grab()
                if not ok:
                    break
            if not ok:
                time.sleep(0.05)
                continue
            ok, img = cap.retrieve()
            if not ok or img is None:
                time.sleep(0.02)
                continue
            self._last_decoded_ts = time.time()
            with self._latest_lock:
                self._latest = (img, self._last_decoded_ts)

    # ----- iteration -------------------------------------------------
    def frames(self) -> Iterator[Frame]:
        """Yield throttled, NEAR-LIVE frames forever, reconnecting on failure.

        Crucially we never re-yield a frame older than the previous
        yield: if the throttle ticks but no fresher frame has arrived,
        we wait.  This prevents the visual "teleporting" you'd
        otherwise see when the source delivers in 2-6 s HLS bursts.
        """
        assert self._cap is not None, "Call open() first"
        last_yield = 0.0
        last_yielded_ts = 0.0
        STALE_LOG_AGE = 0.6   # warn if the frame we're about to yield is older
        WAIT_FOR_FRESH = 3.0  # don't block forever waiting for a new frame

        while True:
            now = time.time()

            # Detect stall -> reconnect.
            if now - self._last_decoded_ts > self.reconnect_after:
                log.warning("No frames for %.0fs - reconnecting",
                            self.reconnect_after)
                try:
                    self.close()
                    time.sleep(1.0)
                    self.open()
                    last_yielded_ts = 0.0  # restart fresh tracking
                except Exception as exc:
                    log.error("Reconnect failed: %s", exc)
                    time.sleep(5.0)
                continue

            # Throttle: don't yield faster than process_fps.
            if self.target_dt and (now - last_yield) < self.target_dt:
                time.sleep(min(0.02, self.target_dt - (now - last_yield)))
                continue

            # Wait for a frame that is FRESHER than the one we last
            # emitted.  Otherwise we'd be emitting stale buffer leftovers
            # while the live edge has already moved on.
            fresh_start = time.time()
            while True:
                with self._latest_lock:
                    latest = self._latest
                if latest is not None and latest[1] > last_yielded_ts:
                    break
                if time.time() - fresh_start > WAIT_FOR_FRESH:
                    latest = None
                    break
                time.sleep(0.01)

            if latest is None:
                # nothing fresh within WAIT_FOR_FRESH s; loop and try again
                continue

            img, ts = latest
            age = time.time() - ts
            if age > STALE_LOG_AGE:
                log.debug("yielding frame that's %.0fms old", age * 1000)

            last_yield = time.time()
            last_yielded_ts = ts
            self._frame_index += 1
            yield Frame(image=img, ts=ts, index=self._frame_index)

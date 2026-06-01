"""
Livestream reader.

Resolves any public livestream URL (YouTube, m3u8, direct file, or a
webcam index) into an OpenCV VideoCapture.  Includes auto-reconnect
and frame-count decimation so the rest of the pipeline sees a
CONTINUOUS substream at `process_fps` instead of the source's native
rate.
"""
from __future__ import annotations

import logging
import math
import queue
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
    ts: float       # synthetic VIDEO-time clock (seconds; see Livestream)
    index: int      # monotonically increasing per session


class Livestream:
    """Iterator-style reader optimised for CONTINUITY, not liveness.

    The source (HLS / YouTube) delivers in 2-10 s segment bursts.  A
    naive "keep only the latest frame" drainer would hand the consumer
    frames that are a whole segment apart in video time, which makes
    the picture "teleport" and breaks downstream tracking / dwell math.

    Instead a background thread drains the decoder sequentially with
    cheap `grab()` calls and `retrieve()`s every `stride`-th frame,
    where `stride = round(src_fps / process_fps)`.  The emitted frames
    are therefore consecutive samples of the video, ~1/process_fps of
    VIDEO time apart, regardless of how bursty wall-clock delivery is.

    Frames are handed over through a small bounded queue.  The grab
    loop never blocks on a full queue (that would let FFmpeg drop
    frames uncontrollably); when full it pops the oldest item and
    enqueues the new one, bounding latency while keeping the decoder
    continuously drained.

    `Frame.ts` is a synthetic VIDEO clock (stream_start + emit_index /
    process_fps), NOT wall-clock, so consecutive frames are exactly
    1/process_fps apart even though a burst arrives within milliseconds
    of itself in wall-clock time.
    """

    _QUEUE_MAXSIZE = 8

    def __init__(
        self,
        url: str | int,
        process_fps: float = 2.0,
        reconnect_after_seconds: float = 10.0,
    ):
        self.url = url
        self.process_fps = float(process_fps)
        self.reconnect_after = reconnect_after_seconds
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_index = 0

        # Reader -> consumer hand-off and lifecycle.
        self._queue: "queue.Queue[Tuple[np.ndarray, float]]" = queue.Queue(
            maxsize=self._QUEUE_MAXSIZE
        )
        self._stop_event = threading.Event()
        self._reader_thread: Optional[threading.Thread] = None

        # Decimation + clocks.
        self._src_fps = 25.0
        self._stride = 1
        self._stream_start = 0.0          # video-clock origin (wall-clock secs)
        self._emit_index = 0              # monotonic; NOT reset on reconnect
        self._last_decoded_ts = 0.0       # wall-clock, for stall detection

    # ----- lifecycle -------------------------------------------------
    def open(self) -> None:
        resolved = resolve_source(self.url)
        log.info("Opening stream: %s", resolved if isinstance(resolved, int)
                 else "<resolved url>")
        self._cap = cv2.VideoCapture(resolved)
        # smaller internal buffer => lower latency (best-effort: not
        # honoured by every backend, hence the reader thread below)
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:
            pass
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open stream: {self.url}")

        # Source fps -> frame-count stride.
        src_fps = self._cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(src_fps) or src_fps <= 0 or src_fps > 120:
            src_fps = 25.0
        self._src_fps = src_fps
        if self.process_fps > 0:
            self._stride = max(1, int(round(src_fps / self.process_fps)))
        else:
            self._stride = 1
        log.info("Stream src_fps=%.2f process_fps=%.2f -> stride=%d",
                 src_fps, self.process_fps, self._stride)

        # Fresh hand-off queue + clocks.  Keep the video clock monotonic
        # across reconnects: only set the origin on the very first open.
        self._queue = queue.Queue(maxsize=self._QUEUE_MAXSIZE)
        if self._stream_start == 0.0:
            self._stream_start = time.time()
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
        """Drain the decoder sequentially and decimate by frame stride.

        We `grab()` every frame (cheap: advances the decoder without the
        YUV->BGR conversion) and only `retrieve()` every `stride`-th
        frame, enqueuing it together with its synthetic video timestamp.

        The queue is bounded, but we never block the grab loop on it: a
        blocked grab loop lets FFmpeg silently drop frames and breaks
        continuity.  When the queue is full we drop our OLDEST queued
        frame instead, keeping the decoder continuously drained while
        bounding hand-off latency.

        The decoder is also paced to REAL-TIME: YouTube live HLS exposes a
        multi-minute DVR rewind buffer and OpenCV opens at its START, so an
        unthrottled grab loop would race through that buffer at many times
        real-time, overflow the bounded queue and re-introduce large
        video-time gaps between surviving frames.  We therefore emit at most
        `process_fps` decimated frames per WALL-CLOCK second (schedule local
        to this run, reset on reconnect).  Pacing only ever THROTTLES a reader
        running too fast; a reader that has fallen behind emits immediately
        (no sleep-debt catch-up), so it can never exceed real-time but is
        free to lag on slow CPUs.
        """
        grabbed = 0
        dropped = 0
        pace_start = time.monotonic()
        emitted_local = 0
        while not self._stop_event.is_set():
            cap = self._cap
            if cap is None:
                time.sleep(0.05)
                continue

            if not cap.grab():
                time.sleep(0.05)
                continue
            grabbed += 1
            if grabbed % self._stride != 0:
                continue

            ok, img = cap.retrieve()
            if not ok or img is None:
                time.sleep(0.02)
                continue

            self._last_decoded_ts = time.time()
            if self.process_fps > 0:
                video_ts = self._stream_start + self._emit_index / self.process_fps
            else:
                video_ts = self._last_decoded_ts
            self._emit_index += 1

            # Pace to real-time: throttle a too-fast reader to ~process_fps
            # emissions per wall-clock second.  Never accumulate sleep debt -
            # if we're already behind schedule, emit at once.
            if self.process_fps > 0:
                target = pace_start + emitted_local / self.process_fps
                while not self._stop_event.is_set() and time.monotonic() < target:
                    self._stop_event.wait(min(0.05, target - time.monotonic()))
            emitted_local += 1

            item = (img, video_ts)
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                try:
                    self._queue.get_nowait()  # evict oldest
                except queue.Empty:
                    pass
                dropped += 1
                if dropped % 50 == 0:
                    log.debug("reader queue full; dropped %d frames so far",
                              dropped)
                try:
                    self._queue.put_nowait(item)
                except queue.Full:
                    pass

    # ----- iteration -------------------------------------------------
    def frames(self) -> Iterator[Frame]:
        """Yield CONTINUOUS frames forever, reconnecting on stall.

        The reader thread already decimates to `process_fps`, so we just
        pull from the hand-off queue.  Each `Frame.ts` is the synthetic
        video clock (see the class docstring), so consecutive frames are
        ~1/process_fps of VIDEO time apart even when delivery is bursty.
        """
        assert self._cap is not None, "Call open() first"

        while True:
            # Detect stall -> reconnect.
            if time.time() - self._last_decoded_ts > self.reconnect_after:
                log.warning("No frames for %.0fs - reconnecting",
                            self.reconnect_after)
                try:
                    self.close()
                    time.sleep(1.0)
                    self.open()  # keeps _stream_start / _emit_index monotonic
                except Exception as exc:
                    log.error("Reconnect failed: %s", exc)
                    time.sleep(5.0)
                continue

            try:
                img, video_ts = self._queue.get(timeout=0.5)
            except queue.Empty:
                # nothing queued yet; loop and re-check the stall condition
                continue

            self._frame_index += 1
            yield Frame(image=img, ts=video_ts, index=self._frame_index)

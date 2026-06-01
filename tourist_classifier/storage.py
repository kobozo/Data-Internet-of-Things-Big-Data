"""
Storage layer.

Three sinks, all optional/parallel:
  * frame metrics  -> CSV   (one row per processed frame)
  * finished tracks -> CSV  (one row per person when their track ends)
  * events         -> SQLite + MQTT publish

The MQTT publisher is best-effort: if the broker is unreachable, we
log and keep running.  Real IoT deployments would queue locally and
flush on reconnect; for an academic project we keep it simple.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import queue
import sqlite3
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .classifier import Score
from .events import Event, FrameMetrics
from .tracker import Track

log = logging.getLogger(__name__)


# =====================================================================
# CSV writers
# =====================================================================
class CsvSink:
    """Append-only CSV with header autodetection."""

    def __init__(self, path: str, header: List[str]):
        self.path = path
        self.header = header
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        needs_header = not os.path.exists(path) or os.path.getsize(path) == 0
        self._fh = open(path, "a", newline="", encoding="utf-8")
        self._w = csv.writer(self._fh)
        if needs_header:
            self._w.writerow(header)
            self._fh.flush()

    def write_row(self, row: List) -> None:
        self._w.writerow(row)

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.flush()
        finally:
            self._fh.close()


# =====================================================================
# SQLite events store
# =====================================================================
class EventsDB:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS events (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        ts        REAL    NOT NULL,
        ts_iso    TEXT    NOT NULL,
        type      TEXT    NOT NULL,
        payload   TEXT    NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
    CREATE INDEX IF NOT EXISTS idx_events_ts   ON events(ts);
    """

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()

    def insert(self, ev: Event) -> None:
        iso = datetime.fromtimestamp(ev.ts, tz=timezone.utc).isoformat()
        self.conn.execute(
            "INSERT INTO events(ts, ts_iso, type, payload) VALUES (?,?,?,?)",
            (ev.ts, iso, ev.type, json.dumps(ev.payload)),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


# =====================================================================
# MQTT publisher (non-blocking)
# =====================================================================
class MqttPublisher:
    """Tiny wrapper around paho-mqtt that publishes JSON payloads from
    a background thread."""

    def __init__(self, host: str, port: int, topic_prefix: str,
                 client_id: Optional[str] = None):
        self.host = host
        self.port = port
        self.topic_prefix = topic_prefix.rstrip("/")
        self.client_id = client_id or f"tourist-classifier-{int(time.time())}"
        self._q: queue.Queue = queue.Queue(maxsize=10_000)
        self._stop = threading.Event()
        self._client = None
        self._connected = False
        self._t: Optional[threading.Thread] = None

    # ---------------- lifecycle -------------------------------------
    def start(self) -> None:
        try:
            import paho.mqtt.client as mqtt
        except ImportError:  # pragma: no cover
            log.error("paho-mqtt not installed; MQTT disabled")
            return

        def on_connect(c, userdata, flags, rc, *_):
            self._connected = (rc == 0)
            log.info("MQTT %s (rc=%s)",
                     "connected" if self._connected else "connect_failed", rc)

        def on_disconnect(c, userdata, rc, *_):
            self._connected = False
            log.warning("MQTT disconnected (rc=%s)", rc)

        self._client = mqtt.Client(client_id=self.client_id)
        self._client.on_connect = on_connect
        self._client.on_disconnect = on_disconnect
        try:
            self._client.connect_async(self.host, self.port, keepalive=30)
            self._client.loop_start()
        except Exception as exc:
            log.warning("MQTT connect_async failed: %s", exc)

        self._t = threading.Thread(target=self._drain, daemon=True)
        self._t.start()

    def stop(self) -> None:
        self._stop.set()
        if self._t:
            self._t.join(timeout=2.0)
        if self._client:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:  # pragma: no cover
                pass

    # ---------------- publish ---------------------------------------
    def publish(self, subtopic: str, payload: dict, qos: int = 0) -> None:
        try:
            self._q.put_nowait((subtopic, payload, qos))
        except queue.Full:
            log.warning("MQTT queue full, dropping message")

    def _drain(self) -> None:
        while not self._stop.is_set():
            try:
                subtopic, payload, qos = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            topic = f"{self.topic_prefix}/{subtopic.lstrip('/')}"
            data = json.dumps(payload, default=str)
            if self._client and self._connected:
                try:
                    self._client.publish(topic, data, qos=qos)
                except Exception as exc:  # pragma: no cover
                    log.warning("MQTT publish failed: %s", exc)
            else:
                log.debug("MQTT not connected, dropping %s", topic)


# =====================================================================
# Aggregated façade used by main.py
# =====================================================================
class StorageBundle:
    FRAMES_HEADER = ["ts_iso", "ts", "n_people", "n_tourists", "n_locals",
                     "n_uncertain", "tourist_ratio", "avg_dwell_s",
                     "avg_speed_px"]
    TRACKS_HEADER = ["ts_iso_first", "ts_iso_last", "track_id",
                     "lifetime_s", "avg_speed_px", "max_dwell_s",
                     "has_suitcase", "has_backpack", "has_handbag",
                     "holding_phone_high", "final_score", "final_label",
                     "reasons"]

    def __init__(self, cfg: dict):
        data_dir = cfg["data_dir"]
        os.makedirs(data_dir, exist_ok=True)
        self.frames_csv = CsvSink(os.path.join(data_dir, cfg["frames_csv"]),
                                  self.FRAMES_HEADER)
        self.tracks_csv = CsvSink(os.path.join(data_dir, cfg["tracks_csv"]),
                                  self.TRACKS_HEADER)
        self.events_db = EventsDB(os.path.join(data_dir, cfg["events_db"]))

        mqtt_cfg = cfg.get("mqtt", {})
        self.mqtt: Optional[MqttPublisher] = None
        if mqtt_cfg.get("enabled"):
            self.mqtt = MqttPublisher(
                host=mqtt_cfg["host"],
                port=int(mqtt_cfg.get("port", 1883)),
                topic_prefix=mqtt_cfg["topic_prefix"],
            )
            self.mqtt.start()

    # ---- writers ---------------------------------------------------
    def write_frame(self, m: FrameMetrics) -> None:
        iso = datetime.fromtimestamp(m.ts, tz=timezone.utc).isoformat()
        self.frames_csv.write_row([
            iso, m.ts, m.n_people, m.n_tourists, m.n_locals,
            m.n_uncertain, round(m.tourist_ratio, 3),
            m.avg_dwell, m.avg_speed,
        ])
        self.frames_csv.flush()
        if self.mqtt:
            self.mqtt.publish("metrics", {
                "ts": m.ts, "ts_iso": iso,
                "n_people": m.n_people,
                "n_tourists": m.n_tourists,
                "n_locals": m.n_locals,
                "n_uncertain": m.n_uncertain,
                "tourist_ratio": round(m.tourist_ratio, 3),
                "avg_dwell": m.avg_dwell,
                "avg_speed": m.avg_speed,
            })

    def write_track(self, track: Track,
                    score: Score, ts_now: float) -> None:
        iso_first = datetime.fromtimestamp(
            track.first_seen_ts, tz=timezone.utc).isoformat()
        iso_last = datetime.fromtimestamp(
            track.last_seen_ts, tz=timezone.utc).isoformat()
        self.tracks_csv.write_row([
            iso_first, iso_last, track.track_id,
            round(track.lifetime, 2),
            round(track.avg_speed_px, 2),
            round(track.dwell_seconds, 2),
            int(track.has_suitcase), int(track.has_backpack),
            int(track.has_handbag), int(track.holding_phone_high),
            score.score, score.label,
            "|".join(score.reasons),
        ])
        self.tracks_csv.flush()

    def write_event(self, ev: Event) -> None:
        self.events_db.insert(ev)
        if self.mqtt:
            self.mqtt.publish(f"events/{ev.type}", {
                "ts": ev.ts,
                "type": ev.type,
                "payload": ev.payload,
            })

    # ---- shutdown --------------------------------------------------
    def close(self) -> None:
        self.frames_csv.close()
        self.tracks_csv.close()
        self.events_db.close()
        if self.mqtt:
            self.mqtt.stop()

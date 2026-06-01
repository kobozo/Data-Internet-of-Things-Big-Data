"""
Tiny MQTT subscriber for the screencast / demo.

Use instead of `mosquitto_sub` if that command errors with
'Bad file descriptor' on your macOS install (a known bug in the
Homebrew-shipped mosquitto-clients on macOS Sonoma/Sequoia).

Usage:
    python tools/mqtt_listener.py                  # default: localhost
    python tools/mqtt_listener.py --host test.mosquitto.org
    python tools/mqtt_listener.py --topic 'iot_bigdata/yannick/tourist_classifier/events/#'
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime

try:
    import paho.mqtt.client as mqtt
except ImportError:
    sys.exit("paho-mqtt is required.  Activate your venv and run "
             "`pip install paho-mqtt`.")


# ANSI colour helpers (only used when stdout is a TTY)
RESET  = "\033[0m"  if sys.stdout.isatty() else ""
DIM    = "\033[2m"  if sys.stdout.isatty() else ""
BOLD   = "\033[1m"  if sys.stdout.isatty() else ""
RED    = "\033[31m" if sys.stdout.isatty() else ""
GREEN  = "\033[32m" if sys.stdout.isatty() else ""
YELLOW = "\033[33m" if sys.stdout.isatty() else ""
CYAN   = "\033[36m" if sys.stdout.isatty() else ""


def on_connect(client, userdata, flags, rc, *_):
    # paho-mqtt 1.x: rc is int (0 = ok).
    # paho-mqtt 2.x: rc is a ReasonCode object with .is_failure
    if isinstance(rc, int):
        ok = (rc == 0)
    else:
        ok = not getattr(rc, "is_failure", False)
    if ok:
        print(f"{GREEN}[connected]{RESET}  {userdata['host']}:{userdata['port']}",
              flush=True)
        client.subscribe(userdata["topic"])
        print(f"{GREEN}[subscribed]{RESET} {userdata['topic']}\n", flush=True)
    else:
        print(f"{RED}[connect failed]{RESET} rc={rc}", flush=True)


def on_message(client, userdata, msg):
    ts = datetime.now().strftime("%H:%M:%S")
    topic = msg.topic
    try:
        payload = json.loads(msg.payload.decode())
        pretty  = json.dumps(payload, indent=None, separators=(", ", ": "))
    except Exception:
        pretty = msg.payload.decode("utf-8", errors="replace")

    # colour events differently from metrics
    if "/events/" in topic:
        topic_col = f"{BOLD}{RED}{topic}{RESET}"
    elif "/metrics" in topic:
        topic_col = f"{CYAN}{topic}{RESET}"
    else:
        topic_col = topic
    print(f"{DIM}{ts}{RESET}  {topic_col}  {pretty}", flush=True)


def main():
    p = argparse.ArgumentParser(description="MQTT listener for demo")
    p.add_argument("--host",  default="localhost")
    p.add_argument("--port",  type=int, default=1883)
    p.add_argument("--topic",
                   default="iot_bigdata/yannick/tourist_classifier/#")
    args = p.parse_args()

    userdata = {"host": args.host, "port": args.port, "topic": args.topic}
    # paho-mqtt 2.x default callback API
    try:
        client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"listener-{int(time.time())}",
            userdata=userdata,
        )
    except AttributeError:
        # paho-mqtt 1.x fallback
        client = mqtt.Client(client_id=f"listener-{int(time.time())}",
                             userdata=userdata)
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"Connecting to {args.host}:{args.port} ...", flush=True)
    client.connect(args.host, args.port, keepalive=30)
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nBye.")


if __name__ == "__main__":
    main()

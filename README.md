# Tourist-vs-Local livestream analyser

> Postgraduate Program in IoT & Big Data — Data Extraction from Livestreams
> Author: Yannick Baxter

This project reads a **public livestream of a public place** — by
default the UNESCO-listed **Grand Place in Brussels** — classifies
every detected pedestrian as **TOURIST**, **LOCAL**, or
**UNCERTAIN** based on visible cues (luggage, photo pose, walking
speed, dwell time, group membership), generates **events** when
something notable happens, and stores everything in a shape that
scales to a city-wide IoT pipeline (CSV time-series + MQTT event
bus + SQLite event log).

> The Grand Place is the most-touristed square in Belgium — easily
> several million visitors per year — so the tourist-vs-local signal
> is strong and the demo is visually convincing.

## Why is this useful?

Tourism boards, city councils, retailers and transport operators
spend a lot of money on surveys to estimate the **tourist share** at
any given location and time. A camera that already exists on a public
street can do this passively, in real time, at zero marginal cost.

Concrete questions the data answers, using the **Brussels Grand
Place** default stream:

* what fraction of pedestrians on the Grand Place are tourists, *per
  hour* of day?  (visit.brussels currently relies on hotel-booking
  proxies; this gives a direct head-count.)
* when do tour groups arrive — early morning bus drop-offs, after-
  lunch peaks, evening light-show audiences?
* during a heritage event (e.g. *Tapis de Fleurs*), how much does the
  tourist share rise vs a normal weekend?
* are visitor patterns shifting after the new pedestrianisation rules
  in the Pentagon district?

## Pipeline

```
Livestream  ->  Frame throttle  ->  YOLOv8 detection  ->  Centroid tracker
                                                                 |
                                              Tourist-vs-Local scoring
                                                                 |
                       +-----------------+----------------+------+
                       v                 v                v
                 frames.csv         tracks.csv     event engine
                  (metrics)       (per-person)        |
                                                      v
                                              SQLite + MQTT publish
                                                      |
                                                      v
                                    Dashboards / alerting / analytics
```

The five assignment parts map onto:

| Part | File(s) |
|------|---------|
| 1. Open & display a livestream | `tourist_classifier/stream.py`, display loop in `main.py` |
| 2. Extract useful information | `detector.py`, `tracker.py`, `classifier.py` |
| 3. Generate events | `events.py` |
| 4. Store data | `storage.py` (CSV + SQLite + MQTT) |
| 5. Big-data context | `analysis/analyze.py` + the README section below |

## Installation

The project targets **Python 3.10+** on macOS / Linux / Windows.

```bash
# 1. clone the repository
git clone git@github.com:kobozo/Data-Internet-of-Things-Big-Data.git
cd Data-Internet-of-Things-Big-Data

# 2. create an isolated virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows:  .venv\Scripts\activate

# 3. install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

The first time you start the analyser, **ultralytics** automatically
downloads `yolov8n.pt` (~6 MB) into the working directory.  No
manual model setup needed.

### Optional: MQTT broker

By default the app publishes to the free public broker
`test.mosquitto.org`.  No setup required, but the broker is best-effort
(no auth, no persistence).  If you'd rather run your own:

```bash
docker run -it --rm -p 1883:1883 eclipse-mosquitto
```

Then change `storage.mqtt.host` in `config.yaml` to `localhost`.

### Optional: streamlink fallback

When YouTube rotates its HLS URLs faster than `yt-dlp` can follow,
`streamlink` is used as a backup resolver:

```bash
pip install streamlink     # already in requirements.txt
```

## Running

### Live analysis with the default stream

```bash
python -m tourist_classifier.main --config config.yaml
```

An OpenCV window opens with the live feed, boxed detections coloured
**red** (TOURIST), **green** (LOCAL) or **yellow** (UNCERTAIN), plus a
banner with the running counts.  Press `q` or `ESC` to stop.

### Other sources

```bash
# webcam (useful for desk demos)
python -m tourist_classifier.main --source 0

# any HTTP(S) stream
python -m tourist_classifier.main --source "https://example.com/feed.m3u8"

# headless (no GUI), useful on a server
python -m tourist_classifier.main --headless

# bounded run for the screencast (auto-stop after 5 min)
python -m tourist_classifier.main --max-seconds 300
```

### Inspect MQTT traffic

```bash
mosquitto_sub -h test.mosquitto.org \
  -t "iot_bigdata/yannick/tourist_classifier/#" -v
```

You'll see two streams:
* `.../metrics`  – every processed frame's headcount and tourist ratio
* `.../events/<type>`  – tourist_hotspot, tourist_group_arrived,
  density_spike, quiet_period

### Generate the offline analytics report

```bash
python analysis/analyze.py --data-dir data --out-dir data/report
```

This writes four PNGs and a `summary.md` with breakdowns by hour of
day, dwell-time distribution per class, and an event timeline.

## How the classifier works

For every YOLO person bbox we maintain a lightweight track (centroid
tracker, no deep features) and accumulate evidence:

| Cue | Detected via | Weight |
|-----|--------------|--------|
| suitcase overlapping the person | YOLO class `28` | +3.0 |
| backpack overlapping the person | YOLO class `24` | +1.5 |
| handbag overlapping the person | YOLO class `26` | +0.5 |
| 'photo pose': phone bbox in upper third | YOLO class `67` + geometry | +2.0 |
| dwell time > 3 s | tracker history | +1.5 |
| slow walker (< 2.5 px / step) | tracker history | +1.0 |
| group member (another scoring person < 120 px away) | spatial | +0.5 |

* score ≥ 3.0  → **TOURIST**
* score < 1.0  → **LOCAL**
* otherwise    → **UNCERTAIN**

All weights and thresholds live in `config.yaml` so they can be tuned
per camera (a station camera differs from a park camera).

## Events

Generated by `tourist_classifier/events.py` on a 30-second rolling
window with per-event cooldowns, so the downstream MQTT bus never gets
spammed:

| Event | Trigger |
|-------|---------|
| `tourist_hotspot` | ≥ 5 tourists in last 30 s **and** tourist ratio ≥ 40 % |
| `tourist_group_arrived` | ≥ 3 tourists co-located within 200 px |
| `density_spike` | current head-count ≥ 2× the window baseline |
| `quiet_period` | head-count ≤ 1 for 60 consecutive seconds |

## Big-data context (Part 5)

**What we collect, per camera:**

* a steady ~2 Hz time-series of `(n_people, n_tourists, n_locals,
  avg_dwell, avg_speed)`
* per-person finalised tracks (one row at end-of-life)
* sparse, high-value events on a separate MQTT topic

**Volume math:**

| Layer | 1 camera | 200 cameras (city) | 5 000 cameras (country) |
|-------|----------|---------------------|-------------------------|
| frames.csv rows | ~170 k / day | ~34 M / day | ~864 M / day |
| events / day    | ~50–200 | ~10–40 k | ~250 k–1 M |

**Reference architecture** for the full pipeline:

```
camera_1 ──┐
camera_2 ──┤  paho-MQTT publishers
   ...   ──┤
camera_N ──┘
            │
            v
       Mosquitto / EMQX broker  (topic: iot_bigdata/.../metrics, .../events)
            │
   ┌────────┼─────────────┐
   v                       v
Telegraf / Kafka       Stream processor (Flink / Spark Structured Streaming)
   │                       │
   v                       v
InfluxDB / ClickHouse   alerting (PagerDuty, Slack, dashboards)
   │
   v
Grafana dashboards  +  ad-hoc analytics (the script in `analysis/`)
```

The schemas we already write are the **landing zone** for that
pipeline: `frames.csv` becomes the metrics measurement in InfluxDB,
`events.sqlite` becomes a Kafka topic feeding the alerting layer.
Nothing needs to change in the camera-side code as we scale: each
camera is an independent IoT node publishing to the same broker.

**Stakeholders / consumers of the data:**

* **Tourism boards** – hourly tourist share per location, year-on-year
* **Retailers** on the street – staffing decisions, sign translations
* **Mobility operators** – tour-bus pickup-point load
* **Urban planners** – validate or refute pedestrianisation policies
* **Emergency services** – `density_spike` + `quiet_period` events
  flag crowd surges or sudden evacuations

## Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `Could not open stream` | YouTube rotated the live URL.  Pick a different one in `config.yaml`. |
| `yt-dlp` extraction fails | `pip install -U yt-dlp` (extractor changes weekly). |
| Black window, no detections | Stream is night-mode or low-res.  Try `--source 0` (webcam) to verify the pipeline. |
| MQTT 'connect_failed' | Public broker is throttled.  Run a local broker (`docker run -p 1883:1883 eclipse-mosquitto`). |
| Slow on CPU | Lower `process_fps` in `config.yaml` (1 fps is plenty for analytics). |

## File map

```
.
├── README.md                  # this file
├── requirements.txt
├── config.yaml                # all tunables
├── tourist_classifier/
│   ├── __init__.py
│   ├── main.py                # entry point
│   ├── stream.py              # Part 1: livestream reader
│   ├── detector.py            # Part 2: YOLOv8 wrapper
│   ├── tracker.py             # Part 2: centroid tracker
│   ├── classifier.py          # Part 2: tourist scoring
│   ├── events.py              # Part 3: event engine
│   └── storage.py             # Part 4: CSV + SQLite + MQTT
├── analysis/
│   └── analyze.py             # Part 5: offline analytics report
└── data/                      # collected output (gitignored)
    ├── frames.csv
    ├── tracks.csv
    ├── events.sqlite
    └── report/                # generated PNGs + summary.md
```

## License

Academic project, MIT-style — free to reuse with attribution.

"""
Offline analytics on data collected by the live pipeline.

Reads:
    data/frames.csv      (one row per processed frame)
    data/tracks.csv      (one row per finalised person)
    data/events.sqlite   (event log)

Produces:
    data/report/tourist_ratio_over_time.png
    data/report/people_by_hour.png
    data/report/dwell_distribution.png
    data/report/events_timeline.png
    data/report/summary.md

This is the part of the assignment that demonstrates the **Big Data
context**: even though our single camera produces only ~1 MB of CSV
per day, the same schema scales horizontally across hundreds of
cameras into a city-wide tourism analytics dataset.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import pandas as pd


def load_frames(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ts_iso"])
    df = df.sort_values("ts_iso").reset_index(drop=True)
    return df


def load_tracks(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["ts_iso_first", "ts_iso_last"])
    return df


def load_events(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["ts_iso", "type", "payload"])
    conn = sqlite3.connect(path)
    df = pd.read_sql_query("SELECT ts_iso, type, payload FROM events", conn,
                           parse_dates=["ts_iso"])
    conn.close()
    return df


def plot_tourist_ratio(df_frames: pd.DataFrame, out: Path) -> None:
    if df_frames.empty:
        return
    # 1-minute rolling means
    df = df_frames.set_index("ts_iso").sort_index()
    rolling = df[["n_tourists", "n_locals", "n_people"]].rolling("1min").mean()
    ratio = (rolling["n_tourists"] / rolling["n_people"]).fillna(0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ratio.index, ratio.values, color="#d6481b")
    ax.set_title("Tourist ratio over time (1-min rolling)")
    ax.set_ylabel("tourists / total people")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_people_by_hour(df_frames: pd.DataFrame, out: Path) -> None:
    if df_frames.empty:
        return
    df = df_frames.copy()
    df["hour"] = df["ts_iso"].dt.hour
    agg = df.groupby("hour")[["n_people", "n_tourists", "n_locals"]].mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    agg.plot(kind="bar", stacked=False, ax=ax,
             color=["#444", "#d6481b", "#1c9c43"])
    ax.set_title("Average head-count per hour-of-day")
    ax.set_xlabel("hour")
    ax.set_ylabel("avg people in frame")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_dwell_distribution(df_tracks: pd.DataFrame, out: Path) -> None:
    if df_tracks.empty or "max_dwell_s" not in df_tracks.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    df_tracks.boxplot(column="max_dwell_s", by="final_label", ax=ax,
                      grid=False)
    ax.set_title("Dwell time by classification")
    ax.set_ylabel("max dwell (s)")
    ax.set_xlabel("")
    plt.suptitle("")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_events_timeline(df_events: pd.DataFrame, out: Path) -> None:
    if df_events.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 3.5))
    types = df_events["type"].unique()
    palette = plt.get_cmap("tab10")
    type_to_y = {t: i for i, t in enumerate(types)}
    for t in types:
        sub = df_events[df_events["type"] == t]
        ax.scatter(sub["ts_iso"], [type_to_y[t]] * len(sub),
                   color=palette(type_to_y[t]), label=t, s=40)
    ax.set_yticks(list(type_to_y.values()))
    ax.set_yticklabels(list(type_to_y.keys()))
    ax.set_title("Generated events over time")
    ax.grid(axis="x", alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def write_summary(df_frames: pd.DataFrame, df_tracks: pd.DataFrame,
                  df_events: pd.DataFrame, out: Path) -> None:
    lines = ["# Analysis summary\n"]

    if df_frames.empty:
        lines.append("_No frame data collected yet._\n")
    else:
        start = df_frames["ts_iso"].min()
        end = df_frames["ts_iso"].max()
        n_frames = len(df_frames)
        avg_people = df_frames["n_people"].mean()
        avg_tourist_ratio = (df_frames["n_tourists"].sum()
                             / max(1, df_frames["n_people"].sum()))
        lines += [
            f"* observation window : **{start} -> {end}**",
            f"* frames processed   : **{n_frames:,}**",
            f"* avg people / frame : **{avg_people:.2f}**",
            f"* overall tourist share: **{avg_tourist_ratio:.1%}**",
            "",
        ]

    if not df_tracks.empty:
        n_tracks = len(df_tracks)
        per_label = df_tracks["final_label"].value_counts().to_dict()
        lines += [
            f"* tracks observed    : **{n_tracks:,}**",
            "* breakdown by class :",
        ]
        for k, v in per_label.items():
            lines.append(f"  * `{k}` : {v}")
        lines.append("")

    if not df_events.empty:
        ev_counts = df_events["type"].value_counts().to_dict()
        lines.append("## Events fired")
        for k, v in ev_counts.items():
            lines.append(f"* `{k}` x {v}")
        lines.append("")

    lines += [
        "## Big-data context",
        "",
        "A single street camera at 2 fps produces ~170 k frame-rows per day.",
        "Scaling to 200 cameras across a city = ~34 M rows/day, which is the",
        "natural break-even point where SQLite stops working and a real",
        "OLAP store (ClickHouse, BigQuery, InfluxDB) is required.  The",
        "schema we already write (`ts`, `n_people`, `n_tourists`,",
        "`avg_dwell`) lands directly into a time-series database without",
        "transformation; events go into a separate, lower-volume topic that",
        "drives the alerting layer.",
        "",
    ]
    out.write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="data/report")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_frames = load_frames(data_dir / "frames.csv") \
        if (data_dir / "frames.csv").exists() else pd.DataFrame()
    df_tracks = load_tracks(data_dir / "tracks.csv")
    df_events = load_events(data_dir / "events.sqlite")

    plot_tourist_ratio(df_frames, out_dir / "tourist_ratio_over_time.png")
    plot_people_by_hour(df_frames, out_dir / "people_by_hour.png")
    plot_dwell_distribution(df_tracks, out_dir / "dwell_distribution.png")
    plot_events_timeline(df_events, out_dir / "events_timeline.png")
    write_summary(df_frames, df_tracks, df_events, out_dir / "summary.md")

    print(f"Report written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()

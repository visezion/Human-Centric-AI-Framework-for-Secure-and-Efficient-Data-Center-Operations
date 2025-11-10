#!/usr/bin/env python3
"""
Download the UCI Green DC dataset, normalize it, and optionally replay messages over MQTT.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import List

try:
    from urllib.request import urlretrieve
except ImportError:
    urlretrieve = None  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "datasets" / "raw" / "occupancy_data"
ZIP_PATH = PROJECT_ROOT / "datasets" / "raw" / "occupancy_data.zip"
TRAIN_FILE = RAW_DIR / "datatraining.txt"
PROCESSED_FILE = PROJECT_ROOT / "datasets" / "processed" / "green_dc_training.csv"
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip"


def download_dataset() -> None:
    if TRAIN_FILE.exists():
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if urlretrieve is None:
        raise RuntimeError("urllib not available to download dataset.")
    print(f"Downloading UCI Green DC archive to {ZIP_PATH} ...")
    urlretrieve(DATA_URL, ZIP_PATH)
    print("Extracting archive...")
    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        archive.extractall(RAW_DIR)


def run_prepare(limit: int | None) -> None:
    cmd: List[str] = [
        sys.executable,
        str(PROJECT_ROOT / "datasets" / "prepare_dataset.py"),
        "--dataset",
        "green_dc_uci",
        "--input",
        str(TRAIN_FILE),
        "--output",
        str(PROCESSED_FILE),
    ]
    if limit:
        cmd.extend(["--limit", str(limit)])
    subprocess.run(cmd, check=True)


def publish_to_mqtt(host: str, port: int, topic: str, interval: float) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "datasets" / "prepare_dataset.py"),
        "--dataset",
        "green_dc_uci",
        "--input",
        str(TRAIN_FILE),
        "--output",
        str(PROCESSED_FILE),
        "--publish-mqtt",
        "--mqtt-host",
        host,
        "--mqtt-port",
        str(port),
        "--mqtt-topic",
        topic,
        "--mqtt-interval",
        str(interval),
    ]
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay UCI Green DC dataset through the stack.")
    parser.add_argument("--limit", type=int, help="Only process the first N rows.")
    parser.add_argument("--publish", action="store_true", help="Also publish rows via MQTT.")
    parser.add_argument("--mqtt-host", default="localhost")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--mqtt-topic", default="sensors/public/green_dc")
    parser.add_argument("--mqtt-interval", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_dataset()
    run_prepare(args.limit)
    if args.publish:
        publish_to_mqtt(args.mqtt_host, args.mqtt_port, args.mqtt_topic, args.mqtt_interval)
        print(f"Published normalized rows to MQTT topic {args.mqtt_topic}")
    else:
        print(f"Normalized data stored at {PROCESSED_FILE}")


if __name__ == "__main__":
    main()

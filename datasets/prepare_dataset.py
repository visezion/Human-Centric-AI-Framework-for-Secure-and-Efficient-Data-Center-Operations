#!/usr/bin/env python3
"""
Normalize public datacenter datasets into the stack's canonical telemetry schema
and optionally replay records over MQTT.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

try:
    from paho.mqtt import client as mqtt
except ImportError:  # pragma: no cover
    mqtt = None


CANONICAL_DEFAULTS = {
    "cpu_util": 40.0,
    "memory_util": 55.0,
    "temp_c": 27.0,
    "fan_rpm": 7600.0,
    "power_kw": 2.8,
}

NUMERIC_FIELDS = {"cpu_util", "memory_util", "temp_c", "fan_rpm", "power_kw"}
STRING_FIELDS = {"rack_id", "room", "sensor_type"}


def _is_na(value: Any) -> bool:
    if value is None:
        return True
    if pd is not None:
        try:
            return bool(pd.isna(value))
        except Exception:  # pragma: no cover
            return False
    if isinstance(value, float):
        return value != value  # NaN
    return False


def _load_metadata(dataset_key: str) -> Dict[str, Any]:
    metadata_path = Path(__file__).with_name("metadata.json")
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    if dataset_key not in data:
        raise KeyError(f"Dataset '{dataset_key}' not found in {metadata_path}")
    entry = data[dataset_key]
    entry["dataset_key"] = dataset_key
    return entry


def _coerce_timestamp(value: Any, rule: Dict[str, Any]) -> str:
    if _is_na(value):
        return datetime.now(timezone.utc).isoformat()

    unit = rule.get("unit", "seconds")
    if isinstance(value, (int, float)):
        if unit == "microseconds":
            value = value / 1_000_000
        elif unit == "milliseconds":
            value = value / 1_000
        dt = datetime.fromtimestamp(value, tz=timezone.utc)
        return dt.isoformat()

    if pd is not None:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(ts):
            return datetime.now(timezone.utc).isoformat()
        return ts.isoformat()

    try:
        ts = datetime.fromisoformat(str(value))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.isoformat()
    except ValueError:
        return datetime.now(timezone.utc).isoformat()


def _string_value(source: Dict[str, Any], rule: Dict[str, Any], field: str) -> str:
    value: Optional[str]
    if "column" in rule:
        value = source.get(rule["column"])
    else:
        value = rule.get("value")
    if _is_na(value):
        value = rule.get("default")
    if value is None:
        value = f"{field}-unknown"
    value = str(value)
    if "prefix" in rule:
        value = f"{rule['prefix']}{value}"
    if "suffix" in rule:
        value = f"{value}{rule['suffix']}"
    return value


def _numeric_value(source: Dict[str, Any], rule: Dict[str, Any], field: str) -> float:
    if not rule:
        return CANONICAL_DEFAULTS[field]
    value = None
    if "column" in rule:
        value = source.get(rule["column"])
    if _is_na(value):
        value = rule.get("default", CANONICAL_DEFAULTS[field])
    scale = rule.get("scale", 1.0)
    offset = rule.get("offset", 0.0)
    try:
        numeric = float(value) * float(scale) + float(offset)
    except (TypeError, ValueError):
        numeric = CANONICAL_DEFAULTS[field]
    return float(numeric)


def normalize_dataset(rows: Iterable[Dict[str, Any]], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    mapping = metadata.get("column_mapping", {})
    records: List[Dict[str, Any]] = []
    for row in rows:
        record: Dict[str, Any] = {"source_dataset": metadata["dataset_key"]}
        ts_rule = mapping.get("timestamp", {"unit": "seconds"})
        ts_source = row.get(ts_rule.get("column")) if ts_rule.get("column") else row.get("timestamp")
        record["timestamp"] = _coerce_timestamp(ts_source, ts_rule)

        for field in STRING_FIELDS:
            rule = mapping.get(field, {})
            record[field] = _string_value(row, rule, field)

        for field in NUMERIC_FIELDS:
            rule = mapping.get(field, {})
            record[field] = _numeric_value(row, rule, field)

        records.append(record)
    return records


def _read_input_rows(input_path: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    suffix = input_path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        if pd is None:
            raise RuntimeError("Parquet support requires pandas. Install via pip install pandas>=2.0.")
        frame = pd.read_parquet(input_path)
        if limit:
            frame = frame.head(limit)
        return frame.to_dict(orient="records")

    if pd is not None:
        frame = pd.read_csv(input_path)
        if limit:
            frame = frame.head(limit)
        return frame.to_dict(orient="records")

    # CSV fallback without pandas
    import csv

    rows: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            rows.append(row)
            if limit and idx + 1 >= limit:
                break
    return rows

def _write_output(records: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(output_path, index=False)


def _publish_mqtt(records: List[Dict[str, Any]], host: str, port: int, topic: str, interval: float) -> None:
    if mqtt is None:
        raise RuntimeError("paho-mqtt is required to publish records. Install via datasets/requirements.txt.")

    client = mqtt.Client(client_id=f"dataset-replay-{int(time.time())}")
    client.connect(host, port, keepalive=60)
    client.loop_start()
    try:
        for record in records:
            payload = json.dumps(record)
            client.publish(topic, payload, qos=0, retain=False)
            time.sleep(interval)
    finally:
        client.loop_stop()
        client.disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize public datasets for the Human-Centric AI stack.")
    parser.add_argument("--dataset", required=True, help="Dataset key defined in datasets/metadata.json.")
    parser.add_argument("--input", required=True, type=Path, help="Path to the raw CSV/Parquet file.")
    parser.add_argument("--output", type=Path, help="Destination CSV for normalized telemetry.")
    parser.add_argument("--limit", type=int, help="Optional max number of rows to process.")
    parser.add_argument("--publish-mqtt", action="store_true", help="Replay normalized rows to MQTT.")
    parser.add_argument("--mqtt-host", default="localhost")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--mqtt-topic", default="sensors/public_dataset/telemetry")
    parser.add_argument("--mqtt-interval", type=float, default=0.5, help="Sleep between MQTT messages (seconds).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = _load_metadata(args.dataset)
    rows = _read_input_rows(args.input, args.limit)
    if not rows:
        print("Input dataset is empty. Check --input path.", file=sys.stderr)
        sys.exit(1)

    records = normalize_dataset(rows, metadata)

    output_path = args.output or Path(__file__).with_name("processed") / f"{args.dataset}.csv"
    _write_output(records, output_path)
    print(f"Wrote {len(records)} normalized rows to {output_path}")

    if args.publish_mqtt:
        _publish_mqtt(records, args.mqtt_host, args.mqtt_port, args.mqtt_topic, args.mqtt_interval)
        print(f"Published {len(records)} records to MQTT topic {args.mqtt_topic}")


if __name__ == "__main__":
    main()

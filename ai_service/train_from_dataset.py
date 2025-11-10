#!/usr/bin/env python3
"""
Train an Isolation Forest model from a normalized telemetry CSV and persist it as a joblib artifact
that the ai_service can load at runtime via MODEL_ARTIFACT_PATH.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

FEATURE_ORDER: List[str] = [
    "cpu_util",
    "memory_util",
    "temp_c",
    "fan_rpm",
    "power_kw",
]


def _load_dataset(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    missing = [feature for feature in FEATURE_ORDER if feature not in frame.columns]
    if missing:
        raise ValueError(f"Columns missing from dataset: {', '.join(missing)}")
    frame = frame.dropna(subset=FEATURE_ORDER, how="all")
    frame = frame.fillna(frame.mean(numeric_only=True))
    return frame


def _train_isolation_forest(matrix: np.ndarray) -> IsolationForest:
    model = IsolationForest(
        n_estimators=512,
        contamination=0.08,
        random_state=42,
        bootstrap=True,
        n_jobs=-1,
    )
    model.fit(matrix)
    return model


def train(csv_path: Path, output_path: Path) -> Dict[str, str]:
    frame = _load_dataset(csv_path)
    scaler = StandardScaler()
    raw_matrix = frame[FEATURE_ORDER].values.astype(np.float32)
    scaled_matrix = scaler.fit_transform(raw_matrix)

    model = _train_isolation_forest(scaled_matrix)

    artifact = {
        "version": "1.0",
        "model_type": "isolation_forest",
        "scaler": scaler,
        "model": model,
        "training_matrix": raw_matrix,
        "metadata": {
            "model_backend": "isolation_forest",
            "n_estimators": model.n_estimators,
            "contamination": model.contamination,
            "dataset_rows": raw_matrix.shape[0],
            "dataset_source": str(csv_path),
        },
        "feature_order": FEATURE_ORDER,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    return artifact["metadata"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Isolation Forest model from normalized telemetry CSV.")
    parser.add_argument("--input", required=True, type=Path, help="Path to normalized CSV (e.g., datasets/processed/*.csv).")
    parser.add_argument("--output", required=True, type=Path, help="Destination joblib file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = train(args.input, args.output)
    print(f"Saved artifact to {args.output}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

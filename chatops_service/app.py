import csv
import os
import re
from datetime import datetime
from typing import Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "https://ai.vicezion.com/predict")
INFLUX_URL = os.getenv("INFLUX_URL", "https://influx.vicezion.com")
INFLUX_ORG = os.getenv("INFLUX_ORG", "human-centric")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "telemetry")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")

RACK_REGEX = re.compile(r"rack[-\s]?([0-9]{2}[A-Z])", re.IGNORECASE)
DEFAULT_RACKS = ["rack-12A", "rack-07B", "rack-03C"]

app = FastAPI(title="Human-Centric ChatOps API")


class Question(BaseModel):
    question: str
    rack_hint: Optional[str] = None


async def _query_influx(rack_id: str) -> Optional[Dict[str, str]]:
    flux_query = f'''
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: -30m)
  |> filter(fn: (r) => r._measurement == "rack_telemetry" and r.rack_id == "{rack_id}")
  |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
  |> sort(columns: ["_time"], desc: true)
  |> limit(n:1)
'''
    headers = {
        "Authorization": f"Token {INFLUX_TOKEN}",
        "Content-Type": "application/vnd.flux",
        "Accept": "application/csv",
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(f"{INFLUX_URL}/api/v2/query", params={"org": INFLUX_ORG}, headers=headers, content=flux_query)
        if resp.status_code != 200:
            return None
        cleaned_lines = [line for line in resp.text.splitlines() if line and not line.startswith("#")]
        if not cleaned_lines:
            return None
        reader = csv.DictReader(cleaned_lines)
        for row in reader:
            return row
    return None


async def _call_ai(features: Dict[str, float]) -> Dict:
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(AI_SERVICE_URL, json=features)
        resp.raise_for_status()
        return resp.json()


async def _build_answer(rack_id: str) -> Dict:
    latest = await _query_influx(rack_id)
    if not latest:
        raise HTTPException(status_code=404, detail=f"No telemetry found for {rack_id}")

    features = {
        "cpu_util": float(latest.get("cpu_util", 0.0)),
        "memory_util": float(latest.get("memory_util", 0.0)),
        "temp_c": float(latest.get("temp_c", 0.0)),
        "fan_rpm": float(latest.get("fan_rpm", 0.0)),
        "power_kw": float(latest.get("power_kw", 0.0)),
    }

    ai_result = await _call_ai(features)
    shap_values = ai_result.get("explanations", {}).get("shap", {})
    dominant_feature = max(shap_values, key=lambda k: abs(shap_values[k])) if shap_values else None

    timestamp_ms = int(datetime.utcnow().timestamp() * 1000)
    answer = {
        "rack_id": rack_id,
        "status": "alert" if ai_result.get("is_anomaly") else "normal",
        "anomaly_score": ai_result.get("anomaly_score"),
        "primary_driver": dominant_feature,
        "explanation": f"{rack_id} is {'an anomalous' if ai_result.get('is_anomaly') else 'stable'} node. Highest SHAP impact: {dominant_feature or 'n/a'}.",
        "timestamp_ms": timestamp_ms,
    }
    return {"answer": answer, "metrics": features, "ai_payload": ai_result}


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/question")
async def ask(question: Question) -> Dict:
    rack_hint = question.rack_hint
    if not rack_hint:
        match = RACK_REGEX.search(question.question)
        if match:
            rack_hint = f"rack-{match.group(1).upper()}"
    rack_id = rack_hint or DEFAULT_RACKS[0]
    return await _build_answer(rack_id)


@app.post("/search")
async def search() -> List[str]:
    return DEFAULT_RACKS


@app.post("/query")
async def query(request: Request) -> List[Dict]:
    body = await request.json()
    targets = body.get("targets", [])
    response_frames = []
    for target in targets:
        rack_id = target.get("target") or DEFAULT_RACKS[0]
        payload = target.get("payload") or {}
        data = await _build_answer(rack_id)
        answer = data["answer"]
        if payload.get("mode") != "table":
            response_frames.append(
                {
                    "target": rack_id,
                    "datapoints": [[answer["anomaly_score"], answer["timestamp_ms"]]],
                }
            )
        response_frames.append(
            {
                "type": "table",
                "columns": [
                    {"text": "time", "type": "time"},
                    {"text": "rack_id", "type": "string"},
                    {"text": "status", "type": "string"},
                    {"text": "primary_driver", "type": "string"},
                    {"text": "explanation", "type": "string"},
                ],
                "rows": [
                    [
                        answer["timestamp_ms"],
                        answer["rack_id"],
                        answer["status"],
                        answer["primary_driver"],
                        answer["explanation"],
                    ]
                ],
            }
        )
    return response_frames

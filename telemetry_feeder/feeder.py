import json
import os
import random
import time
from datetime import datetime
from typing import Dict, List

import httpx
import numpy as np
from paho.mqtt import client as mqtt
from tenacity import retry, stop_after_attempt, wait_fixed

RACK_IDS: List[str] = ["rack-12A", "rack-07B", "rack-03C"]
ROOMS = {"rack-12A": "pod-alpha", "rack-07B": "pod-beta", "rack-03C": "pod-gamma"}
SENSOR_TYPES = ["wazuh", "snmp", "zeek"]

MQTT_BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "mqtt.vicezion.com")
MQTT_BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", "1883"))
MQTT_TOPIC_PREFIX = os.getenv("MQTT_TOPIC_PREFIX", "sensors")
MQTT_USERNAME = os.getenv("MQTT_USERNAME")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")
PUBLISH_INTERVAL = float(os.getenv("PUBLISH_INTERVAL_SECONDS", "5"))
AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "https://ai.vicezion.com/predict")

rng = np.random.default_rng(123)


def _simulate_payload(rack_id: str) -> Dict[str, float]:
    temp_spike = 15 if random.random() < 0.15 else 0
    cpu_spike = 30 if random.random() < 0.2 else 0
    payload = {
        "measurement": "rack_telemetry",
        "rack_id": rack_id,
        "room": ROOMS.get(rack_id, "pod-unknown"),
        "sensor_type": random.choice(SENSOR_TYPES),
        "cpu_util": float(rng.normal(45, 15) + cpu_spike),
        "memory_util": float(rng.normal(55, 10)),
        "temp_c": float(rng.normal(27, 5) + temp_spike),
        "fan_rpm": float(rng.normal(7600, 900)),
        "power_kw": float(rng.normal(2.9, 0.5)),
        "humidity": float(rng.normal(40, 4)),
        "status": "nominal",
        "timestamp": datetime.utcnow().isoformat(),
    }
    return payload


@retry(stop=stop_after_attempt(5), wait=wait_fixed(3))
def _build_client() -> mqtt.Client:
    client = mqtt.Client(client_id="human-centric-feeder", clean_session=True, userdata=None, protocol=mqtt.MQTTv311)
    if MQTT_USERNAME and MQTT_PASSWORD:
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, keepalive=60)
    return client


def _call_ai_service(features: Dict[str, float]) -> Dict[str, float]:
    if not AI_SERVICE_URL:
        return {}
    try:
        response = httpx.post(AI_SERVICE_URL, json={k: features[k] for k in ("cpu_util", "memory_util", "temp_c", "fan_rpm", "power_kw")}, timeout=5.0)
        response.raise_for_status()
        return response.json()
    except Exception:
        return {}


def main() -> None:
    client = _build_client()
    while True:
        for rack_id in RACK_IDS:
            payload = _simulate_payload(rack_id)
            ai_result = _call_ai_service(payload)
            if ai_result:
                payload["status"] = "alert" if ai_result.get("is_anomaly") else "nominal"
                payload["anomaly_score"] = ai_result.get("anomaly_score", 0.0)
            topic = f"{MQTT_TOPIC_PREFIX}/{rack_id}/telemetry"
            client.publish(topic, json.dumps(payload), qos=0, retain=False)
        time.sleep(PUBLISH_INTERVAL)


if __name__ == "__main__":
    main()

# Human-Centric AI Framework for Secure and Efficient Data Center Operations

A PhD-ready yet lightweight reference stack that demonstrates end-to-end telemetry ingestion, explainable AI analytics, and a human-centric dashboard for secure and efficient data center operations. By Victor Ayodeji Oluwasusi.

## Stack Overview

| Layer | Component | Purpose |
| --- | --- | --- |
| Data Collection | Telegraf -> InfluxDB | Collect and store server, network, and environmental metrics.
| Security / Logs | (hook) Wazuh, Zeek | Integrate later for intrusion + flow analytics via additional inputs.
| AI & Explainability | Flask AI microservice (PyTorch-ready) + SHAP + LIME + MLflow | Runs anomaly detection, surfaces explanations, and tracks experiments.
| Human Interface | Grafana + (optional) custom UI | Unified dashboard for operators plus space to embed AI responses.

The minimal `docker-compose.yml` brings up Telegraf, InfluxDB, Grafana, the AI microservice, and MLflow on a single workstation or edge server. Everything is 100% open-source and extensible for industrial PoCs.

### Live Telemetry + MQTT Spine

- `mosquitto` exposes an MQTT broker for Wazuh/Zeek/syslog style feeds.
- `telemetry_feeder` simulates racks and publishes JSON payloads (`rack_telemetry` measurement) to MQTT and optionally calls the AI API for enrichment.
- Telegraf now ingests both host metrics and MQTT events before writing everything into InfluxDB.

### Automation + ChatOps

- `chatops_service` offers a FastAPI layer that answers questions such as “Why did rack 12A alert?” by pulling the latest telemetry from InfluxDB, calling the AI inference endpoint, and returning SHAP-backed rationales.
- The same service implements the `simpod-json-datasource` contract so Grafana can render natural-language explanations inside the dashboard without additional glue code.

## Repository Layout

```
.
|-- ai_service/              # Flask API + IsolationForest baseline + SHAP/LIME explanations
|-- grafana/                 # Provisioned datasource + starter dashboard
|-- telegraf/                # Agent configuration for metrics collection
|-- docker-compose.yml       # One-command deployment for the full stack
`-- README.md
```

## Getting Started

1. **Prerequisites:** Docker + Docker Compose (v2). Ensure ports `3000`, `5001`, and `8000` are free.
2. **Launch the stack:**
   ```bash
   docker compose up -d --build
   ```
3. **Access the services:**
   - InfluxDB UI: https://influx.vicezion.com (user `human_admin`, password `changeme123`)
   - Grafana: https://grafana.vicezion.com (user `admin`, password `admin123`)
   - AI API: https://ai.vicezion.com/health and `/predict`
   - MLflow UI: https://mlflow.vicezion.com
   - ChatOps API / JSON datasource: https://chatops.vicezion.com

Telegraf immediately streams host metrics into InfluxDB, which in turn populate the provided Grafana dashboard (`Human-Centric AI`).

## AI Microservice (VAE + Explainability)

The `ai_service` container bootstraps a synthetic dataset, trains a **Variational Auto-Encoder (VAE)** by default (fallback IsolationForest is still available), and exposes `/predict` with:

- Anomaly score + boolean alert
- SHAP contribution values per feature
- LIME explanation list for rapid operator review
- Optional MLflow logging of every inference (disable by omitting `MLFLOW_TRACKING_URI`)

### Example Request

```bash
curl -X POST https://ai.vicezion.com/predict \
  -H "Content-Type: application/json" \
  -d '{
        "cpu_util": 78,
        "memory_util": 83,
        "temp_c": 41,
        "fan_rpm": 10500,
        "power_kw": 4.2
      }'
```

### Example Response (abridged)

```json
{
  "anomaly_score": 0.42,
  "is_anomaly": true,
  "explanations": {
    "shap": {"cpu_util": 0.18, ...},
    "lime": [{"feature": "cpu_util > 70", "weight": 0.22}, ...]
  }
}
```

Use this endpoint to feed Grafana (via the JSON API plugin) or a dedicated React/Vue interface for richer, human-in-the-loop triage.

## Telemetry + MQTT Flow

1. `telemetry_feeder` publishes rack metrics (and AI-enriched anomaly scores) to MQTT topics such as `sensors/rack-12A/telemetry`.
2. Telegraf subscribes via `inputs.mqtt_consumer`, tags each message with `rack_id`, and writes the event into the `rack_telemetry` measurement alongside classic host stats.
3. Grafana renders CPU/memory/fan telemetry plus the AI anomaly score trend per rack.

Swap the simulator with a real Wazuh/Zeek/MQTT publisher by pointing those agents at the Mosquitto broker—Telegraf does not need to change.

## Human-Centric Dashboard + ChatOps

- Grafana now auto-installs the `simpod-json-datasource` plugin and defines a `ChatOps JSON` datasource that queries the FastAPI service.
- The **ChatOps Explanations** table panel issues JSON requests per rack and prints the latest status, anomaly score, and SHAP driver inline with the metrics.
- Ask the ChatOps API directly:

```bash
curl -X POST https://chatops.vicezion.com/question \
  -H "Content-Type: application/json" \
  -d '{"question":"Why did rack 12A alert?"}'
```

The response includes the latest metrics pulled from InfluxDB plus the AI prediction/explanation, making it easy to pipe into Slack/Teams bots.

## Extending the Framework

- **Security telemetry:** Drop Wazuh/Zeek outputs into Kafka/MQTT, ingest with additional Telegraf inputs, and land into InfluxDB or TimescaleDB.
- **Advanced models:** Swap the IsolationForest baseline for your VAE/SENTRY-AI pipeline inside `ai_service/app.py`; the Dockerfile already supplies PyData + MLflow dependencies.
- **Automation:** Wrap the compose stack with Airflow or Ansible for dataset refresh, retraining, and CI-backed experiments.
- **ChatOps:** Pair the AI API with an Ollama-powered assistant so operators can ask "Why did rack 12A alert?" and receive SHAP/LIME context.

## Teardown

```bash
docker compose down -v
```

This removes containers and named volumes (InfluxDB, Grafana, MLflow). Re-run `docker compose up` to recreate the full research environment.

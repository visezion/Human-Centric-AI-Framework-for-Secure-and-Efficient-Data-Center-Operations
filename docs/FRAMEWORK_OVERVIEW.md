# Human-Centric AI Framework – Overview

## What This Framework Does

- **Telemetry ingestion & storage**: Secure MQTT (Mosquitto) + Telegraf ingest any rack or IoT signal, enrich metadata (rack, room, sensor), and persist into InfluxDB.
- **AI-driven insights**: The AI service detects anomalies (VAE or pretrained Isolation Forest), logs to MLflow, and exposes SHAP/LIME explanations via REST.
- **Human-centric experience**: Grafana dashboards visualize metrics + AI context, while the ChatOps API answers “why” questions and powers the Grafana JSON datasource.
- **Automation & DevOps**: GitHub Actions build/push containers, deploy to Portainer, train model artifacts, and replay public datasets for test environments.
- **Operations console**: The Ingestion Portal toggles simulator vs. live mode, autogenerates MQTT credentials, and links to every subsystem.

## Core Services

| Service | Purpose | Default Host/Port |
| --- | --- | --- |
| Mosquitto | MQTT broker with auth for all devices. | `mqtt.vicezion.com:1883` |
| Telegraf | Subscribes to MQTT + host stats, writes to InfluxDB. | internal |
| InfluxDB | Time-series DB for telemetry & ChatOps queries. | `https://influx.vicezion.com` |
| Grafana | Dashboards (Human-Centric overview, JSON datasource). | `https://grafana.vicezion.com` |
| AI Service | `/predict` with anomaly score + SHAP/LIME, loads custom models. | `https://ai.vicezion.com` |
| MLflow | Experiment tracking + inference logs. | `https://mlflow.vicezion.com` |
| ChatOps API | FastAPI answering rack questions, used in Grafana JSON plugin. | `https://chatops.vicezion.com` |
| Prometheus | Scrapes AI + ChatOps metrics for observability. | `http://<host>:9091` |
| Ingestion Portal | Web console to toggle simulator/live, onboard devices, link to services. | `http://<host>:8600` |

## Typical Use Cases

1. **Proof-of-Concept / Demo Labs**
   - Replay public datasets (Google/Alibaba/Azure/UCI) via `datasets/prepare_dataset.py` or `scripts/replay_green_dc.py`.
   - Show Grafana dashboards updating with AI explanations in minutes.

2. **Edge / Data-Center Operations**
   - Onboard sensors using the portal (per-device credentials, TLS via your reverse proxy).
   - Run live telemetry, detect anomalies, and push ChatOps answers to ticketing/Slack.

3. **Model Lifecycle / Research**
   - Retrain Isolation Forests with `ai_service/train_from_dataset.py` or the CI workflow.
   - Track experiments in MLflow, observe inference metrics in Prometheus/Grafana.

4. **Automation Integrations**
   - GitHub Actions deploy images to Portainer or direct Compose hosts.
   - CI jobs can run dataset normalization, training, or replay steps automatically.

## Workflow at a Glance

1. **Data Flow**: Device → MQTT (Mosquitto) → Telegraf → InfluxDB.
2. **Inference**: Grafana / ChatOps call AI service → anomaly score + explanations → MLflow logs.
3. **Operator View**: Grafana dashboards + ChatOps tables show health + context; Prometheus tracks service metrics.
4. **Controls**: Ingestion Portal toggles simulator/live, issues MQTT credentials, links to Grafana/Influx/MLflow/etc.

## Adopting on Your Infrastructure

1. Clone or pull the repo onto your server.
2. Update DNS + TLS for `*.yourdomain.com`, edit `docker-compose.yml` env vars.
3. Generate MQTT secrets (`scripts/generate_mqtt_password.py`) or use the portal.
4. Train/attach your AI model artifact or rely on the included VAE.
5. Run `docker compose up -d` (no MQTT secrets required unless you re-enable authentication).
6. Replay data for testing (`scripts/replay_green_dc.py --publish ...`), confirm Grafana/ChatOps/Prometheus.
7. Point real agents at the broker; manage ingestion + credentials via the portal going forward.

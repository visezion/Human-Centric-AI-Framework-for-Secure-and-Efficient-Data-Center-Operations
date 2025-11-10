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
   - Operations Console: https://portal.vicezion.com (or `http://<host>:8600`) to toggle simulator/live mode, generate MQTT credentials, inspect container health, and deep-link into every subsystem.

Telegraf immediately streams host metrics into InfluxDB, which in turn populate the provided Grafana dashboard (`Human-Centric AI`).

### Public Reference Datasets

Accelerate experimentation by grounding simulations on open telemetry traces:

| Dataset | Description | Link |
| --- | --- | --- |
| Google Cluster Data v2 | Long-running trace of Google Borg jobs, machines, and scheduling decisions. | https://github.com/google/cluster-data |
| Alibaba Cluster Trace 2018 | Production-scale resource usage + workload metadata from Alibaba’s datacenters. | https://github.com/alibaba/clusterdata |
| Azure 2020 Traces (Microsoft) | VM, container, and job statistics from Azure’s 2020 public dataset program. | https://github.com/Azure/AzurePublicDataset |
| Green DC Dataset (UCI) | Energy and thermal telemetry from a green data center testbed suitable for anomaly studies. | https://archive.ics.uci.edu |

Normalize any of these traces with `datasets/prepare_dataset.py` (after `pip install -r datasets/requirements.txt`) to generate canonical telemetry CSVs and optionally replay them over MQTT into the running stack. See `datasets/README.md` for examples.

- Quick start: a toy Google Cluster CSV lives under `datasets/samples/google_sample.csv`. Run `python datasets/prepare_dataset.py --dataset google_cluster_v2 --input datasets/samples/google_sample.csv --publish-mqtt` to emit normalized telemetry and pipe it into the live MQTT spine for testing.
- Prefer automation? `scripts/replay_green_dc.py --publish --mqtt-host mqtt.vicezion.com --mqtt-topic sensors/public/green_dc` downloads the UCI archive, normalizes it, and streams telemetry into the stack in one command.
- Real dataset workflow (Green DC example):
  1. Download and extract the UCI archive:
     ```powershell
     Invoke-WebRequest -Uri https://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip -OutFile datasets/raw/occupancy_data.zip
     Expand-Archive -LiteralPath datasets/raw/occupancy_data.zip -DestinationPath datasets/raw/occupancy_data -Force
     ```
  2. Normalize the CSV so it matches the stack schema:
     ```bash
     python datasets/prepare_dataset.py \
       --dataset green_dc_uci \
       --input datasets/raw/occupancy_data/datatraining.txt \
       --output datasets/processed/green_dc_training.csv
     ```
  3. Retrain the Isolation Forest from that dataset and store the artifact:
     ```bash
     python ai_service/train_from_dataset.py \
       --input datasets/processed/green_dc_training.csv \
       --output ai_service/models/isolation_forest_green.joblib
     ```
  4. Point the AI service at the artifact by setting `MODEL_ARTIFACT_PATH=/app/models/isolation_forest_green.joblib` (already wired in `docker-compose.yml`) so every container loads the pre-trained model instead of bootstrapping a synthetic one.
- Full end-to-end runbooks (automation, security, testing) live in `docs/OPERATIONS.md`.

## AI Microservice (VAE + Explainability)

The `ai_service` container bootstraps a synthetic dataset, trains a **Variational Auto-Encoder (VAE)** by default (fallback IsolationForest is still available), and exposes `/predict` with:

- Anomaly score + boolean alert
- SHAP contribution values per feature
- LIME explanation list for rapid operator review
- Optional MLflow logging of every inference (disable by omitting `MLFLOW_TRACKING_URI`)

Prefer training on real telemetry? Use `python ai_service/train_from_dataset.py --input datasets/processed/<file>.csv --output ai_service/models/<artifact>.joblib` and set `MODEL_ARTIFACT_PATH` (already mounted in the compose file) so the service loads your Isolation Forest weights instead of retraining at startup. You can also trigger `.github/workflows/train-model-artifact.yml` to run the same flow in GitHub Actions and download the resulting artifact for deployment.

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

Swap the simulator with a real Wazuh/Zeek/MQTT publisher by pointing those agents at the Mosquitto broker-Telegraf does not need to change.

### Securing the MQTT Spine

- `mosquitto/mosquitto.conf` requires authentication. Generate a password hash with `python scripts/generate_mqtt_password.py --password <secret>` (writes `mosquitto/passwordfile`), or use the Operations Console UI to create credentials.
- If you re-enable authentication, share the same credentials with external devices or create additional entries in `mosquitto/passwordfile` as needed.
- Use the Operations Console (`http://<host>:8600`) to flip between Simulator/Live modes; it automatically restarts Mosquitto after writing the password file and can start/stop/restart monitored containers.

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

## Testing & Observability

- **Replay:** `python scripts/replay_green_dc.py --publish --mqtt-host mqtt.vicezion.com --mqtt-topic sensors/public/green_dc`.
- **APIs:** `curl https://ai.vicezion.com/health` and `curl -X POST https://chatops.vicezion.com/question -H "Content-Type: application/json" -d '{"question":"Why did rack 12A alert?"}'`.
- **Dashboard:** open Grafana → _Human-Centric Data Center Overview_ to confirm anomaly scores + ChatOps panels update in near real time.
- **Prometheus:** scrape `http://localhost:9091` (or your reverse proxy) for `/metrics` emitted by the AI and ChatOps services; wire Grafana/alerts as needed.
- For detailed checklists, see `docs/OPERATIONS.md`.

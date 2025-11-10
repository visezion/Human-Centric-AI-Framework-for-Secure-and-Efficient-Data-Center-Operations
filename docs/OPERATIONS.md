# Operations, Real-Data Usage, and Testing

This guide explains how to pull public telemetry traces into the stack, retrain the AI service, secure ingestion, and verify everything end to end.

## 1. Prerequisites

- Docker/Compose with access to your `docker-compose.yml`.
- Python 3.11+ with `pip`.
- Valid MQTT credentials (see [Section 4](#4-secure-mqtt-ingestion)).
- Access to DNS records for `*.vicezion.com` and the Portainer endpoint.

## 2. Replay Real Telemetry

### Option A – One-command helper

```bash
python scripts/replay_green_dc.py --publish \
  --mqtt-host mqtt.vicezion.com \
  --mqtt-topic sensors/public/green_dc \
  --mqtt-interval 0.2
```

What it does:
1. Downloads the UCI Green DC archive.
2. Normalizes it via `datasets/prepare_dataset.py`.
3. Streams normalized rows through MQTT so Telegraf → InfluxDB → Grafana/ChatOps light up with real data.

Use `--limit 2000` to shorten a run or omit `--publish` to just write `datasets/processed/green_dc_training.csv`.

### Option B – Manual control

```bash
# 1. Fetch & extract (PowerShell example)
Invoke-WebRequest -Uri https://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip `
  -OutFile datasets/raw/occupancy_data.zip
Expand-Archive -LiteralPath datasets/raw/occupancy_data.zip -DestinationPath datasets/raw/occupancy_data -Force

# 2. Normalize
python datasets/prepare_dataset.py \
  --dataset green_dc_uci \
  --input datasets/raw/occupancy_data/datatraining.txt \
  --output datasets/processed/green_dc_training.csv \
  --limit 5000

# 3. MQTT replay
python datasets/prepare_dataset.py \
  --dataset green_dc_uci \
  --input datasets/raw/occupancy_data/datatraining.txt \
  --publish-mqtt \
  --mqtt-host mqtt.vicezion.com \
  --mqtt-topic sensors/public/green_dc \
  --mqtt-interval 0.25
```

## 3. Train & Deploy Isolation Forest Models

### Manual training

```bash
python ai_service/train_from_dataset.py \
  --input datasets/processed/green_dc_training.csv \
  --output ai_service/models/isolation_forest_green.joblib
```

Set the AI container to load the artifact:

- Ensure `docker-compose.yml` mounts `./ai_service/models` and exports `MODEL_ARTIFACT_PATH=/app/models/isolation_forest_green.joblib`.
- Redeploy (`docker compose up -d ai_service`) and check logs for “Loaded pretrained model artifact…”.

### CI automation

`.github/workflows/train-model-artifact.yml` performs the same steps inside GitHub Actions:
1. Downloads the UCI dataset.
2. Normalizes 5k rows.
3. Trains the Isolation Forest.
4. Uploads `isolation_forest_green_ci.joblib` as a workflow artifact.

Hook it into your release process by downloading the artifact and dropping it in `ai_service/models/` before the Portainer redeploy.

## 4. Secure MQTT Ingestion

1. Generate a password file:
   ```bash
   python scripts/generate_mqtt_password.py --password "super-secret"
   ```
   This writes `mosquitto/passwordfile` with a PBKDF2 hash for user `hcai_operator`.

2. Confirm `mosquitto/mosquitto.conf` has:
   ```
   allow_anonymous false
   password_file /mosquitto/config/passwordfile
   ```

3. Pass credentials to Telegraf & telemetry_feeder via environment:
   ```bash
   export MQTT_PASSWORD="super-secret"
   docker compose up -d telegraf telemetry_feeder
   ```

4. Update IoT publishers to use the same username/password combo.

## 5. Observability & Testing

1. **Prometheus** now scrapes the AI and ChatOps services (`prometheus/prometheus.yml`). Access Prometheus on port `9091` and add Grafana panels pointing at:
   - `http://prometheus:9090`
   - Metrics paths `/metrics` for both services.

2. **Integration verification**
   - Run `scripts/replay_green_dc.py --publish …`.
   - Watch `docker compose logs -f telegraf grafana ai-service chatops` for ingestion, inference, and ChatOps queries.
   - Hit `https://ai.vicezion.com/health` and `https://chatops.vicezion.com/question` (sample payload in README) to confirm the pre-trained model is responding.
   - In Grafana, open the “Human-Centric Data Center Overview” dashboard and confirm:
     - “Rack Anomaly Score” panel updates with the replayed data.
     - “ChatOps Explanations” table populates with context from the AI service.

3. **CI validation**
   - Trigger “Train Isolation Forest Artifact” workflow from GitHub → Actions.
   - Download the produced artifact and redeploy the stack.

4. **Regression checklist**
   - `python datasets/prepare_dataset.py --dataset google_cluster_v2 --input datasets/samples/google_sample.csv` (verifies canonical schema).
   - `python ai_service/train_from_dataset.py --input datasets/processed/green_dc_training.csv --output /tmp/test.joblib` (ensures training script works on your machine).
   - `python scripts/generate_mqtt_password.py --password <pwd>` followed by `docker compose up -d mosquitto telegraf telemetry_feeder` (ensures secured ingestion path works).

With these steps, you can iteratively incorporate new telemetry sources, keep the AI models fresh, and prove the end-to-end flow is healthy before promoting changes.

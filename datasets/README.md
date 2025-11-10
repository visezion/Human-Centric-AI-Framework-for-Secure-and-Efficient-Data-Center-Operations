# Public Dataset Toolkit

The `datasets` folder lets you normalize large-scale telemetry traces (Google, Alibaba, Azure, UCI Green DC) into the canonical schema that powers the Human-Centric AI stack. Once normalized, you can replay the rows through MQTT to exercise the full pipeline without synthetic data.

## Contents

- `metadata.json` &mdash; column mappings, defaults, and links for each supported dataset.
- `prepare_dataset.py` &mdash; CLI tool to normalize and optionally replay telemetry.
- `requirements.txt` &mdash; lightweight dependencies (`pandas`, `paho-mqtt`) for the tool.

## Usage

1. **Download a dataset** from the official source (links in `metadata.json` or the root README table). Extract the CSV/Parquet file you want to replay.
2. **Install tool dependencies** (ideally inside a virtual environment). For CSV-only workflows the stdlib reader is enough, but install pandas if you need Parquet support or large-scale processing:
   ```bash
   pip install -r datasets/requirements.txt
   ```
3. **Normalize the raw file** by referencing the dataset key (`google_cluster_v2`, `alibaba_cluster_2018`, `azure_2020_traces`, `green_dc_uci`):
   ```bash
   python datasets/prepare_dataset.py \
     --dataset google_cluster_v2 \
     --input /data/google/task_usage.csv \
     --limit 50000
   ```
   The script writes a cleaned CSV to `datasets/processed/<dataset>.csv` unless you pass `--output`.
4. **Replay over MQTT (optional)** to feed the running stack:
   ```bash
   python datasets/prepare_dataset.py \
     --dataset alibaba_cluster_2018 \
     --input /data/alibaba/container_usage.csv \
     --publish-mqtt \
     --mqtt-host mqtt.vicezion.com \
     --mqtt-topic sensors/public/telemetry \
     --mqtt-interval 0.25
   ```
   Every normalized row is published as JSON, so Telegraf’s existing `mqtt_consumer` input can ingest it immediately.

## Extending

- **Bulk downloads:** for datasets hosted as archives (e.g., UCI Green DC), use:
  ```powershell
  Invoke-WebRequest -Uri https://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip -OutFile datasets/raw/occupancy_data.zip
  Expand-Archive -LiteralPath datasets/raw/occupancy_data.zip -DestinationPath datasets/raw/occupancy_data -Force
  ```
  Then target `datasets/raw/occupancy_data/datatraining.txt` (or any of the other CSVs) in `prepare_dataset.py`.
- Update `metadata.json` with new datasets or finer-grained mappings (e.g., multiple files per trace). You can add `scale`, `offset`, `prefix`, or constant `value` entries per field.
- Use `--limit` to trim giant files during experimentation or dry runs.
- Pair the normalized CSV with `telemetry_feeder` or other tooling if you need to enrich the rows further before they hit Grafana/Influx.

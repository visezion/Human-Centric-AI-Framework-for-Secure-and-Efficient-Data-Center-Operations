# Model Artifacts

Place trained joblib files here (e.g., `isolation_forest_green.joblib`) and make sure `MODEL_ARTIFACT_PATH` in `docker-compose.yml` points to the one you want the AI service to load.

Artifacts are intentionally git-ignored; generate them via:

```bash
python ai_service/train_from_dataset.py \
  --input datasets/processed/green_dc_training.csv \
  --output ai_service/models/isolation_forest_green.joblib
```

Or download the latest from the `Train Isolation Forest Artifact` GitHub Action.

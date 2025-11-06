import os
import socket
from datetime import datetime
from typing import Dict, List

import mlflow
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, request
from lime import lime_tabular
from mlflow.exceptions import MlflowException
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from waitress import serve

app = Flask(__name__)

FEATURE_ORDER = [
    "cpu_util",
    "memory_util",
    "temp_c",
    "fan_rpm",
    "power_kw",
]

BASELINE_DEFAULTS = {
    "cpu_util": 40.0,
    "memory_util": 55.0,
    "temp_c": 27.0,
    "fan_rpm": 7800.0,
    "power_kw": 2.6,
}

rng = np.random.default_rng(42)


def _simulate_dataset(rows: int = 768) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "cpu_util": rng.normal(45, 12, rows).clip(5, 99),
            "memory_util": rng.normal(60, 15, rows).clip(10, 99),
            "temp_c": rng.normal(29, 4, rows).clip(15, 60),
            "fan_rpm": rng.normal(7500, 800, rows).clip(3000, 12000),
            "power_kw": rng.normal(2.8, 0.6, rows).clip(0.5, 6.5),
        }
    )

    anomaly_rows = max(20, rows // 8)
    anomaly_block = base.sample(anomaly_rows).copy().reset_index(drop=True)
    anomaly_block["cpu_util"] += rng.normal(35, 18, anomaly_rows)
    anomaly_block["temp_c"] += rng.normal(12, 4, anomaly_rows)
    anomaly_block["power_kw"] += rng.normal(1.8, 0.4, anomaly_rows)

    synthetic = pd.concat([base, anomaly_block], ignore_index=True)
    synthetic = synthetic.clip(lower=0)
    return synthetic.sample(frac=1.0, random_state=42).reset_index(drop=True)


class IsolationForestScorer:
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=300,
            contamination=0.08,
            random_state=42,
            bootstrap=True,
        )
        self.training_frame = _simulate_dataset()
        features = self.training_frame[FEATURE_ORDER]
        scaled = self.scaler.fit_transform(features)
        self.model.fit(scaled)
        self.training_matrix = features.values.astype(np.float32)

    def score(self, payload: np.ndarray) -> Dict[str, float]:
        scaled = self.scaler.transform(payload)
        score = float(-self.model.decision_function(scaled)[0])
        label = int(self.model.predict(scaled)[0])
        return {
            "anomaly_score": score,
            "is_anomaly": label == -1,
        }

    def predict_values(self, raw_matrix: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(raw_matrix)
        return -self.model.decision_function(scaled)

    def metadata(self) -> Dict[str, float]:
        return {
            "model_backend": "isolation_forest",
            "n_estimators": 300,
            "contamination": 0.08,
        }


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 3, hidden_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        recon = self.decoder(z)
        return {"recon": recon, "mu": mu, "logvar": logvar}


class VAEAnomalyScorer:
    def __init__(self, latent_dim: int = 3, hidden_dim: int = 32, epochs: int = 40, lr: float = 1e-3) -> None:
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device("cpu")
        self.training_frame = _simulate_dataset()
        self.training_matrix = self.training_frame[FEATURE_ORDER].values.astype(np.float32)
        self.model = VariationalAutoEncoder(len(FEATURE_ORDER), latent_dim, hidden_dim).to(self.device)
        self.threshold = 0.5
        self.training_history: List[float] = []
        self._fit()

    def _fit(self) -> None:
        dataset = TensorDataset(torch.from_numpy(self.training_matrix))
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for _ in range(self.epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                output = self.model(batch)
                recon_loss = F.mse_loss(output["recon"], batch, reduction="mean")
                kld = -0.5 * torch.mean(1 + output["logvar"] - output["mu"].pow(2) - output["logvar"].exp())
                loss = recon_loss + 0.01 * kld
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            self.training_history.append(epoch_loss / max(1, len(loader)))
        self.model.eval()
        with torch.no_grad():
            tensor_data = torch.from_numpy(self.training_matrix).to(self.device)
            output = self.model(tensor_data)
            errors = F.mse_loss(output["recon"], tensor_data, reduction="none").mean(dim=1)
            self.threshold = float(np.percentile(errors.cpu().numpy(), 92))
            self.baseline_errors = errors.cpu().numpy()

    def score(self, payload: np.ndarray) -> Dict[str, float]:
        tensor = torch.from_numpy(payload.astype(np.float32)).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
            error = F.mse_loss(output["recon"], tensor, reduction="none").mean().item()
        return {
            "anomaly_score": float(error),
            "is_anomaly": error > self.threshold,
        }

    def predict_values(self, raw_matrix: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(raw_matrix.astype(np.float32)).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
            errors = F.mse_loss(output["recon"], tensor, reduction="none").mean(dim=1)
        return errors.cpu().numpy()

    def metadata(self) -> Dict[str, float]:
        return {
            "model_backend": "vae",
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "epochs": self.epochs,
            "threshold": self.threshold,
            "final_loss": self.training_history[-1] if self.training_history else 0.0,
        }


USE_VAE = os.getenv("ENABLE_VAE", "false").lower() in {"1", "true", "yes"}
if USE_VAE:
    anomaly_scorer = VAEAnomalyScorer()
else:
    anomaly_scorer = IsolationForestScorer()


def _model_predict(raw_matrix: np.ndarray) -> np.ndarray:
    return anomaly_scorer.predict_values(raw_matrix)


background = shap.utils.sample(anomaly_scorer.training_matrix, min(50, len(anomaly_scorer.training_matrix)), random_state=42)
shap_explainer = shap.KernelExplainer(_model_predict, background)
lime_explainer = lime_tabular.LimeTabularExplainer(
    training_data=anomaly_scorer.training_matrix,
    feature_names=FEATURE_ORDER,
    mode="regression",
)

mlflow_available = False
mlflow_experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", "human-centric-ai")
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

if tracking_uri:
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(mlflow_experiment)
        with mlflow.start_run(run_name="bootstrap-model", description="Human-centric baseline"):
            mlflow.log_params(anomaly_scorer.metadata())
            mlflow.log_metric("training_rows", anomaly_scorer.training_matrix.shape[0])
            if USE_VAE:
                mlflow.log_metric("vae_threshold", anomaly_scorer.metadata().get("threshold", 0.0))
        mlflow_available = True
    except MlflowException as exc:
        app.logger.warning("MLflow unavailable: %s", exc)
    except Exception as exc:  # pragma: no cover
        app.logger.warning("Failed to configure MLflow: %s", exc)


def _vectorize_payload(payload: Dict[str, float]) -> np.ndarray:
    vector = [float(payload.get(feature, BASELINE_DEFAULTS[feature])) for feature in FEATURE_ORDER]
    return np.array([vector], dtype=np.float32)


def _shap_explain(sample: np.ndarray) -> Dict[str, float]:
    values = shap_explainer.shap_values(sample, nsamples=50)
    shap_values = values[0] if isinstance(values, list) else values
    return {feature: float(shap_values[0][idx]) for idx, feature in enumerate(FEATURE_ORDER)}


def _lime_explain(sample: np.ndarray) -> List[Dict[str, float]]:
    explanation = lime_explainer.explain_instance(sample[0], _model_predict, num_features=len(FEATURE_ORDER))
    return [{"feature": feature, "weight": float(weight)} for feature, weight in explanation.as_list()]


def _log_to_mlflow(record: Dict[str, float]) -> None:
    if not mlflow_available:
        return
    try:
        with mlflow.start_run(run_name="online-inference", tags={"host": socket.gethostname()}):
            mlflow.log_metrics({"anomaly_score": record["anomaly_score"]})
            mlflow.log_params({k: record[k] for k in FEATURE_ORDER})
            mlflow.set_tag("is_anomaly", str(record["is_anomaly"]))
    except Exception as exc:  # pragma: no cover
        app.logger.warning("MLflow logging skipped: %s", exc)


@app.route("/health", methods=["GET"])
def health() -> str:
    return "ok"


@app.route("/predict", methods=["POST"])
def predict() -> tuple:
    payload = request.get_json(force=True, silent=True) or {}
    feature_vector = _vectorize_payload(payload)

    model_result = anomaly_scorer.score(feature_vector)
    shap_summary = _shap_explain(feature_vector)
    lime_summary = _lime_explain(feature_vector)

    response = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": os.getenv("MODEL_NAME", "sentry-lite-vae"),
        "features": {feature: float(feature_vector[0][idx]) for idx, feature in enumerate(FEATURE_ORDER)},
        "anomaly_score": model_result["anomaly_score"],
        "is_anomaly": model_result["is_anomaly"],
        "explanations": {
            "shap": shap_summary,
            "lime": lime_summary,
        },
    }

    _log_to_mlflow({**response["features"], **model_result})
    return jsonify(response), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    serve(app, host="0.0.0.0", port=port)

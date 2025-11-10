import base64
import os
import secrets
from pathlib import Path
from typing import Dict, List

import hashlib
import httpx
import docker
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

BASE_DIR = Path(__file__).parent
STATE_FILE = BASE_DIR / "ingestion_state.json"
MOSQUITTO_PWD_FILE = Path("/mosquitto/passwordfile")
DEFAULT_USERNAME = os.getenv("MQTT_USERNAME", "hcai_operator")
FEEDER_CONTAINER = os.getenv("FEEDER_CONTAINER_NAME", "telemetry-feeder")
MOSQUITTO_CONTAINER = os.getenv("MOSQUITTO_CONTAINER_NAME", "mosquitto")
docker_client = docker.DockerClient(base_url=os.getenv("DOCKER_BASE_URL", "unix://var/run/docker.sock"))

app = FastAPI(title="Ingestion Portal")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

templates = Environment(
    loader=FileSystemLoader(BASE_DIR / "templates"),
    autoescape=select_autoescape(["html", "xml"]),
)

SERVICE_DIRECTORY: List[Dict[str, str]] = [
    {
        "name": "Grafana Dashboards",
        "role": "Unified metrics and anomaly visualizations.",
        "url": os.getenv("GRAFANA_URL", "https://grafana.vicezion.com"),
        "cta": "Launch Grafana",
    },
    {
        "name": "InfluxDB",
        "role": "Time-series database for telemetry storage and queries.",
        "url": os.getenv("INFLUX_URL_PUBLIC", "https://influx.vicezion.com"),
        "cta": "Open InfluxDB",
    },
    {
        "name": "AI Service",
        "role": "Anomaly detection + SHAP/LIME explanations.",
        "url": os.getenv("AI_SERVICE_URL_PUBLIC", "https://ai.vicezion.com/docs"),
        "health": os.getenv("AI_SERVICE_HEALTH", "http://ai-service:8000/health"),
        "cta": "View API Docs",
    },
    {
        "name": "ChatOps API",
        "role": "Natural-language Q&A for racks with Grafana datasource support.",
        "url": os.getenv("CHATOPS_URL_PUBLIC", "https://chatops.vicezion.com/docs"),
        "health": os.getenv("CHATOPS_HEALTH", "http://chatops:8500/health"),
        "cta": "View API Docs",
    },
    {
        "name": "MLflow",
        "role": "Experiment tracking and model registry.",
        "url": os.getenv("MLFLOW_URL", "https://mlflow.vicezion.com"),
        "cta": "Open MLflow",
    },
    {
        "name": "Prometheus",
        "role": "Service metrics, used for alerting and Grafana panels.",
        "url": os.getenv("PROMETHEUS_URL", "http://localhost:9091"),
        "cta": "Inspect Metrics",
    },
    {
        "name": "Documentation",
        "role": "Framework overview, operations guide, and dataset workflows.",
        "url": os.getenv("DOCS_URL", "https://github.com/visezion/.../docs"),
        "cta": "Read Docs",
    },
    {
        "name": "Portainer / Deploy",
        "role": "Stack deployment & container lifecycle management.",
        "url": os.getenv("PORTAINER_URL", "https://docker.vicezion.com"),
        "cta": "Open Portainer",
    },
]


def _read_state() -> Dict[str, str]:
    if STATE_FILE.exists():
        return {"mode": STATE_FILE.read_text().strip()}
    return {"mode": "simulator"}


def _write_state(mode: str) -> None:
    STATE_FILE.write_text(mode)


def _hash_password(password: str, iterations: int = 101) -> str:
    salt = secrets.token_bytes(12)
    digest = hashlib.pbkdf2_hmac("sha512", password.encode("utf-8"), salt, iterations)
    return f"$7${iterations}${base64.b64encode(salt).decode()}${base64.b64encode(digest).decode()}"


def _manage_container(name: str, action: str) -> None:
    try:
        container = docker_client.containers.get(name)
    except docker.errors.NotFound:
        raise docker.errors.APIError(f"Container {name} not found")
    container.reload()
    if action == "start" and container.status != "running":
        container.start()
    elif action == "stop" and container.status == "running":
        container.stop()
    elif action == "restart":
        container.restart()


def _generate_password(username: str, password: str) -> None:
    hashed = _hash_password(password)
    entries = []
    if MOSQUITTO_PWD_FILE.exists():
        entries = [line for line in MOSQUITTO_PWD_FILE.read_text().splitlines() if line and not line.startswith(f"{username}:")]
    entries.append(f"{username}:{hashed}")
    MOSQUITTO_PWD_FILE.write_text("\n".join(entries) + "\n")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    state = _read_state()
    services = []
    async with httpx.AsyncClient(timeout=2.0) as client:
        for svc in SERVICE_DIRECTORY:
            status = None
            if "health" in svc:
                try:
                    resp = await client.get(svc["health"])
                    status = "online" if resp.status_code == 200 else "error"
                except Exception:
                    status = "offline"
            services.append({**svc, "status": status})
    template = templates.get_template("index.html")
    return HTMLResponse(
        template.render(
            mode=state["mode"],
            mqtt_host=os.getenv("MQTT_HOST", "mqtt.vicezion.com"),
            mqtt_port=os.getenv("MQTT_PORT", "1883"),
            default_username=DEFAULT_USERNAME,
            services=services,
            framework_doc=os.getenv("FRAMEWORK_DOC_URL", "https://github.com/visezion/.../docs/FRAMEWORK_OVERVIEW.md"),
            operations_doc=os.getenv("OPERATIONS_DOC_URL", "https://github.com/visezion/.../docs/OPERATIONS.md"),
        )
    )


@app.post("/mode")
async def toggle_mode(mode: str = Form(...)):
    _write_state(mode)
    if mode == "simulator":
        _manage_container(FEEDER_CONTAINER, "start")
    else:
        _manage_container(FEEDER_CONTAINER, "stop")
    return RedirectResponse("/", status_code=303)


@app.post("/device")
async def register_device(username: str = Form(...), password: str = Form(...)):
    try:
        _generate_password(username, password)
        _manage_container(MOSQUITTO_CONTAINER, "restart")
        return RedirectResponse("/", status_code=303)
    except docker.errors.DockerException as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

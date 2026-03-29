from fastapi import FastAPI
from fastapi.responses import JSONResponse
from typing import Optional, Union
from .schemas import IngestEvent, DriftAlert
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift.monitor import FairnessDriftMonitor

app = FastAPI(title="Temporal Fairness Drift Monitor API")

# Initialize orchestrator
monitor = FairnessDriftMonitor(
    metric_fn="demographic_parity_difference",
    sensitive_attr="group",
    batch_size=100
)

@app.post("/api/v1/drift/ingest", response_model=DriftAlert, response_model_exclude_none=True)
def ingest_event(event: IngestEvent):
    """
    Ingest a single prediction event.
    Returns a DriftAlert ONLY if drift has been detected.
    """
    alert = monitor.add_element(event.y_true, event.y_pred, event.sensitive_attr)
    if alert:
        return DriftAlert(**alert)
    return JSONResponse(content={"drift_detected": False}, status_code=200)

@app.get("/api/v1/drift/history")
def get_history():
    return {"history": monitor.history}

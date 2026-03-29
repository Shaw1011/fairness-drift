import json
import os
import atexit
import threading
import logging
from contextlib import contextmanager, asynccontextmanager
from typing import Optional, AsyncGenerator

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse

from .schemas import (
    IngestEvent, MultiIngestEvent, DriftAlert, 
    MonitorConfig, HealthResponse
)
from .middleware import RateLimitMiddleware
from drift.monitor import FairnessDriftMonitor, MultiAttributeMonitor
from drift.metrics import METRICS_REGISTRY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Persistence file path (relative to project root)
STATE_FILE = os.environ.get(
    "FAIRNESS_DRIFT_STATE_FILE",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "monitor_state.json")
)

@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator:
    """Modern lifespan handler — replaces deprecated on_event."""
    with locked():
        _load_state()
    logger.info("Fairness Drift Monitor API started.")
    yield
    with locked():
        _save_state()
    logger.info("Fairness Drift Monitor API shutting down.")


app = FastAPI(
    title="Temporal Fairness Drift Monitor API",
    description="Production-grade API for monitoring temporal fairness drift in ML models.",
    version="2.0.0",
    lifespan=lifespan
)

# Add rate limiting middleware
app.add_middleware(
    RateLimitMiddleware,
    max_requests=int(os.environ.get("RATE_LIMIT_MAX", "1000")),
    window_seconds=int(os.environ.get("RATE_LIMIT_WINDOW", "60")),
    max_payload_bytes=int(os.environ.get("MAX_PAYLOAD_BYTES", "10240"))
)

# Thread lock for all monitor access
_lock = threading.Lock()

# Initialize default single-attribute monitor
monitor = FairnessDriftMonitor(
    metric_fn="demographic_parity_difference",
    sensitive_attr="group",
    batch_size=100
)

# Initialize multi-attribute monitor manager
multi_monitor = MultiAttributeMonitor()


@contextmanager
def locked():
    """Context manager for thread-safe monitor access."""
    _lock.acquire()
    try:
        yield
    finally:
        _lock.release()


# --- Persistence ---

def _save_state():
    """Save all monitor states to disk."""
    try:
        state = {
            "default_monitor": monitor.get_state(),
            "default_config": monitor.get_config(),
            "multi_monitors": {},
        }
        for key, m in multi_monitor.monitors.items():
            state["multi_monitors"][key] = {
                "state": m.get_state(),
                "config": m.get_config(),
            }
        
        # Atomic write: write to temp file then rename
        tmp_file = STATE_FILE + ".tmp"
        with open(tmp_file, "w") as f:
            json.dump(state, f, indent=2, default=str)
        
        # os.replace is atomic on most OS
        os.replace(tmp_file, STATE_FILE)
        logger.info(f"State saved to {STATE_FILE}")
    except Exception as e:
        logger.error(f"Failed to save state: {e}")


def _load_state():
    """Load monitor states from disk on startup."""
    global monitor, multi_monitor
    if not os.path.exists(STATE_FILE):
        logger.info("No saved state found, starting fresh.")
        return
    
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
        
        if "default_monitor" in state:
            monitor.load_state(state["default_monitor"])
            logger.info(f"Default monitor state restored: {monitor.total_processed} events processed.")
        
        if "multi_monitors" in state:
            for key, data in state["multi_monitors"].items():
                cfg = data.get("config", {})
                if key not in multi_monitor.monitors:
                    # Recreate monitor from config
                    multi_monitor.add_monitor(
                        sensitive_attr=cfg.get("sensitive_attr", key.split(":")[0]),
                        metric_fn=cfg.get("metric", "demographic_parity_difference"),
                        batch_size=cfg.get("batch_size", 100),
                        baseline_value=cfg.get("baseline_value", 0.0),
                    )
                multi_monitor.monitors[key].load_state(data["state"])
            logger.info(f"Multi-monitors restored: {list(multi_monitor.monitors.keys())}")
                
    except Exception as e:
        logger.error(f"Failed to load state: {e}. Starting fresh.")


# Register shutdown hook as backup for non-graceful termination
atexit.register(_save_state)


# --- Health ---

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint for k8s liveness/readiness probes."""
    with locked():
        processed = {"default": monitor.total_processed}
        for key, m in multi_monitor.monitors.items():
            processed[key] = m.total_processed
        
        return HealthResponse(
            status="healthy",
            monitors_active=1 + len(multi_monitor.monitors),
            total_processed=processed
        )


# --- Single-Attribute Endpoints ---

@app.post("/api/v1/drift/ingest", response_model=DriftAlert, response_model_exclude_none=True)
def ingest_event(event: IngestEvent):
    """
    Ingest a single prediction event into the default monitor.
    Returns a DriftAlert ONLY if drift has been detected.
    """
    with locked():
        alert = monitor.add_element(event.y_true, event.y_pred, event.sensitive_attr)
    
    if alert:
        # Auto-save state on drift detection (important events)
        with locked():
            _save_state()
        return DriftAlert(**alert)
    return JSONResponse(content={"drift_detected": False}, status_code=200)


@app.get("/api/v1/drift/history")
def get_history(
    limit: int = Query(default=100, ge=1, le=10000, description="Number of entries to return"),
    offset: int = Query(default=0, ge=0, description="Offset from the end of history")
):
    """
    Get drift detection history with pagination.
    Returns the most recent entries by default.
    """
    with locked():
        history_list = list(monitor.history)
    
    total = len(history_list)
    start = max(0, total - offset - limit)
    end = total - offset
    page = history_list[start:end]
    
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "history": page
    }


@app.post("/api/v1/drift/reset")
def reset_detectors():
    """Manually reset all detector states in the default monitor."""
    with locked():
        monitor.reset_detectors()
        _save_state()
    return {"status": "reset", "message": "All detectors have been reset."}


@app.get("/api/v1/drift/config")
def get_config():
    """Get current monitor configuration."""
    with locked():
        return monitor.get_config()


# --- Multi-Attribute Endpoints ---

@app.post("/api/v1/multi/monitors")
def create_monitor(config: MonitorConfig):
    """Create a new monitor for a sensitive attribute."""
    with locked():
        try:
            key = multi_monitor.add_monitor(
                sensitive_attr=config.sensitive_attr,
                metric_fn=config.metric_fn,
                batch_size=config.batch_size,
                baseline_value=config.baseline_value,
                adwin_delta=config.adwin_delta,
                adwin_max_window=config.adwin_max_window,
                ewma_lambda=config.ewma_lambda,
                ewma_threshold_multiplier=config.ewma_threshold_multiplier,
                ewma_min_instances=config.ewma_min_instances,
                ph_threshold=config.ph_threshold,
                ph_delta=config.ph_delta,
                ph_min_instances=config.ph_min_instances,
            )
            _save_state()
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))
    
    return {"status": "created", "monitor_key": key}


@app.get("/api/v1/multi/monitors")
def list_monitors():
    """List all active multi-attribute monitors."""
    with locked():
        return {
            "monitors": multi_monitor.list_monitors(),
            "configs": multi_monitor.get_all_configs()
        }


@app.delete("/api/v1/multi/monitors/{monitor_key:path}")
def delete_monitor(monitor_key: str):
    """Remove a multi-attribute monitor."""
    with locked():
        try:
            multi_monitor.remove_monitor(monitor_key)
            _save_state()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
    return {"status": "deleted", "monitor_key": monitor_key}


@app.post("/api/v1/multi/ingest")
def multi_ingest_event(event: MultiIngestEvent):
    """
    Ingest a prediction event across all multi-attribute monitors simultaneously.
    Returns alerts for any monitors that detected drift.
    """
    with locked():
        alerts = multi_monitor.add_element(event.y_true, event.y_pred, event.sensitive_attrs)
    
    if alerts:
        with locked():
            _save_state()
        return {
            "drift_detected": True,
            "alerts": alerts
        }
    return JSONResponse(content={"drift_detected": False, "alerts": {}}, status_code=200)


@app.get("/api/v1/multi/history/{monitor_key:path}")
def get_multi_history(
    monitor_key: str,
    limit: int = Query(default=100, ge=1, le=10000),
    offset: int = Query(default=0, ge=0)
):
    """Get history for a specific multi-attribute monitor."""
    with locked():
        if monitor_key not in multi_monitor.monitors:
            raise HTTPException(status_code=404, detail=f"Monitor '{monitor_key}' not found")
        m = multi_monitor.monitors[monitor_key]
        history_list = list(m.history)
    
    total = len(history_list)
    start = max(0, total - offset - limit)
    end = total - offset
    page = history_list[start:end]
    
    return {
        "monitor_key": monitor_key,
        "total": total,
        "limit": limit,
        "offset": offset,
        "history": page
    }


# --- Utility ---

@app.get("/api/v1/metrics")
def list_available_metrics():
    """List all available fairness metrics."""
    return {"available_metrics": list(METRICS_REGISTRY.keys())}

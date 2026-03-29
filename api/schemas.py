from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class IngestEvent(BaseModel):
    """Single prediction event for fairness monitoring."""
    y_true: int = Field(..., ge=0, le=1, description="Ground truth label (0 or 1)")
    y_pred: int = Field(..., ge=0, le=1, description="Predicted label (0 or 1)")
    sensitive_attr: str = Field(..., min_length=1, max_length=100, description="Demographic group value")

class MultiIngestEvent(BaseModel):
    """Prediction event for multi-attribute monitoring."""
    y_true: int = Field(..., ge=0, le=1, description="Ground truth label (0 or 1)")
    y_pred: int = Field(..., ge=0, le=1, description="Predicted label (0 or 1)")
    sensitive_attrs: Dict[str, str] = Field(
        ..., 
        description="Mapping of attribute names to values, e.g. {'race': 'A', 'gender': 'M'}"
    )

class DriftAlert(BaseModel):
    """Alert payload when drift is detected."""
    drift_detected: bool
    severity: str = Field(..., description="WARNING (1 detector) or CRITICAL (2+ detectors)")
    detector: str
    detectors_triggered: int
    confidence: float
    p_value: float
    affected_group: str
    metric: str
    drift_onset_estimate: int
    current_value: float
    baseline_value: float
    delta: float

class MonitorConfig(BaseModel):
    """Configuration for creating a new monitor."""
    sensitive_attr: str = Field(..., min_length=1, max_length=100)
    metric_fn: str = Field(default="demographic_parity_difference")
    batch_size: int = Field(default=100, ge=10, le=10000)
    baseline_value: float = Field(default=0.0)
    # Detector tuning
    adwin_delta: float = Field(default=0.005, gt=0, lt=1)
    adwin_max_window: int = Field(default=2000, ge=30)
    ewma_lambda: float = Field(default=0.2, gt=0, lt=1)
    ewma_threshold_multiplier: float = Field(default=2.0, gt=0)
    ewma_min_instances: int = Field(default=10, ge=2)
    ph_threshold: float = Field(default=0.05, gt=0)
    ph_delta: float = Field(default=0.005, ge=0)
    ph_min_instances: int = Field(default=10, ge=1)

class HistoryQuery(BaseModel):
    """Query parameters for history endpoint."""
    limit: int = Field(default=100, ge=1, le=10000)
    offset: int = Field(default=0, ge=0)

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    monitors_active: int
    total_processed: Dict[str, int]

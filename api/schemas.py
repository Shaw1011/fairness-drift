from pydantic import BaseModel
from typing import Optional

class IngestEvent(BaseModel):
    y_true: int
    y_pred: int
    sensitive_attr: str

class DriftAlert(BaseModel):
    drift_detected: bool
    detector: str
    confidence: float
    p_value: float
    affected_group: str
    metric: str
    drift_onset_estimate: int
    current_value: float
    baseline_value: float
    delta: float

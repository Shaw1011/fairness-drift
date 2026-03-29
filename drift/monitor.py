from typing import Callable, Any, Dict, Optional, List
from collections import deque
import logging
import math

from .f_adwin import ADWINDetector
from .f_ewma import EWMADetector
from .f_page_hinkley import PageHinkleyDetector
from .metrics import METRICS_REGISTRY

logger = logging.getLogger(__name__)

# Alert severity levels
SEVERITY_WARNING = "WARNING"      # 1 detector fired
SEVERITY_CRITICAL = "CRITICAL"    # 2+ detectors fired


class FairnessDriftMonitor:
    def __init__(
        self, 
        metric_fn: Any, 
        sensitive_attr: str, 
        batch_size: int = 100, 
        baseline_value: float = 0.0,
        max_history: int = 10000,
        # Detector tuning
        adwin_delta: float = 0.005,
        adwin_max_window: int = 2000,
        ewma_lambda: float = 0.2,
        ewma_threshold_multiplier: float = 2.0,
        ewma_min_instances: int = 10,
        ph_threshold: float = 0.05,
        ph_delta: float = 0.005,
        ph_min_instances: int = 10,
    ):
        """
        Orchestrator for the Temporal Fairness Detection Triangle.
        
        :param metric_fn: Callable or string name of the fairness function in METRICS_REGISTRY.
        :param sensitive_attr: Name of the demographic attribute being monitored.
        :param batch_size: Tumbling window size to compute statistical independence (micro-batching).
        :param baseline_value: Historical baseline for delta calculation.
        :param max_history: Maximum number of history entries to retain (prevents OOM).
        :raises ValueError: If metric_fn string is not in METRICS_REGISTRY.
        :raises ValueError: If batch_size < 10 or baseline_value is not finite.
        """
        if isinstance(metric_fn, str):
            if metric_fn not in METRICS_REGISTRY:
                available = ", ".join(METRICS_REGISTRY.keys())
                raise ValueError(
                    f"Unknown metric '{metric_fn}'. Available: {available}"
                )
            self.metric_fn = METRICS_REGISTRY[metric_fn]
            self.metric_name = metric_fn
        else:
            self.metric_fn = metric_fn
            self.metric_name = getattr(metric_fn, '__name__', 'custom')
            
        if batch_size < 10:
            raise ValueError(f"batch_size must be >= 10, got {batch_size}")
        if not math.isfinite(baseline_value):
            raise ValueError(f"baseline_value must be finite, got {baseline_value}")
            
        self.sensitive_attr = sensitive_attr
        self.batch_size = batch_size
        self.baseline_value = baseline_value
        self.max_history = max_history
        
        # Store detector config for serialization/reset
        self._detector_config = {
            "adwin_delta": adwin_delta,
            "adwin_max_window": adwin_max_window,
            "ewma_lambda": ewma_lambda,
            "ewma_threshold_multiplier": ewma_threshold_multiplier,
            "ewma_min_instances": ewma_min_instances,
            "ph_threshold": ph_threshold,
            "ph_delta": ph_delta,
            "ph_min_instances": ph_min_instances,
        }
        
        # Initialize the Detection Triangle
        self.adwin = ADWINDetector(delta=adwin_delta, max_window=adwin_max_window)
        self.ewma = EWMADetector(lambda_=ewma_lambda, threshold_multiplier=ewma_threshold_multiplier, min_instances=ewma_min_instances)
        self.ph = PageHinkleyDetector(threshold=ph_threshold, delta=ph_delta, min_instances=ph_min_instances)
        
        self.current_batch: list = []
        self.total_processed: int = 0
        self.skipped_batches: int = 0  # Track batches skipped due to insufficient data
        self.history: deque = deque(maxlen=max_history)
        
        # Number of detectors for Bonferroni correction
        self._num_detectors = 3
        
    def add_element(self, y_true: int, y_pred: int, target_attr_val: Any) -> Optional[Dict]:
        """
        Ingest a single prediction event.
        Returns a JSON-serializable alert payload if drift is detected, otherwise None.
        
        :param y_true: Ground truth label (0 or 1).
        :param y_pred: Predicted label (0 or 1).
        :param target_attr_val: The demographic group value for this prediction.
        """
        self.current_batch.append({'y_true': y_true, 'y_pred': y_pred, 'attr': target_attr_val})
        self.total_processed += 1
        
        # Evaluate micro-batch
        if len(self.current_batch) == self.batch_size:
            group_stats = {}
            for item in self.current_batch:
                grp = item['attr']
                if grp not in group_stats:
                    group_stats[grp] = {'N': 0, 'pos_preds': 0, 'true_pos': 0, 'actual_pos': 0}
                group_stats[grp]['N'] += 1
                if item['y_pred'] == 1:
                    group_stats[grp]['pos_preds'] += 1
                if item['y_true'] == 1:
                    group_stats[grp]['actual_pos'] += 1
                    if item['y_pred'] == 1:
                        group_stats[grp]['true_pos'] += 1
                        
            metric_val = self.metric_fn(group_stats)
            
            # Reset batch BEFORE potential early return
            self.current_batch = []
            
            # Handle insufficient data (metrics return None when groups are too small)
            if metric_val is None:
                self.skipped_batches += 1
                logger.debug(
                    f"Batch at t={self.total_processed}: insufficient group data, skipping detectors. "
                    f"Groups: { {g: s['N'] for g, s in group_stats.items()} }"
                )
                # Still record history with None for visualization gap awareness
                self.history.append((self.total_processed, None, self.ewma.current_control_limit))
                return None
            
            # Feed metric to Detection Triangle
            adwin_alert = self.adwin.add_element(metric_val)
            ewma_alert = self.ewma.add_element(metric_val)
            ph_alert = self.ph.add_element(metric_val)
            
            # Record visualization history
            self.history.append((self.total_processed, metric_val, self.ewma.current_control_limit))
            
            detectors_triggered = sum([adwin_alert, ewma_alert, ph_alert])
            if detectors_triggered >= 1:
                trigger_names = []
                if adwin_alert: trigger_names.append("FADWIN")
                if ewma_alert: trigger_names.append("F-EWMA")
                if ph_alert: trigger_names.append("Page-Hinkley")
                
                # Synthesize p-values with Bonferroni correction for multiple testing
                p_values = [
                    self.adwin.current_p_value if adwin_alert else 1.0,
                    self.ewma.current_p_value if ewma_alert else 1.0,
                    self.ph.current_p_value if ph_alert else 1.0
                ]
                
                # Bonferroni: multiply minimum p-value by number of tests
                min_p = min(p_values)
                corrected_p = min(min_p * self._num_detectors, 1.0)
                
                # Severity based on number of detectors that fired
                severity = SEVERITY_CRITICAL if detectors_triggered >= 2 else SEVERITY_WARNING
                
                # Clamp drift onset estimate to valid range
                onset_offset = self.adwin.current_onset_index * self.batch_size if adwin_alert else 0
                drift_onset = max(0, self.total_processed - onset_offset)
                
                alert_payload = {
                    "drift_detected": True,
                    "severity": severity,
                    "detector": " + ".join(trigger_names),
                    "detectors_triggered": detectors_triggered,
                    "confidence": round(1.0 - corrected_p, 4),
                    "p_value": round(corrected_p, 5),
                    "affected_group": self.sensitive_attr, 
                    "metric": self.metric_name,
                    "drift_onset_estimate": drift_onset,
                    "current_value": round(metric_val, 4),
                    "baseline_value": self.baseline_value,
                    "delta": round(metric_val - self.baseline_value, 4)
                }
                
                logger.warning(
                    f"DRIFT ALERT [{severity}]: {' + '.join(trigger_names)} fired. "
                    f"Metric={metric_val:.4f}, p={corrected_p:.5f}"
                )
                
                return alert_payload
        return None

    def reset_detectors(self):
        """Manually reset all detector states."""
        cfg = self._detector_config
        self.adwin = ADWINDetector(delta=cfg["adwin_delta"], max_window=cfg["adwin_max_window"])
        self.ewma = EWMADetector(lambda_=cfg["ewma_lambda"], threshold_multiplier=cfg["ewma_threshold_multiplier"], min_instances=cfg["ewma_min_instances"])
        self.ph = PageHinkleyDetector(threshold=cfg["ph_threshold"], delta=cfg["ph_delta"], min_instances=cfg["ph_min_instances"])
        self.current_batch = []
        self.skipped_batches = 0
        logger.info("All detectors manually reset.")

    def get_config(self) -> dict:
        """Return current monitor configuration."""
        return {
            "metric": self.metric_name,
            "sensitive_attr": self.sensitive_attr,
            "batch_size": self.batch_size,
            "baseline_value": self.baseline_value,
            "max_history": self.max_history,
            "total_processed": self.total_processed,
            "skipped_batches": self.skipped_batches,
            "detectors": self._detector_config
        }

    def get_state(self) -> dict:
        """Serialize full monitor state for persistence."""
        return {
            "total_processed": self.total_processed,
            "skipped_batches": self.skipped_batches,
            "current_batch": self.current_batch,
            "history": list(self.history),
            "adwin": self.adwin.get_state(),
            "ewma": self.ewma.get_state(),
            "ph": self.ph.get_state(),
        }
    
    def load_state(self, state: dict):
        """Restore full monitor state from persistence."""
        self.total_processed = state["total_processed"]
        self.skipped_batches = state.get("skipped_batches", 0)
        self.current_batch = state["current_batch"]
        self.history = deque(
            [tuple(h) for h in state["history"]], 
            maxlen=self.max_history
        )
        self.adwin.load_state(state["adwin"])
        self.ewma.load_state(state["ewma"])
        self.ph.load_state(state["ph"])
        logger.info(f"Monitor state restored. total_processed={self.total_processed}")


class MultiAttributeMonitor:
    """
    Manages multiple FairnessDriftMonitor instances, one per sensitive attribute.
    Enables simultaneous monitoring of race, gender, age, etc.
    """
    def __init__(self, configs: List[Dict[str, Any]] = None):
        """
        :param configs: List of monitor configuration dicts. Each must have at minimum:
            - "metric_fn": str or callable
            - "sensitive_attr": str
            Optional keys match FairnessDriftMonitor constructor params.
        """
        self.monitors: Dict[str, FairnessDriftMonitor] = {}
        if configs:
            for cfg in configs:
                self.add_monitor(**cfg)
    
    def add_monitor(self, sensitive_attr: str, metric_fn: Any = "demographic_parity_difference", **kwargs) -> str:
        """
        Add a new monitor for a sensitive attribute.
        Returns a key identifying this monitor.
        """
        key = f"{sensitive_attr}:{metric_fn if isinstance(metric_fn, str) else metric_fn.__name__}"
        if key in self.monitors:
            raise ValueError(f"Monitor already exists for key '{key}'")
        self.monitors[key] = FairnessDriftMonitor(
            metric_fn=metric_fn,
            sensitive_attr=sensitive_attr,
            **kwargs
        )
        logger.info(f"Added monitor: {key}")
        return key
    
    def remove_monitor(self, key: str):
        """Remove a monitor by its key."""
        if key not in self.monitors:
            raise KeyError(f"No monitor found for key '{key}'")
        del self.monitors[key]
        logger.info(f"Removed monitor: {key}")
    
    def add_element(self, y_true: int, y_pred: int, sensitive_attrs: Dict[str, Any]) -> Dict[str, Optional[Dict]]:
        """
        Ingest a prediction event across all monitors.
        
        :param y_true: Ground truth label.
        :param y_pred: Predicted label.
        :param sensitive_attrs: Dict mapping attribute names to values, e.g. {"race": "A", "gender": "M"}
        :returns: Dict mapping monitor key -> alert payload (or None if no alert).
        """
        results = {}
        for key, monitor in self.monitors.items():
            attr_name = monitor.sensitive_attr
            if attr_name in sensitive_attrs:
                alert = monitor.add_element(y_true, y_pred, sensitive_attrs[attr_name])
                if alert:
                    results[key] = alert
        return results
    
    def get_all_configs(self) -> Dict[str, dict]:
        """Return configs for all monitors."""
        return {key: m.get_config() for key, m in self.monitors.items()}
    
    def get_all_states(self) -> Dict[str, dict]:
        """Get serializable state for all monitors."""
        return {key: m.get_state() for key, m in self.monitors.items()}
    
    def load_all_states(self, states: Dict[str, dict]):
        """Restore state for all monitors."""
        for key, state in states.items():
            if key in self.monitors:
                self.monitors[key].load_state(state)
    
    def list_monitors(self) -> List[str]:
        """List all monitor keys."""
        return list(self.monitors.keys())

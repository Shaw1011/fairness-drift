from typing import Callable, Any, Dict, Optional
from .f_adwin import ADWINDetector
from .f_ewma import EWMADetector
from .f_page_hinkley import PageHinkleyDetector
from .metrics import METRICS_REGISTRY

class FairnessDriftMonitor:
    def __init__(self, metric_fn: Any, sensitive_attr: str, batch_size: int = 100, baseline_value: float = 0.0):
        """
        Orchestrator for the Temporal Fairness Detection Triangle.
        
        :param metric_fn: Callable or string name of the fairness function in METRICS_REGISTRY.
        :param sensitive_attr: Name of the demographic attribute being monitored.
        :param batch_size: Tumbling window size to compute statistical independence (micro-batching).
        :param baseline_value: Historical baseline for delta calculation.
        """
        if isinstance(metric_fn, str):
            self.metric_fn = METRICS_REGISTRY[metric_fn]
            self.metric_name = metric_fn
        else:
            self.metric_fn = metric_fn
            self.metric_name = metric_fn.__name__
            
        self.sensitive_attr = sensitive_attr
        self.batch_size = batch_size
        self.baseline_value = baseline_value
        
        # Initialize the Detection Triangle
        self.adwin = ADWINDetector(delta=0.005)
        self.ewma = EWMADetector(lambda_=0.2, threshold_multiplier=2.0, min_instances=10)
        self.ph = PageHinkleyDetector(threshold=0.05, delta=0.005, min_instances=10)
        
        self.current_batch = []
        self.total_processed = 0
        self.history = []  # Time-series history for viz.py
        
    def add_element(self, y_true: int, y_pred: int, target_attr_val: Any) -> Optional[Dict]:
        """
        Ingest a single prediction event.
        Returns a JSON-serializable alert payload if drift is detected, otherwise None.
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
            
            # Feed metric to Detection Triangle
            adwin_alert = self.adwin.add_element(metric_val)
            ewma_alert = self.ewma.add_element(metric_val)
            ph_alert = self.ph.add_element(metric_val)
            
            # Record visualization history with the new EWMA control threshold bounds
            self.history.append((self.total_processed, metric_val, self.ewma.current_control_limit))
            
            # Reset batch
            self.current_batch = []
            
            detectors_triggered = sum([adwin_alert, ewma_alert, ph_alert])
            if detectors_triggered >= 1:
                trigger_names = []
                if adwin_alert: trigger_names.append("FADWIN")
                if ewma_alert: trigger_names.append("F-EWMA")
                if ph_alert: trigger_names.append("Page-Hinkley")
                
                # Synthesize p-values
                p_values = [
                    self.adwin.current_p_value if adwin_alert else 1.0,
                    self.ewma.current_p_value if ewma_alert else 1.0,
                    self.ph.current_p_value if ph_alert else 1.0
                ]
                
                min_p = min(p_values)
                alert_payload = {
                    "drift_detected": True,
                    "detector": " + ".join(trigger_names),
                    "confidence": round(1.0 - min_p, 4),
                    "p_value": round(min_p, 5),
                    "affected_group": self.sensitive_attr, 
                    "metric": self.metric_name,
                    "drift_onset_estimate": self.total_processed - (self.adwin.current_onset_index * self.batch_size if adwin_alert else 0),
                    "current_value": round(metric_val, 4),
                    "baseline_value": self.baseline_value,
                    "delta": round(metric_val - self.baseline_value, 4)
                }
                return alert_payload
        return None

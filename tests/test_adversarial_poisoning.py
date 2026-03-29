import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from drift.monitor import FairnessDriftMonitor
from drift.baseline import NaiveSlidingWindow
from drift.metrics import METRICS_REGISTRY

def test_triangle_beats_baseline_on_slow_poisoning():
    np.random.seed(42)
    
    events = []
    # 5000 events fair
    for i in range(5000):
        grp = np.random.choice(['A', 'B'])
        y_pred = np.random.binomial(1, 0.5)
        y_true = y_pred if np.random.uniform() > 0.1 else 1-y_pred
        events.append((y_true, y_pred, grp))
        
    # 5000 events slow poisoning
    for i in range(5000, 10000):
        grp = np.random.choice(['A', 'B'])
        if grp == 'B':
            drift_factor = min(0.4, (i - 5000) / 3000.0 * 0.4)
            y_pred = np.random.binomial(1, max(0.0, 0.5 - drift_factor))
        else:
            y_pred = np.random.binomial(1, 0.5)
        y_true = y_pred if np.random.uniform() > 0.1 else 1-y_pred
        events.append((y_true, y_pred, grp))
        
    baseline = NaiveSlidingWindow(window_size=1000, threshold=0.15, metric_fn=METRICS_REGISTRY['demographic_parity_difference'])
    baseline_latency = float('inf')
    
    for t, (y_t, y_p, attr) in enumerate(events):
        if baseline.add_element(y_t, y_p, attr) and t >= 5000 and baseline_latency == float('inf'):
            baseline_latency = t - 5000
            break
            
    monitor = FairnessDriftMonitor("demographic_parity_difference", "group", batch_size=100)
    monitor_latency = float('inf')
    triggered_detectors = []
    
    for t, (y_t, y_p, attr) in enumerate(events):
        alert = monitor.add_element(y_t, y_p, attr)
        if alert and t >= 5000 and monitor_latency == float('inf'):
            monitor_latency = t - 5000
            triggered_detectors.append(alert['detector'])
            break
            
    assert monitor_latency < baseline_latency, f"Monitor latency ({monitor_latency}) should be less than baseline latency ({baseline_latency})"
    assert len(triggered_detectors) > 0

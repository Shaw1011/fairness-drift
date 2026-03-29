import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from drift.monitor import FairnessDriftMonitor
from drift.baseline import NaiveSlidingWindow
from drift.metrics import METRICS_REGISTRY
from drift.viz import generate_drift_plot

def run_synthetic_benchmark():
    np.random.seed(42)  # For reproducible papers
    
    # Simulate an adversary dropping approval rate for Group B
    # Total samples: 10,000.  Drift starts exactly at t = 5,000.
    events = []
    
    # t < 5000: Fair behavior
    for i in range(5000):
        grp = np.random.choice(['A', 'B'])
        y_pred = np.random.binomial(1, 0.5)
        y_true = y_pred if np.random.uniform() > 0.1 else 1-y_pred
        events.append((y_true, y_pred, grp))
        
    # t >= 5000: Slow monotonic drift injected (Simulates slow-drip attack)
    # Group B approval rate slowly linearly drops from 0.5 to 0.1 over 3000 steps.
    for i in range(5000, 10000):
        grp = np.random.choice(['A', 'B'])
        if grp == 'B':
            drift_factor = min(0.4, (i - 5000) / 3000.0 * 0.4)
            y_pred = np.random.binomial(1, max(0.0, 0.5 - drift_factor))
        else:
            y_pred = np.random.binomial(1, 0.5)
            
        y_true = y_pred if np.random.uniform() > 0.1 else 1-y_pred
        events.append((y_true, y_pred, grp))
        
    print(f"Total Events: {len(events)}. Drift injected at t=5000")
    print("-" * 50)
    
    # 1. Run naive baseline (duct-tape threshold)
    baseline = NaiveSlidingWindow(window_size=1000, threshold=0.15, metric_fn=METRICS_REGISTRY['demographic_parity_difference'])
    baseline_latency = None
    
    for t, (y_t, y_p, attr) in enumerate(events):
        alert = baseline.add_element(y_t, y_p, attr)
        if alert and t >= 5000 and baseline_latency is None:
            baseline_latency = t - 5000
            print(f"[Naive Baseline] Triggered at t={t}")
            print(f"                 Latency: {baseline_latency} steps")
            break
            
    if baseline_latency is None:
        print("[Naive Baseline] FAILED to detect drift within 10k steps.")
        baseline_latency = 5000 # Max penalty for calculation
            
    print("-" * 50)
    # 2. Run our Detection Triangle
    monitor = FairnessDriftMonitor(metric_fn="demographic_parity_difference", sensitive_attr="group", batch_size=100, baseline_value=0.0)
    monitor_latency = None
    alerts_for_viz = []
    
    for t, (y_t, y_p, attr) in enumerate(events):
        alert = monitor.add_element(y_t, y_p, attr)
        if alert:
            alerts_for_viz.append((t, alert['detector'], alert['current_value']))
            if t >= 5000 and monitor_latency is None:
                monitor_latency = t - 5000
                print(f"[Detection Triangle] Triggered at t={t}")
                print(f"                     Latency: {monitor_latency} steps")
                print(f"                     Detector: {alert['detector']} | Conf: {alert['confidence']}")
                
    print("-" * 50)
    if monitor_latency and baseline_latency:
        pct_improvement = (baseline_latency - monitor_latency) / float(baseline_latency) * 100
        print(f"CONCLUSION: Model is {pct_improvement:.1f}% faster at detecting data poisoning attacks.")

    # Dump graph for paper Figure 1
    generate_drift_plot(monitor.history, alerts=alerts_for_viz, title="Temporal Fairness Drift: Adversarial Attack vs Detection", filename="paper/figures/adversarial_drift.png")

if __name__ == "__main__":
    run_synthetic_benchmark()

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from drift.monitor import FairnessDriftMonitor
from drift.viz import generate_drift_plot

def simulate_compas_stream(n_samples=10000):
    """
    Simulates a COMPAS-like dataset stream with groups 'Caucasian' and 'African-American'.
    Replicates the disparate impact and demographic parity issues found in recidivism scoring.
    """
    np.random.seed(42)
    print("Simulating COMPAS-like data stream...")
    
    events = []
    
    # Phase 1: Historical Baseline Bias (t < 5000)
    for i in range(5000):
        grp = np.random.choice(['Caucasian', 'African-American'], p=[0.4, 0.6])
        if grp == 'Caucasian':
            y_pred = np.random.binomial(1, 0.3)
        else:
            y_pred = np.random.binomial(1, 0.45)
            
        y_true = y_pred if np.random.uniform() > 0.3 else 1-y_pred
        events.append((y_true, y_pred, grp))
        
    # Phase 2: Post-drift (t >= 5000)
    # Drift event: A new policy heavily inflates the predicted risk for the minority group.
    for i in range(5000, 10000):
        grp = np.random.choice(['Caucasian', 'African-American'], p=[0.4, 0.6])
        if grp == 'Caucasian':
            y_pred = np.random.binomial(1, 0.3)
        else:
            y_pred = np.random.binomial(1, 0.75) # Sudden spike
            
        y_true = y_pred if np.random.uniform() > 0.3 else 1-y_pred
        events.append((y_true, y_pred, grp))
        
    return events

def run_compas_benchmark():
    events = simulate_compas_stream()
    
    monitor = FairnessDriftMonitor("demographic_parity_difference", "race", batch_size=100)
    
    alerts_for_viz = []
    drift_detected_at = None
    
    for t, (y_t, y_p, attr) in enumerate(events):
        alert = monitor.add_element(y_t, y_p, attr)
        if alert:
            alerts_for_viz.append((t, alert['detector'], alert['current_value']))
            if t >= 5000 and drift_detected_at is None:
                drift_detected_at = t
                print(f"[COMPAS Simulator] Real-world proxy drift detected at t={t}")
                print(f"                   Latency: {t - 5000} steps")
                print(f"                   Detectors: {alert['detector']}")
                print(f"                   P-Value: {alert['p_value']}")
                
    if drift_detected_at is None:
        print("[COMPAS Simulator] FAILED to detect the disparate impact surge.")
        
    generate_drift_plot(
        monitor.history, 
        alerts=alerts_for_viz, 
        title="COMPAS Disparate Impact Surge (Adverse Feature Drift)", 
        filename="paper/figures/compas_drift.png"
    )

if __name__ == "__main__":
    run_compas_benchmark()

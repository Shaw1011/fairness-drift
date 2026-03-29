import matplotlib.pyplot as plt
import os

def generate_drift_plot(history, alerts=None, title="Temporal Fairness Drift", filename="paper/figures/drift_plot.png"):
    """
    Generates a time-series plot proving the drift detection points.
    history: list of tuples (time_step, metric_val, ewma_control_limit)
    alerts: list of tuples (time_step, detector_name, metric_val) for visual markers
    """
    if not history:
        print("Empty history, skipping plot.")
        return
        
    t_vals = [x[0] for x in history]
    m_vals = [x[1] for x in history]
    c_vals = [x[2] for x in history]  # EWMA controls
    
    plt.figure(figsize=(10, 5))
    
    # Plot real fairness metric
    plt.plot(t_vals, m_vals, label="Metric Value", color="#1f77b4", linewidth=1.5, alpha=0.9)
    plt.plot(t_vals, c_vals, label="EWMA Control Limit", color="red", linestyle="--", alpha=0.7)
    
    # Baseline
    plt.axhline(y=0.0, color='gray', linestyle=':', label="Ideal Baseline")
    
    # Mark Alert Points
    if alerts:
        for alert_t, detector, val in alerts:
            plt.scatter(alert_t, val, color='red', s=100, zorder=5)
            plt.annotate(f"Drift: {detector}", (alert_t, val), textcoords="offset points", xytext=(0,10), ha='center', color='darkred')
            
    plt.title(title, fontweight='bold')
    plt.xlabel("Predictions Processed")
    plt.ylabel("Disparity Metric Value")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Plot saved to {filename}")

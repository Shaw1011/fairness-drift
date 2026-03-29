import numpy as np
from drift.f_ewma import EWMADetector

def test_ewma_detects_slow_shift():
    np.random.seed(42)
    ewma = EWMADetector(lambda_=0.2, threshold_multiplier=3.0)
    alerts = 0
    
    # Feed noisy values to ensure variance > 0
    for _ in range(200):
        val = np.random.normal(0.1, 0.02)
        if ewma.add_element(max(0, val)): alerts += 1
        
    assert alerts == 0, f"Expected 0 false alarms during baseline stabilization, got {alerts}"
    
    # Shift upwards gradually
    reset_happened = False
    for i in range(100):
        val = np.random.normal(0.1 + (i/100)*0.5, 0.02)
        if ewma.add_element(val): 
            alerts += 1
            if ewma.z == ewma.mean:
                reset_happened = True
        
    assert alerts > 0, "Failed to detect the slow drift"
    assert reset_happened, "Reset bug: z should anchor to the current mean."

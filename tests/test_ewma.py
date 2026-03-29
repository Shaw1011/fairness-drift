import numpy as np
import pytest
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
    for i in range(100):
        val = np.random.normal(0.1 + (i/100)*0.5, 0.02)
        if ewma.add_element(val): 
            alerts += 1
        
    assert alerts > 0, "Failed to detect the slow drift"


def test_ewma_full_reset_detects_repeated_drift():
    """After detecting drift, the EWMA should fully reset and detect subsequent drifts equally fast."""
    np.random.seed(42)
    ewma = EWMADetector(lambda_=0.2, threshold_multiplier=3.0)
    
    detection_times = []
    
    for episode in range(3):
        # Stable phase
        for _ in range(100):
            ewma.add_element(np.random.normal(0.1, 0.02))
        
        # Drift phase — measure how long to detect
        steps_to_detect = None
        for i in range(200):
            val = np.random.normal(0.1 + (i/50)*0.3, 0.02)
            if ewma.add_element(val):
                steps_to_detect = i
                break
        
        assert steps_to_detect is not None, f"Episode {episode}: failed to detect drift"
        detection_times.append(steps_to_detect)
    
    # Key assertion: second and third detections should not take dramatically longer
    # than the first (old bug: incomplete reset made detector progressively desensitized)
    assert detection_times[2] <= detection_times[0] * 3, (
        f"Detector desensitization detected! Times: {detection_times}. "
        f"Third detection took {detection_times[2]} steps vs first {detection_times[0]}."
    )


def test_ewma_zero_variance_no_false_positive():
    """When all inputs are identical, EWMA should NOT trigger false positives."""
    ewma = EWMADetector(lambda_=0.2, threshold_multiplier=3.0)
    alerts = 0
    
    # Send identical values (zero variance)
    for _ in range(500):
        if ewma.add_element(0.1): 
            alerts += 1
    
    assert alerts == 0, f"Expected 0 false alarms on constant input, got {alerts}"


def test_ewma_nan_inf_handling():
    """NaN and Inf values should be skipped without crashing."""
    ewma = EWMADetector(lambda_=0.2, threshold_multiplier=3.0)
    
    for _ in range(50):
        ewma.add_element(0.1)
    
    # These should not crash
    result_nan = ewma.add_element(float('nan'))
    result_inf = ewma.add_element(float('inf'))
    result_ninf = ewma.add_element(float('-inf'))
    
    assert result_nan == False
    assert result_inf == False
    assert result_ninf == False
    
    # Should still work after NaN/Inf
    ewma.add_element(0.1)


def test_ewma_parameter_validation():
    """Invalid parameters should raise ValueError."""
    with pytest.raises(ValueError):
        EWMADetector(lambda_=0.0)
    with pytest.raises(ValueError):
        EWMADetector(lambda_=1.0)
    with pytest.raises(ValueError):
        EWMADetector(lambda_=0.5, threshold_multiplier=-1)
    with pytest.raises(ValueError):
        EWMADetector(lambda_=0.5, min_instances=1)


def test_ewma_state_serialization():
    """State should survive serialization/deserialization."""
    ewma = EWMADetector(lambda_=0.2, threshold_multiplier=3.0)
    
    for i in range(50):
        ewma.add_element(0.1 + i * 0.001)
    
    state = ewma.get_state()
    
    ewma2 = EWMADetector(lambda_=0.2, threshold_multiplier=3.0)
    ewma2.load_state(state)
    
    assert ewma2.n == ewma.n
    assert abs(ewma2.mean - ewma.mean) < 1e-10
    assert abs(ewma2.z - ewma.z) < 1e-10

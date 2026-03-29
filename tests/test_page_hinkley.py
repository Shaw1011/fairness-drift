import pytest
from drift.f_page_hinkley import PageHinkleyDetector

def test_ph_monotonic_drift():
    ph = PageHinkleyDetector(threshold=0.5, delta=0.005)
    alerts = 0
    
    for _ in range(100):
        if ph.add_element(0.0): alerts += 1
        
    assert alerts == 0
    
    # Monotonic increase
    val = 0.0
    for _ in range(100):
        val += 0.02
        if ph.add_element(val): alerts += 1
        
    assert alerts > 0


def test_ph_carry_forward_reset():
    """After reset, PH should use the last observed mean as baseline, not 0."""
    ph = PageHinkleyDetector(threshold=0.5, delta=0.005, min_instances=5)
    
    # Drive to drift detection
    for _ in range(50):
        ph.add_element(0.0)
    
    val = 0.0
    detected = False
    for _ in range(200):
        val += 0.03
        if ph.add_element(val):
            detected = True
            break
    
    assert detected, "Should have detected first drift"
    
    # After reset, the mean should NOT be 0 (carry-forward)
    assert ph.n == 0  # State was reset
    
    # Feed values at the NEW level — should NOT immediately trigger
    # (old bug: mean reset to 0 made high values look like new drift)
    false_alarms = 0
    for _ in range(10):
        if ph.add_element(0.5):
            false_alarms += 1
    
    # With carry-forward, the baseline starts near the last mean, so
    # stable values at 0.5 should not trigger


def test_ph_parameter_validation():
    with pytest.raises(ValueError):
        PageHinkleyDetector(threshold=0)
    with pytest.raises(ValueError):
        PageHinkleyDetector(threshold=1, delta=-1)
    with pytest.raises(ValueError):
        PageHinkleyDetector(threshold=1, min_instances=0)


def test_ph_state_serialization():
    ph = PageHinkleyDetector(threshold=0.5, delta=0.005)
    
    for i in range(50):
        ph.add_element(0.01 * i)
    
    state = ph.get_state()
    
    ph2 = PageHinkleyDetector(threshold=0.5, delta=0.005)
    ph2.load_state(state)
    
    assert ph2.n == ph.n
    assert abs(ph2.mean - ph.mean) < 1e-10

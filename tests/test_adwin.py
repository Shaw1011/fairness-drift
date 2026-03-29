import pytest
from drift.f_adwin import ADWINDetector

def test_adwin_detects_sudden_shift():
    adwin = ADWINDetector(delta=0.005, max_window=2000)
    alerts = 0
    
    # Stream 200 normal values (mean 0.1)
    for _ in range(200):
        if adwin.add_element(0.1): alerts += 1
        
    assert alerts == 0
    
    # Sudden shift to 0.5 for 50 values
    for _ in range(50):
        if adwin.add_element(0.5): alerts += 1
        
    assert alerts > 0


def test_adwin_parameter_validation():
    """Invalid parameters should raise ValueError."""
    with pytest.raises(ValueError):
        ADWINDetector(delta=0)
    with pytest.raises(ValueError):
        ADWINDetector(delta=-0.1)
    with pytest.raises(ValueError):
        ADWINDetector(max_window=10)


def test_adwin_nan_handling():
    """NaN values should be skipped without crashing."""
    adwin = ADWINDetector(delta=0.005)
    
    for _ in range(50):
        adwin.add_element(0.1)
    
    result = adwin.add_element(float('nan'))
    assert result == False


def test_adwin_state_serialization():
    """State should survive roundtrip."""
    adwin = ADWINDetector(delta=0.005)
    
    for _ in range(100):
        adwin.add_element(0.1)
    
    state = adwin.get_state()
    
    adwin2 = ADWINDetector(delta=0.005)
    adwin2.load_state(state)
    
    assert adwin2.width == adwin.width
    assert abs(adwin2.total - adwin.total) < 1e-10

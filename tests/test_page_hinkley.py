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

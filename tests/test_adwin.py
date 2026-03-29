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

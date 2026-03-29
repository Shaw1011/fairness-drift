from drift.monitor import FairnessDriftMonitor

def test_monitor_orchestration():
    monitor = FairnessDriftMonitor("demographic_parity_difference", "group", batch_size=50)
    
    alerts = []
    # Test fair data for both groups
    for _ in range(400):
        # alternate groups A and B getting predicted 1
        grp = 'A' if _ % 2 == 0 else 'B'
        alert = monitor.add_element(1, 1, grp)
        if alert: alerts.append(alert)
        
    assert len(alerts) == 0
    
    # Infuse bias by forcing Group B to get predicted 0
    for _ in range(200):
        grp = 'A' if _ % 2 == 0 else 'B'
        y_pred = 1 if grp == 'A' else 0
        alert = monitor.add_element(1, y_pred, grp)
        if alert: alerts.append(alert)
        
    assert len(alerts) > 0
    assert alerts[0]['drift_detected'] == True
    assert 'confidence' in alerts[0]
    assert 'p_value' in alerts[0]

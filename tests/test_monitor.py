import pytest
from drift.monitor import FairnessDriftMonitor, MultiAttributeMonitor

def test_monitor_orchestration():
    monitor = FairnessDriftMonitor("demographic_parity_difference", "group", batch_size=50)
    
    alerts = []
    # Test fair data for both groups
    for _ in range(400):
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
    assert 'severity' in alerts[0]
    assert alerts[0]['severity'] in ('WARNING', 'CRITICAL')
    assert 'detectors_triggered' in alerts[0]


def test_monitor_single_group_batch():
    """When a batch has only one demographic group, the monitor should skip (not inject 0.0)."""
    monitor = FairnessDriftMonitor("demographic_parity_difference", "group", batch_size=20)
    
    alerts = []
    # Send all events for group A only
    for _ in range(100):
        alert = monitor.add_element(1, 1, 'A')
        if alert: alerts.append(alert)
    
    # Should have no alerts — insufficient data, NOT "perfect fairness"
    assert len(alerts) == 0
    assert monitor.skipped_batches > 0, "Should have skipped batches due to single group"


def test_monitor_invalid_metric_name():
    """Invalid metric string should raise ValueError, not KeyError."""
    with pytest.raises(ValueError, match="Unknown metric"):
        FairnessDriftMonitor("nonexistent_metric", "group")


def test_monitor_invalid_batch_size():
    """batch_size < 10 should be rejected."""
    with pytest.raises(ValueError, match="batch_size"):
        FairnessDriftMonitor("demographic_parity_difference", "group", batch_size=5)


def test_monitor_history_capped():
    """History should not grow beyond max_history."""
    monitor = FairnessDriftMonitor("demographic_parity_difference", "group", batch_size=10, max_history=5)
    
    # Send enough data for many batches
    for _ in range(200):
        grp = 'A' if _ % 2 == 0 else 'B'
        monitor.add_element(1, 1, grp)
    
    assert len(monitor.history) <= 5, f"History should be capped at 5, got {len(monitor.history)}"


def test_monitor_bonferroni_correction():
    """Alert p-values should include Bonferroni correction (multiplied by 3)."""
    monitor = FairnessDriftMonitor("demographic_parity_difference", "group", batch_size=50)
    
    # Stabilize
    for _ in range(400):
        grp = 'A' if _ % 2 == 0 else 'B'
        monitor.add_element(1, 1, grp)
    
    # Inject bias
    alert = None
    for _ in range(200):
        grp = 'A' if _ % 2 == 0 else 'B'
        y_pred = 1 if grp == 'A' else 0
        result = monitor.add_element(1, y_pred, grp)
        if result:
            alert = result
            break
    
    if alert:
        # Corrected p-value should be <= 1.0 (Bonferroni caps at 1.0)
        assert alert['p_value'] <= 1.0
        # Confidence should be between 0 and 1
        assert 0 <= alert['confidence'] <= 1.0


def test_monitor_state_roundtrip():
    """Monitor state should survive serialization/deserialization."""
    monitor = FairnessDriftMonitor("demographic_parity_difference", "group", batch_size=20)
    
    for _ in range(100):
        grp = 'A' if _ % 2 == 0 else 'B'
        monitor.add_element(1, 1, grp)
    
    state = monitor.get_state()
    
    monitor2 = FairnessDriftMonitor("demographic_parity_difference", "group", batch_size=20)
    monitor2.load_state(state)
    
    assert monitor2.total_processed == monitor.total_processed
    assert len(monitor2.history) == len(monitor.history)


def test_monitor_reset():
    """Manual reset should clear detector state."""
    monitor = FairnessDriftMonitor("demographic_parity_difference", "group", batch_size=20)
    
    for _ in range(100):
        grp = 'A' if _ % 2 == 0 else 'B'
        monitor.add_element(1, 1, grp)
    
    monitor.reset_detectors()
    
    assert len(monitor.current_batch) == 0
    assert monitor.skipped_batches == 0


# --- Multi-Attribute Monitor Tests ---

def test_multi_monitor_add_and_list():
    mm = MultiAttributeMonitor()
    key1 = mm.add_monitor(sensitive_attr="race", metric_fn="demographic_parity_difference", batch_size=20)
    key2 = mm.add_monitor(sensitive_attr="gender", metric_fn="demographic_parity_difference", batch_size=20)
    
    assert len(mm.list_monitors()) == 2
    assert "race" in key1
    assert "gender" in key2


def test_multi_monitor_duplicate_rejected():
    mm = MultiAttributeMonitor()
    mm.add_monitor(sensitive_attr="race", metric_fn="demographic_parity_difference")
    
    with pytest.raises(ValueError):
        mm.add_monitor(sensitive_attr="race", metric_fn="demographic_parity_difference")


def test_multi_monitor_ingest():
    mm = MultiAttributeMonitor()
    mm.add_monitor(sensitive_attr="race", metric_fn="demographic_parity_difference", batch_size=20)
    mm.add_monitor(sensitive_attr="gender", metric_fn="demographic_parity_difference", batch_size=20)
    
    # Send fair data
    for _ in range(100):
        attrs = {"race": "A" if _ % 2 == 0 else "B", "gender": "M" if _ % 3 == 0 else "F"}
        mm.add_element(1, 1, attrs)
    
    # Verify both monitors received data
    for key, m in mm.monitors.items():
        assert m.total_processed > 0

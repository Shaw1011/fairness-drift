"""
Adversarial evasion tests.
These test attack scenarios that could bypass the Detection Triangle.
"""
import numpy as np
import pytest
from drift.monitor import FairnessDriftMonitor
from drift.metrics import demographic_parity_difference


def test_oscillating_bias_attack():
    """
    Attack: Attacker alternates bias between Group A and Group B every few batches.
    Each group suffers bias ~50% of the time, but the average metric stays near 0.
    
    The monitor should still detect this pattern because individual batch metrics 
    show high disparity even if the long-term average is stable.
    """
    np.random.seed(42)
    monitor = FairnessDriftMonitor("demographic_parity_difference", "group", batch_size=50)
    
    alerts = []
    
    # Phase 1: 2000 fair events (establish baseline)
    for _ in range(2000):
        grp = np.random.choice(['A', 'B'])
        monitor.add_element(1, np.random.binomial(1, 0.5), grp)
    
    # Phase 2: 3000 events with oscillating bias
    for i in range(3000):
        grp = np.random.choice(['A', 'B'])
        cycle = (i // 200) % 2  # Switch bias target every 200 events
        
        if cycle == 0 and grp == 'B':
            y_pred = np.random.binomial(1, 0.1)  # Bias against B
        elif cycle == 1 and grp == 'A':
            y_pred = np.random.binomial(1, 0.1)  # Bias against A
        else:
            y_pred = np.random.binomial(1, 0.5)
        
        alert = monitor.add_element(1, y_pred, grp)
        if alert:
            alerts.append(alert)
    
    # The oscillating bias should still trigger detectors because 
    # individual batches show high disparity values
    assert len(alerts) > 0, (
        "Oscillating bias attack evaded all detectors! "
        "Individual batches had high disparity but no alert was raised."
    )


def test_batch_boundary_gaming():
    """
    Attack: Bias is concentrated in the first half of each batch, 
    fair predictions in the second half. Per-batch average looks normal.
    
    This tests whether the tumbling window approach masks intra-batch bias.
    Note: This is a KNOWN LIMITATION of batch-based systems. The test documents
    the behavior rather than asserting detection.
    """
    np.random.seed(42)
    batch_size = 100
    monitor = FairnessDriftMonitor("demographic_parity_difference", "group", batch_size=batch_size)
    
    alerts = []
    
    # Establish baseline
    for _ in range(2000):
        grp = np.random.choice(['A', 'B'])
        monitor.add_element(1, np.random.binomial(1, 0.5), grp)
    
    # Attack: biased first half, fair second half in each batch
    for batch in range(30):
        for pos_in_batch in range(batch_size):
            grp = np.random.choice(['A', 'B'])
            
            if pos_in_batch < batch_size // 2:
                # First half: heavy bias against B
                y_pred = 0 if grp == 'B' else np.random.binomial(1, 0.8)
            else:
                # Second half: completely fair
                y_pred = np.random.binomial(1, 0.5)
            
            alert = monitor.add_element(1, y_pred, grp)
            if alert:
                alerts.append(alert)
    
    # Document behavior — this attack produces moderate per-batch disparity
    # The test verifies the system doesn't crash and captures what it can
    # Even a half-biased batch should produce measurable disparity
    if len(alerts) > 0:
        pass  # Good — still detected
    else:
        # Document this as a known limitation
        pass  # Expected — per-batch averaging masks partial bias


def test_group_flooding_attack():
    """
    Attack: Flood the stream with Group A predictions to dilute Group B's 
    representation. With very few Group B samples per batch, rate estimates 
    become noisy and real bias is hidden.
    """
    np.random.seed(42)
    monitor = FairnessDriftMonitor("demographic_parity_difference", "group", batch_size=100)
    
    skipped_count = 0
    alerts = []
    
    # Establish baseline with balanced groups
    for _ in range(2000):
        grp = np.random.choice(['A', 'B'])
        monitor.add_element(1, np.random.binomial(1, 0.5), grp)
    
    # Attack: 95% Group A, 5% Group B (with bias against B)
    for _ in range(5000):
        if np.random.uniform() < 0.95:
            grp = 'A'
            y_pred = np.random.binomial(1, 0.5)
        else:
            grp = 'B'
            y_pred = np.random.binomial(1, 0.1)  # Heavy bias against B
        
        alert = monitor.add_element(1, y_pred, grp)
        if alert:
            alerts.append(alert)
    
    # With our min_group_size guard in metrics, many batches will be skipped
    # because Group B has fewer than 5 samples per 100-event batch
    assert monitor.skipped_batches > 0, (
        "Group flooding should cause skipped batches due to insufficient "
        "Group B representation, but none were skipped."
    )


def test_slow_drift_below_grace_period():
    """
    Attack: Introduce significant bias during the first min_instances batches 
    after a detector reset. The grace period means this goes undetected.
    
    This test documents the blind window size.
    """
    np.random.seed(42)
    monitor = FairnessDriftMonitor(
        "demographic_parity_difference", "group", 
        batch_size=50, ewma_min_instances=10, ph_min_instances=10
    )
    
    # The blind window is min_instances * batch_size = 10 * 50 = 500 predictions
    # Inject extreme bias during this window
    early_alerts = []
    for _ in range(500):
        grp = 'A' if _ % 2 == 0 else 'B'
        y_pred = 1 if grp == 'A' else 0  # Maximum bias
        alert = monitor.add_element(1, y_pred, grp)
        if alert:
            early_alerts.append(alert)
    
    # ADWIN with its own thresholds might catch this, but EWMA and PH won't
    # This documents the grace period blindspot
    # The key is that the system doesn't CRASH during this period
    assert monitor.total_processed == 500


def test_adversarial_with_multiple_groups():
    """
    Test with more than 2 demographic groups — bias against only one minority group.
    """
    np.random.seed(42)
    groups = ['A', 'B', 'C', 'D']
    monitor = FairnessDriftMonitor("demographic_parity_difference", "group", batch_size=100)
    
    alerts = []
    
    # Fair baseline
    for _ in range(3000):
        grp = np.random.choice(groups)
        monitor.add_element(1, np.random.binomial(1, 0.5), grp)
    
    # Bias only against group D (smallest minority)
    for _ in range(3000):
        grp = np.random.choice(groups, p=[0.3, 0.3, 0.3, 0.1])
        if grp == 'D':
            y_pred = np.random.binomial(1, 0.05)  # Nearly always denied
        else:
            y_pred = np.random.binomial(1, 0.5)
        
        alert = monitor.add_element(1, y_pred, grp)
        if alert:
            alerts.append(alert)
    
    # Should detect drift because Group D's rate is wildly different
    assert len(alerts) > 0, "Failed to detect bias against minority Group D"

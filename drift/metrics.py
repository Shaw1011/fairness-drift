from typing import Dict, Any, Callable

def demographic_parity_difference(group_stats: Dict[str, Dict[str, float]]) -> float:
    """
    Computes Demographic Parity Difference.
    The maximum difference in favorable outcome rates (prediction=1) between any two groups.
    
    group_stats format:
    {
        "GroupA": {"N": 100, "pos_preds": 50, "true_pos": 30, "actual_pos": 40},
        "GroupB": {"N": 150, "pos_preds": 30, "true_pos": 15, "actual_pos": 30}
    }
    """
    rates = []
    for g, stats in group_stats.items():
        if stats['N'] > 0:
            rates.append(stats['pos_preds'] / stats['N'])
            
    if len(rates) < 2:
        return 0.0
    return max(rates) - min(rates)

def equal_opportunity_difference(group_stats: Dict[str, Dict[str, float]]) -> float:
    """
    Computes Equal Opportunity Difference (Max True Positive Rate difference).
    TPR = true_pos / actual_pos
    """
    rates = []
    for g, stats in group_stats.items():
        if stats['actual_pos'] > 0:
            rates.append(stats['true_pos'] / stats['actual_pos'])
            
    if len(rates) < 2:
        return 0.0
    return max(rates) - min(rates)

def disparate_impact_ratio(group_stats: Dict[str, Dict[str, float]]) -> float:
    """
    Computes Disparate Impact (Ratio of favorable outcome rates).
    Returns the minimum ratio (always <= 1.0). 1.0 implies perfect fairness.
    """
    rates = []
    for g, stats in group_stats.items():
        if stats['N'] > 0:
            rates.append(stats['pos_preds'] / stats['N'])
            
    if len(rates) < 2:
        return 1.0
        
    min_rate = min(rates)
    max_rate = max(rates)
    
    if max_rate == 0:
        return 1.0
        
    return min_rate / max_rate

# Mapping for easy lookup
METRICS_REGISTRY = {
    "demographic_parity_difference": demographic_parity_difference,
    "equal_opportunity_difference": equal_opportunity_difference,
    "disparate_impact_ratio": disparate_impact_ratio
}

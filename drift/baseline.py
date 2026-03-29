from collections import deque
from typing import Dict, Any, Callable

class NaiveSlidingWindow:
    def __init__(self, window_size: int = 1000, threshold: float = 0.10, metric_fn: Callable = None):
        """
        A naive sliding window baseline representing industry 'duct-tape' standards.
        It keeps the last N predictions and triggers drift if the metric exceeds the threshold.
        """
        self.window_size = window_size
        self.threshold = threshold
        self.metric_fn = metric_fn
        self.window = deque(maxlen=window_size)
        self.current_metric_value = 0.0
        
    def add_element(self, y_true: int, y_pred: int, sensitive_attr: Any) -> bool:
        """
        Adds a single record to the window. 
        Returns True if the fairness metric exceeds the static threshold.
        """
        self.window.append({"y_true": y_true, "y_pred": y_pred, "attr": sensitive_attr})
        
        if len(self.window) < self.window_size:
            return False
            
        group_stats = {}
        for item in self.window:
            grp = item['attr']
            if grp not in group_stats:
                group_stats[grp] = {'N': 0, 'pos_preds': 0, 'true_pos': 0, 'actual_pos': 0}
            
            group_stats[grp]['N'] += 1
            if item['y_pred'] == 1:
                group_stats[grp]['pos_preds'] += 1
            if item['y_true'] == 1:
                group_stats[grp]['actual_pos'] += 1
                if item['y_pred'] == 1:
                    group_stats[grp]['true_pos'] += 1
                    
        metric_val = self.metric_fn(group_stats)
        
        # Handle None return from updated metrics (insufficient group data)
        if metric_val is None:
            self.current_metric_value = 0.0
            return False
        
        self.current_metric_value = metric_val
        
        return self.current_metric_value > self.threshold

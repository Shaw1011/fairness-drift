import logging

logger = logging.getLogger(__name__)

class PageHinkleyDetector:
    def __init__(self, threshold: float = 50.0, delta: float = 0.005, min_instances: int = 30):
        """
        Page-Hinkley test for continuous tracking of monotonic increase in fairness disparities.
        
        :param threshold: lambda parameter. Higher = fewer false positives, more latency.
        :param delta: Expected magnitude of changes to ignore.
        :param min_instances: Grace period before triggering alerts.
        :raises ValueError: If parameters are out of valid ranges.
        """
        if threshold <= 0:
            raise ValueError(f"threshold must be > 0, got {threshold}")
        if delta < 0:
            raise ValueError(f"delta must be >= 0, got {delta}")
        if min_instances < 1:
            raise ValueError(f"min_instances must be >= 1, got {min_instances}")
            
        self.threshold = threshold
        self.delta = delta
        self.min_instances = min_instances
        
        self._init_state()
        
    def _init_state(self, carry_forward_value: float = None):
        """
        Initialize or reset internal state.
        
        :param carry_forward_value: If provided, seed the mean with this value so the 
            detector doesn't become blind after reset. Without this, resetting mean to 0
            after detecting drift at value 0.3 means the detector uses 0 as baseline, 
            making it insensitive to the actual post-drift distribution.
        """
        self.n = 0
        self.mean = 0.0
        self.m_sum = 0.0
        self.m_min = 0.0
        self.current_p_value = 1.0
        self._carry_forward = carry_forward_value
        
    def add_element(self, x: float) -> bool:
        """
        Processes a new fairness metric reading.
        Returns True if monotonic drift is detected.
        """
        self.n += 1
        if self.n == 1:
            # Use carry-forward value if available (post-reset continuity)
            self.mean = self._carry_forward if self._carry_forward is not None else x
            self._carry_forward = None
            return False
            
        # Update Page Hinkley sum for increase detection (disparity getting worse)
        self.m_sum += (x - self.mean - self.delta)
        
        # Update empirical mean incrementally
        self.mean = self.mean + (x - self.mean) / self.n
        
        # Track the minimum sum observed
        if self.m_sum < self.m_min:
            self.m_min = self.m_sum
            
        deviation = self.m_sum - self.m_min
        
        # Calculate a pseudo p-value for the API (approaches 0 when threshold is exceeded)
        if self.threshold > 0:
            self.current_p_value = max(0.0001, 1.0 - (deviation / self.threshold))
            
        if self.n > self.min_instances:
            if deviation > self.threshold:
                logger.info(f"Page-Hinkley drift detected: deviation={deviation:.4f} > threshold={self.threshold}")
                # Carry forward the current mean so the next detection cycle
                # has an accurate baseline instead of starting from 0
                self._reset(carry_forward_value=self.mean)
                return True
                
        return False
        
    def _reset(self, carry_forward_value: float = None):
        """Reset with optional carry-forward to maintain detection sensitivity."""
        self._init_state(carry_forward_value=carry_forward_value)

    def get_state(self) -> dict:
        """Serialize detector state for persistence."""
        return {
            "n": self.n, "mean": self.mean, "m_sum": self.m_sum,
            "m_min": self.m_min, "current_p_value": self.current_p_value,
            "_carry_forward": self._carry_forward
        }
    
    def load_state(self, state: dict):
        """Restore detector state from persistence."""
        self.n = state["n"]
        self.mean = state["mean"]
        self.m_sum = state["m_sum"]
        self.m_min = state["m_min"]
        self.current_p_value = state["current_p_value"]
        self._carry_forward = state.get("_carry_forward")

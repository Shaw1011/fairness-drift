class PageHinkleyDetector:
    def __init__(self, threshold: float = 50.0, delta: float = 0.005, min_instances: int = 30):
        """
        Page-Hinkley test for continuous tracking of monotonic increase in fairness disparities.
        
        :param threshold: lambda parameter. Higher = fewer false positives, more latency.
        :param delta: Expected magnitude of changes to ignore.
        :param min_instances: Grace period before triggering alerts.
        """
        self.threshold = threshold
        self.delta = delta
        self.min_instances = min_instances
        
        self.n = 0
        self.mean = 0.0
        self.m_sum = 0.0
        self.m_min = 0.0
        self.current_p_value = 1.0 # Pseudo p-value based on threshold proximity
        
    def add_element(self, x: float) -> bool:
        """
        Processes a new fairness metric reading.
        Returns True if monotonic drift is detected.
        """
        self.n += 1
        if self.n == 1:
            self.mean = x
            return False
            
        # Update Page Hinkley sum for increase detection (disparity getting worse)
        self.m_sum += (x - self.mean - self.delta)
        
        # Update empirical mean incrementally
        self.mean = self.mean + (x - self.mean) / self.n
        
        # Track the minimum sum observed
        if self.m_sum < self.m_min:
            self.m_min = self.m_sum
            
        deviation = self.m_sum - self.m_min
        
        # Calculate a pseudo p-value for the API (0 approaching when threshold goes over)
        if self.threshold > 0:
            self.current_p_value = max(0.0001, 1.0 - (deviation / self.threshold))
            
        if self.n > self.min_instances:
            if deviation > self.threshold:
                # Drift detected! Reset detector state
                self._reset()
                return True
                
        return False
        
    def _reset(self):
        self.n = 0
        self.mean = 0.0
        self.m_sum = 0.0
        self.m_min = 0.0

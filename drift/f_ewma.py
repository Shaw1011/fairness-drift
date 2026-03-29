import math

class EWMADetector:
    def __init__(self, lambda_: float = 0.1, threshold_multiplier: float = 3.0, min_instances: int = 30):
        """
        Exponentially Weighted Moving Average (EWMA) control chart detector.
        Excellent at capturing 'slow-drip' data poisoning and gradual concept drift.
        
        :param lambda_: Weight given to the most recent data point (0 < lambda < 1).
        :param threshold_multiplier: Number of standard deviations for the control limit.
        :param min_instances: Grace period to establish variance.
        """
        self.lambda_ = lambda_
        self.threshold_multiplier = threshold_multiplier
        self.min_instances = min_instances
        
        self.n = 0
        self.z = 0.0 # EWMA statistic
        
        # Welford's algorithm variables for true running mean/variance
        self.mean = 0.0
        self.variance_sum = 0.0
        
        self.current_control_limit = 0.0
        self.current_p_value = 1.0
        
    def add_element(self, x: float) -> bool:
        self.n += 1
        if self.n == 1:
            self.z = x
            self.mean = x
            self.variance_sum = 0.0
            return False
            
        # Welford's online variance update
        old_mean = self.mean
        self.mean = old_mean + (x - old_mean) / self.n
        self.variance_sum += (x - old_mean) * (x - self.mean)
        
        # EWMA update
        self.z = self.lambda_ * x + (1 - self.lambda_) * self.z
        
        if self.n > self.min_instances:
            std_dev = (self.variance_sum / (self.n - 1)) ** 0.5
            
            # Standard error of EWMA considering lambda and time
            # sigma_z = sigma * sqrt((lambda / (2-lambda)) * (1 - (1-lambda)^(2t)))
            ewma_std_dev = std_dev * math.sqrt((self.lambda_ / (2 - self.lambda_)) * (1 - math.pow(1 - self.lambda_, 2 * self.n)))
            
            self.current_control_limit = self.mean + self.threshold_multiplier * ewma_std_dev
            
            # Pseudo p-value based on standard deviations (Empirical Rule)
            z_score = (self.z - self.mean) / (ewma_std_dev + 1e-9)
            if z_score > 0:
                self.current_p_value = math.exp(-0.717 * z_score - 0.416 * z_score**2) # Approx
            
            if self.z > self.current_control_limit:
                self._reset()
                return True
                
        return False
        
    def _reset(self):
        self.z = self.mean


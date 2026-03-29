import math
import logging

logger = logging.getLogger(__name__)

class EWMADetector:
    def __init__(self, lambda_: float = 0.1, threshold_multiplier: float = 3.0, min_instances: int = 30):
        """
        Exponentially Weighted Moving Average (EWMA) control chart detector.
        Excellent at capturing 'slow-drip' data poisoning and gradual concept drift.
        
        :param lambda_: Weight given to the most recent data point (0 < lambda < 1).
        :param threshold_multiplier: Number of standard deviations for the control limit.
        :param min_instances: Grace period to establish variance.
        :raises ValueError: If parameters are out of valid ranges.
        """
        if not (0.0 < lambda_ < 1.0):
            raise ValueError(f"lambda_ must be in (0, 1), got {lambda_}")
        if threshold_multiplier <= 0:
            raise ValueError(f"threshold_multiplier must be > 0, got {threshold_multiplier}")
        if min_instances < 2:
            raise ValueError(f"min_instances must be >= 2, got {min_instances}")
            
        self.lambda_ = lambda_
        self.threshold_multiplier = threshold_multiplier
        self.min_instances = min_instances
        
        self._init_state()
        
    def _init_state(self):
        """Initialize or fully reset all internal state."""
        self.n = 0
        self.z = 0.0  # EWMA statistic
        
        # Welford's algorithm variables for true running mean/variance
        self.mean = 0.0
        self.variance_sum = 0.0
        
        self.current_control_limit = 0.0
        self.current_p_value = 1.0
        
    def add_element(self, x: float) -> bool:
        if not math.isfinite(x):
            logger.warning("Non-finite value received (NaN/Inf), skipping.")
            return False
            
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
            variance = self.variance_sum / (self.n - 1)
            
            # Guard against zero-variance (constant input stream)
            # Use a minimum epsilon to prevent degenerate control limits
            MIN_VARIANCE = 1e-10
            if variance < MIN_VARIANCE:
                self.current_control_limit = self.mean
                self.current_p_value = 1.0
                return False
            
            std_dev = math.sqrt(variance)
            
            # Standard error of EWMA considering lambda and time
            # sigma_z = sigma * sqrt((lambda / (2-lambda)) * (1 - (1-lambda)^(2t)))
            # For large n, (1-lambda)^(2n) underflows to 0 — clamp to avoid NaN
            decay_term = 1.0 - math.pow(1 - self.lambda_, 2 * min(self.n, 10000))
            ewma_std_dev = std_dev * math.sqrt((self.lambda_ / (2 - self.lambda_)) * decay_term)
            
            self.current_control_limit = self.mean + self.threshold_multiplier * ewma_std_dev
            
            # Pseudo p-value based on z-score
            z_score = (self.z - self.mean) / (ewma_std_dev + 1e-9)
            if z_score > 0:
                # Approximation of upper-tail normal CDF
                self.current_p_value = max(1e-6, math.exp(-0.717 * z_score - 0.416 * z_score**2))
            else:
                # z is below mean — no evidence of upward drift
                self.current_p_value = 1.0
            
            if self.z > self.current_control_limit:
                logger.info(f"EWMA drift detected: z={self.z:.4f} > limit={self.current_control_limit:.4f}")
                self._reset()
                return True
                
        return False
        
    def _reset(self):
        """
        Full reset after drift detection.
        Resets ALL state so the detector can catch subsequent drifts with equal sensitivity.
        Previous implementation only reset z, leaving stale mean/variance that desensitized
        the detector after each detection.
        """
        self._init_state()

    def get_state(self) -> dict:
        """Serialize detector state for persistence."""
        return {
            "n": self.n, "z": self.z, "mean": self.mean,
            "variance_sum": self.variance_sum,
            "current_control_limit": self.current_control_limit,
            "current_p_value": self.current_p_value
        }
    
    def load_state(self, state: dict):
        """Restore detector state from persistence."""
        self.n = state["n"]
        self.z = state["z"]
        self.mean = state["mean"]
        self.variance_sum = state["variance_sum"]
        self.current_control_limit = state["current_control_limit"]
        self.current_p_value = state["current_p_value"]

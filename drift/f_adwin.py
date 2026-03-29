import math
import logging
from collections import deque

logger = logging.getLogger(__name__)

class ADWINDetector:
    def __init__(self, delta: float = 0.002, max_window: int = 2000):
        """
        Adaptive Windowing (ADWIN) algorithm.
        Automatically adjusting window size capable of detecting sudden, large-scale shifts.
        
        :param delta: Statistical confidence parameter (p-value threshold) for drift detection.
                      Must be > 0. Smaller = fewer false positives but higher latency.
        :param max_window: Cap to prevent O(N^2) worst-case slowdown. Must be >= 30.
        :raises ValueError: If parameters are out of valid ranges.
        """
        if delta <= 0:
            raise ValueError(f"delta must be > 0, got {delta}")
        if max_window < 30:
            raise ValueError(f"max_window must be >= 30, got {max_window}")
            
        self.delta = delta
        self.max_window = max_window
        self.window = deque(maxlen=max_window)  # O(1) popleft instead of O(N) list.pop(0)
        self.width = 0
        self.total = 0.0
        self.current_onset_index = 0
        self.current_p_value = 1.0
        
    def add_element(self, x: float) -> bool:
        if not math.isfinite(x):
            logger.warning("Non-finite value received (NaN/Inf), skipping.")
            return False
            
        if self.width >= self.max_window:
            self.total -= self.window[0]
            # deque with maxlen auto-evicts, but we need to track total manually
            # Since we set maxlen, appending when full auto-pops left
            self.width -= 1
            
        self.window.append(x)
        self.width += 1
        self.total += x
        
        return self._detect_drift()
        
    def _detect_drift(self) -> bool:
        drift_detected = False
        
        while True:
            n = self.width
            if n < 30:  # Need minimum samples before running statistical significance test
                break
                
            drift_detected_in_iter = False
            total_w0 = 0.0
            
            for i in range(1, n):
                n0 = i
                n1 = n - i
                
                # Minimum sub-window size to avoid noise
                if n0 < 10 or n1 < 10:
                    total_w0 += self.window[i-1]
                    continue
                    
                total_w0 += self.window[i-1]
                mu0 = total_w0 / n0
                mu1 = (self.total - total_w0) / n1
                
                # ADWIN epsilon math (Hoeffding bound)
                m = 1.0 / (1.0 / n0 + 1.0 / n1)
                delta_prime = self.delta / n
                
                # Guard against domain errors: delta_prime must be > 0 for log
                # This is guaranteed since delta > 0 and n > 0, but clamp defensively
                delta_prime = max(delta_prime, 1e-15)
                
                epsilon = math.sqrt((1.0 / (2 * m)) * math.log(4.0 / delta_prime))
                
                # If absolute difference between sub-windows exceeds epsilon, drift!
                if abs(mu0 - mu1) > epsilon:
                    drift_detected = True
                    drift_detected_in_iter = True
                    
                    self.current_p_value = delta_prime  # Store statistical confidence
                    self.current_onset_index = n0  # The point where the window split (drift onset)
                    
                    logger.info(f"ADWIN drift detected: |mu0-mu1|={abs(mu0-mu1):.4f} > eps={epsilon:.4f}")
                    
                    # Discard older subwindow — rebuild deque from remaining elements
                    remaining = list(self.window)[n0:]
                    self.total -= total_w0
                    self.width -= n0
                    self.window = deque(remaining, maxlen=self.max_window)
                    break
                    
            if not drift_detected_in_iter:
                self.current_p_value = 1.0
                break
                
        return drift_detected

    def get_state(self) -> dict:
        """Serialize detector state for persistence."""
        return {
            "window": list(self.window),
            "width": self.width,
            "total": self.total,
            "current_onset_index": self.current_onset_index,
            "current_p_value": self.current_p_value
        }
    
    def load_state(self, state: dict):
        """Restore detector state from persistence."""
        self.window = deque(state["window"], maxlen=self.max_window)
        self.width = state["width"]
        self.total = state["total"]
        self.current_onset_index = state["current_onset_index"]
        self.current_p_value = state["current_p_value"]

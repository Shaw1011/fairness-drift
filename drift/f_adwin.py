import math

class ADWINDetector:
    def __init__(self, delta: float = 0.002, max_window: int = 2000):
        """
        Adaptive Windowing (ADWIN) algorithm.
        Automatically adjusting window size capable of detecting sudden, large-scale shifts.
        
        :param delta: Statistical confidence parameter (p-value threshold) for drift detection.
        :param max_window: Cap to prevent O(N²) worst-case slowdown.
        """
        self.delta = delta
        self.max_window = max_window
        self.window = []
        self.width = 0
        self.total = 0.0
        self.current_onset_index = 0
        self.current_p_value = 1.0
        
    def add_element(self, x: float) -> bool:
        if self.width >= self.max_window:
            self.total -= self.window[0]
            self.window.pop(0)
            self.width -= 1
            
        self.window.append(x)
        self.width += 1
        self.total += x
        
        return self._detect_drift()
        
    def _detect_drift(self) -> bool:
        drift_detected = False
        
        while True:
            n = self.width
            if n < 30: # Need minimum samples before running statistical significance test
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
                
                epsilon = math.sqrt((1.0 / (2 * m)) * math.log(4.0 / delta_prime))
                
                # If absolute difference between sub-windows exceeds epsilon, drift!
                if abs(mu0 - mu1) > epsilon:
                    drift_detected = True
                    drift_detected_in_iter = True
                    
                    self.current_p_value = delta_prime # Store statistical confidence
                    self.current_onset_index = n0 # The point where the window split (drift onset)
                    
                    # Discard older subwindow
                    self.total -= total_w0
                    self.width -= n0
                    self.window = self.window[n0:]
                    break
                    
            if not drift_detected_in_iter:
                # Calculate proxy p-value (how close it was to threshold)
                # Keep current_p_value as 1.0 when perfectly safe
                self.current_p_value = 1.0
                break
                
        return drift_detected

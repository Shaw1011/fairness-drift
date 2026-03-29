import time
import logging
from collections import defaultdict
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple token-bucket rate limiter per client IP.
    No external dependencies — uses in-memory tracking.
    
    :param max_requests: Maximum requests allowed in the time window.
    :param window_seconds: Time window in seconds.
    :param max_payload_bytes: Maximum request body size in bytes.
    """
    
    def __init__(self, app, max_requests: int = 1000, window_seconds: int = 60, max_payload_bytes: int = 10_240):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_payload_bytes = max_payload_bytes
        self._clients: dict = defaultdict(lambda: {"count": 0, "window_start": 0.0})
    
    async def dispatch(self, request: Request, call_next):
        # Check payload size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_payload_bytes:
            logger.warning(f"Payload too large from {request.client.host}: {content_length} bytes")
            return JSONResponse(
                status_code=413,
                content={"detail": f"Payload too large. Maximum: {self.max_payload_bytes} bytes"}
            )
        
        # Rate limiting by client IP
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        client = self._clients[client_ip]
        
        # Reset window if expired
        if now - client["window_start"] > self.window_seconds:
            client["count"] = 0
            client["window_start"] = now
        
        client["count"] += 1
        
        if client["count"] > self.max_requests:
            remaining_time = self.window_seconds - (now - client["window_start"])
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after_seconds": round(remaining_time, 1)
                }
            )
        
        # Add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(max(0, self.max_requests - client["count"]))
        response.headers["X-RateLimit-Reset"] = str(int(client["window_start"] + self.window_seconds))
        
        return response

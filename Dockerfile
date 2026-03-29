FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY drift/ drift/
COPY api/ api/

# Create non-root user
RUN useradd --create-home appuser
USER appuser

# State persistence directory
ENV FAIRNESS_DRIFT_STATE_FILE=/app/data/monitor_state.json
RUN mkdir -p /app/data || true

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.routes:app", "--host", "0.0.0.0", "--port", "8000"]

import pytest
from fastapi.testclient import TestClient
from api.routes import app, monitor, multi_monitor

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_state():
    """Reset monitor state before each test (works for both functions and class methods)."""
    monitor.reset_detectors()
    monitor.total_processed = 0
    monitor.history.clear()
    # Clear multi-monitors
    for key in list(multi_monitor.monitors.keys()):
        del multi_monitor.monitors[key]
    yield


class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "monitors_active" in data
        assert "total_processed" in data

    def test_health_counts_monitors(self):
        # Create a multi-monitor
        client.post("/api/v1/multi/monitors", json={
            "sensitive_attr": "race",
            "metric_fn": "demographic_parity_difference"
        })
        
        response = client.get("/health")
        assert response.json()["monitors_active"] == 2  # default + race


class TestIngestEndpoint:
    def test_valid_ingest(self):
        response = client.post("/api/v1/drift/ingest", json={
            "y_true": 1, "y_pred": 0, "sensitive_attr": "A"
        })
        assert response.status_code == 200
        assert response.json()["drift_detected"] == False
    
    def test_invalid_y_true(self):
        response = client.post("/api/v1/drift/ingest", json={
            "y_true": 5, "y_pred": 0, "sensitive_attr": "A"
        })
        assert response.status_code == 422  # Pydantic validation error
    
    def test_invalid_y_pred(self):
        response = client.post("/api/v1/drift/ingest", json={
            "y_true": 1, "y_pred": -1, "sensitive_attr": "A"
        })
        assert response.status_code == 422
    
    def test_empty_sensitive_attr(self):
        response = client.post("/api/v1/drift/ingest", json={
            "y_true": 1, "y_pred": 0, "sensitive_attr": ""
        })
        assert response.status_code == 422
    
    def test_oversized_sensitive_attr(self):
        response = client.post("/api/v1/drift/ingest", json={
            "y_true": 1, "y_pred": 0, "sensitive_attr": "A" * 101
        })
        assert response.status_code == 422


class TestHistoryEndpoint:
    def test_empty_history(self):
        response = client.get("/api/v1/drift/history")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["history"] == []
    
    def test_history_pagination(self):
        # Generate some history
        for i in range(500):
            client.post("/api/v1/drift/ingest", json={
                "y_true": 1, "y_pred": 1, "sensitive_attr": "A" if i % 2 == 0 else "B"
            })
        
        response = client.get("/api/v1/drift/history?limit=5&offset=0")
        assert response.status_code == 200
        data = response.json()
        assert len(data["history"]) <= 5
    
    def test_history_invalid_limit(self):
        response = client.get("/api/v1/drift/history?limit=0")
        assert response.status_code == 422


class TestResetEndpoint:
    def test_reset(self):
        # Ingest some data
        for i in range(100):
            client.post("/api/v1/drift/ingest", json={
                "y_true": 1, "y_pred": 1, "sensitive_attr": "A" if i % 2 == 0 else "B"
            })
        
        response = client.post("/api/v1/drift/reset")
        assert response.status_code == 200
        assert response.json()["status"] == "reset"


class TestConfigEndpoint:
    def test_get_config(self):
        response = client.get("/api/v1/drift/config")
        assert response.status_code == 200
        data = response.json()
        assert data["metric"] == "demographic_parity_difference"
        assert data["sensitive_attr"] == "group"
        assert "detectors" in data


class TestMultiMonitorEndpoints:
    def test_create_monitor(self):
        response = client.post("/api/v1/multi/monitors", json={
            "sensitive_attr": "race",
            "metric_fn": "demographic_parity_difference"
        })
        assert response.status_code == 200
        assert "monitor_key" in response.json()
    
    def test_create_duplicate_monitor(self):
        client.post("/api/v1/multi/monitors", json={
            "sensitive_attr": "gender",
            "metric_fn": "demographic_parity_difference"
        })
        response = client.post("/api/v1/multi/monitors", json={
            "sensitive_attr": "gender",
            "metric_fn": "demographic_parity_difference"
        })
        assert response.status_code == 409
    
    def test_list_monitors(self):
        client.post("/api/v1/multi/monitors", json={"sensitive_attr": "age"})
        response = client.get("/api/v1/multi/monitors")
        assert response.status_code == 200
        assert len(response.json()["monitors"]) >= 1
    
    def test_delete_monitor(self):
        create_resp = client.post("/api/v1/multi/monitors", json={"sensitive_attr": "income"})
        key = create_resp.json()["monitor_key"]
        
        delete_resp = client.delete(f"/api/v1/multi/monitors/{key}")
        assert delete_resp.status_code == 200
    
    def test_delete_nonexistent_monitor(self):
        response = client.delete("/api/v1/multi/monitors/fake:metric")
        assert response.status_code == 404
    
    def test_multi_ingest(self):
        client.post("/api/v1/multi/monitors", json={"sensitive_attr": "race", "batch_size": 20})
        
        for i in range(100):
            response = client.post("/api/v1/multi/ingest", json={
                "y_true": 1,
                "y_pred": 1,
                "sensitive_attrs": {"race": "A" if i % 2 == 0 else "B"}
            })
            assert response.status_code == 200


class TestMetricsEndpoint:
    def test_list_metrics(self):
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200
        metrics = response.json()["available_metrics"]
        assert "demographic_parity_difference" in metrics
        assert "equal_opportunity_difference" in metrics
        assert "disparate_impact_ratio" in metrics

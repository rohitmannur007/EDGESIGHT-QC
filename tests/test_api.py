import pytest

pytest.importorskip("multipart")

from fastapi.testclient import TestClient

from edgesight.api import create_app
from edgesight.persistence import EdgeSightStore


class Pipeline:
    def predict(self, payload: bytes, filename: str | None = None):
        return {"decision": "accept", "size": len(payload)}


def client() -> TestClient:
    store = EdgeSightStore(":memory:", allow_sqlite_fallback=True)
    return TestClient(create_app(Pipeline(), store, max_upload_bytes=8))


def test_infer_result_review_and_metrics():
    with client() as api:
        response = api.post(
            "/v1/infer",
            files={"file": ("part.jpg", b"image", "image/jpeg")},
            headers={"X-Request-ID": "request-123"},
        )
        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == "request-123"
        result_id = response.json()["id"]
        assert api.get(f"/v1/results/{result_id}").json()["request_id"] == "request-123"
        review = api.post(
            f"/v1/results/{result_id}/reviews",
            json={"reviewer": "qa", "verdict": "accepted"},
        )
        assert review.status_code == 201
        assert "edgesight_results_total 1" in api.get("/metrics").text


def test_upload_limit_and_readiness():
    with client() as api:
        assert api.get("/health/ready").status_code == 200
        response = api.post("/v1/infer", files={"file": ("large.jpg", b"123456789")})
        assert response.status_code == 413


def test_optional_api_key_protects_non_liveness_routes():
    store = EdgeSightStore(":memory:", allow_sqlite_fallback=True)
    app = create_app(Pipeline(), store, require_api_key=True, api_key="secret")
    with TestClient(app) as api:
        assert api.get("/health").status_code == 200
        assert api.get("/ready").status_code == 401
        assert api.get("/ready", headers={"X-API-Key": "secret"}).status_code == 200

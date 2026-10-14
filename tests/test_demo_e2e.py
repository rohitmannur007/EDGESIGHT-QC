import pytest

pytest.importorskip("multipart")

from fastapi.testclient import TestClient

from edgesight.api import create_app
from edgesight.persistence import EdgeSightStore


class WeightlessDemoPipeline:
    def process(self, payload: bytes):
        return {"mode": "demo", "decision": "review", "score": len(payload) / 100}


def test_weightless_batch_to_review_workflow():
    store = EdgeSightStore(":memory:", allow_sqlite_fallback=True)
    app = create_app(WeightlessDemoPipeline(), store)
    with TestClient(app) as client:
        response = client.post(
            "/v1/batches",
            files=[
                ("files", ("one.jpg", b"one", "image/jpeg")),
                ("files", ("two.jpg", b"two", "image/jpeg")),
            ],
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "complete"
        assert len(body["results"]) == 2

        first_id = body["results"][0]["id"]
        assert (
            client.post(
                f"/v1/results/{first_id}/reviews",
                json={"reviewer": "line-lead", "verdict": "needs_attention"},
            ).status_code
            == 201
        )
        assert store.operational_metrics()["reviews_pending"] == 1

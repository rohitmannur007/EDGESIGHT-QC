"""FastAPI application factory."""

from __future__ import annotations

import hmac
import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from edgesight.persistence import EdgeSightStore

from .adapters import run_pipeline


class ReviewRequest(BaseModel):
    reviewer: str = Field(min_length=1, max_length=200)
    verdict: str = Field(
        pattern=(
            "^(accepted|rejected|needs_attention|true_defect|false_alarm|known_defect|"
            "new_defect_type|needs_further_investigation)$"
        )
    )
    comment: str | None = Field(default=None, max_length=4000)


class BatchOptions(BaseModel):
    metadata: dict[str, Any] = Field(default_factory=dict)


async def _read_limited(upload: UploadFile, maximum: int) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while chunk := await upload.read(min(1024 * 1024, maximum + 1 - size)):
        size += len(chunk)
        if size > maximum:
            raise HTTPException(status_code=413, detail=f"upload exceeds {maximum} bytes")
        chunks.append(chunk)
    if not chunks:
        raise HTTPException(status_code=422, detail="empty upload")
    return b"".join(chunks)


def create_app(
    pipeline: Any | None = None,
    store: EdgeSightStore | None = None,
    *,
    max_upload_bytes: int | None = None,
    max_batch_size: int | None = None,
    require_api_key: bool = False,
    api_key: str | None = None,
) -> FastAPI:
    """Create an app without loading model weights.

    The caller injects a pipeline. This keeps imports and startup independent
    of model artifacts while matching ``edgesight.inference.pipeline``.
    """
    upload_limit = (
        max_upload_bytes
        if max_upload_bytes is not None
        else int(os.getenv("EDGESIGHT_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
    )
    batch_limit = (
        max_batch_size
        if max_batch_size is not None
        else int(os.getenv("EDGESIGHT_MAX_BATCH_SIZE", "100"))
    )
    if upload_limit < 1 or batch_limit < 1:
        raise ValueError("upload and batch limits must be positive")
    if require_api_key and not api_key:
        raise ValueError("API-key authentication is enabled but no key is configured")
    owns_store = store is None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.store is None:
            app.state.store = EdgeSightStore(os.getenv("EDGESIGHT_DB_PATH", "edgesight.duckdb"))
        yield
        if owns_store and app.state.store is not None:
            app.state.store.close()

    app = FastAPI(title="EdgeSight-QC", version="1.0.0", lifespan=lifespan)
    app.state.pipeline = pipeline
    app.state.store = store
    app.state.started_at = time.monotonic()

    @app.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Any:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        if require_api_key and request.url.path not in {
            "/health",
            "/health/live",
        }:
            supplied = request.headers.get("X-API-Key", "")
            if not hmac.compare_digest(supplied, api_key or ""):
                return JSONResponse(
                    {"detail": "invalid or missing API key", "request_id": request_id},
                    status_code=401,
                    headers={"X-Request-ID": request_id},
                )
        content_length = request.headers.get("content-length")
        try:
            too_large = (
                bool(content_length) and int(content_length) > upload_limit * 100 + 1024 * 1024
            )
        except ValueError:
            return JSONResponse(
                {"detail": "invalid Content-Length", "request_id": request_id}, status_code=400
            )
        if too_large:
            response = JSONResponse(
                {"detail": "request too large", "request_id": request_id}, status_code=413
            )
            response.headers["X-Request-ID"] = request_id
            return response
        started = time.monotonic()
        try:
            response = await call_next(request)
        except Exception:
            if request.app.state.store:
                request.app.state.store.record_metric("http_requests", 1, {"status": 500})
            raise
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{(time.monotonic() - started) * 1000:.2f}"
        return response

    @app.get("/health")
    @app.get("/health/live")
    async def live() -> dict[str, Any]:
        return {"status": "ok", "uptime_seconds": round(time.monotonic() - app.state.started_at, 3)}

    @app.get("/ready")
    @app.get("/health/ready")
    async def ready(request: Request) -> JSONResponse:
        checks: dict[str, bool] = {
            "model": request.app.state.pipeline is not None,
            "preprocess": request.app.state.pipeline is not None,
            "database": False,
        }
        with suppress(Exception):
            checks["database"] = request.app.state.store.query("SELECT 1 AS ok")[0]["ok"] == 1
        candidate = request.app.state.pipeline
        if candidate is not None and callable(getattr(candidate, "readiness_checks", None)):
            checks.update(
                {str(name): bool(value) for name, value in candidate.readiness_checks().items()}
            )
        elif candidate is not None and callable(getattr(candidate, "is_ready", None)):
            checks["model"] = bool(candidate.is_ready())
            checks["preprocess"] = checks["model"]
        status = 200 if all(checks.values()) else 503
        return JSONResponse(
            {"status": "ready" if status == 200 else "not_ready", "checks": checks},
            status_code=status,
        )

    async def infer_one(
        request: Request, upload: UploadFile, batch_id: str | None = None
    ) -> dict[str, Any]:
        if request.app.state.pipeline is None:
            raise HTTPException(status_code=503, detail="inference pipeline is not configured")
        payload = await _read_limited(upload, upload_limit)
        request_id = request.state.request_id
        try:
            prediction = await run_pipeline(
                request.app.state.pipeline, payload, filename=upload.filename
            )
            result_id = request.app.state.store.save_result(
                status="complete",
                prediction=prediction,
                batch_id=batch_id,
                request_id=request_id,
                source_name=upload.filename,
            )
            return {"id": result_id, "status": "complete", "prediction": prediction}
        except HTTPException:
            raise
        except Exception as exc:
            failure = {
                "code": "inference_failed",
                "type": type(exc).__name__,
                "message": str(exc),
                "request_id": request_id,
            }
            result_id = request.app.state.store.save_result(
                status="failed",
                error=json.dumps(failure, sort_keys=True),
                batch_id=batch_id,
                request_id=request_id,
                source_name=upload.filename,
            )
            return {"id": result_id, "status": "failed", "error": failure}

    @app.post("/predict")
    @app.post("/v1/infer")
    async def infer(request: Request, file: Annotated[UploadFile, File()]) -> dict[str, Any]:
        result = await infer_one(request, file)
        if result["status"] == "failed":
            raise HTTPException(status_code=422, detail=result)
        return result

    @app.post("/predict/batch")
    @app.post("/v1/batches")
    async def batch(request: Request, files: Annotated[list[UploadFile], File()]) -> dict[str, Any]:
        if not files:
            raise HTTPException(status_code=422, detail="at least one file is required")
        if len(files) > batch_limit:
            raise HTTPException(status_code=413, detail=f"batch exceeds {batch_limit} files")
        batch_id = request.app.state.store.create_batch(len(files))
        results = [await infer_one(request, item, batch_id) for item in files]
        status = (
            "complete"
            if all(item["status"] == "complete" for item in results)
            else "partial_failure"
        )
        request.app.state.store.set_batch_status(batch_id, status)
        return {"id": batch_id, "status": status, "results": results}

    @app.get("/results/{result_id}")
    @app.get("/v1/results/{result_id}")
    async def result(request: Request, result_id: str) -> dict[str, Any]:
        found = request.app.state.store.get_result(result_id)
        if found is None:
            raise HTTPException(status_code=404, detail="result not found")
        return found

    @app.get("/results")
    @app.get("/v1/results")
    async def results(
        request: Request,
        limit: int = 100,
        offset: int = 0,
        batch_id: str | None = None,
        status: str | None = None,
        decision: str | None = None,
        category: str | None = None,
    ) -> dict[str, Any]:
        bounded_limit = min(max(limit, 1), 1000)
        values = request.app.state.store.list_results(
            limit=1000 if any((status, decision, category)) else bounded_limit,
            offset=offset,
            batch_id=batch_id,
        )
        values = [
            row
            for row in values
            if (status is None or row.get("status") == status)
            and (
                decision is None
                or (
                    isinstance(row.get("prediction"), dict)
                    and row["prediction"].get("decision") == decision
                )
            )
            and (
                category is None
                or (
                    isinstance(row.get("prediction"), dict)
                    and row["prediction"].get("category") == category
                )
            )
        ][:bounded_limit]
        return {"items": values, "limit": bounded_limit, "offset": max(offset, 0)}

    @app.post("/review/{result_id}", status_code=201)
    @app.post("/v1/results/{result_id}/reviews", status_code=201)
    async def review(request: Request, result_id: str, body: ReviewRequest) -> dict[str, Any]:
        try:
            review_id = request.app.state.store.add_review(
                result_id, body.reviewer, body.verdict, body.comment
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="result not found") from exc
        return {"id": review_id, "result_id": result_id, **body.model_dump()}

    @app.get("/v1/results/{result_id}/reviews")
    async def reviews(request: Request, result_id: str) -> dict[str, Any]:
        if request.app.state.store.get_result(result_id) is None:
            raise HTTPException(status_code=404, detail="result not found")
        return {"items": request.app.state.store.list_reviews(result_id)}

    @app.get("/review-queue")
    async def review_queue(request: Request, limit: int = 100) -> dict[str, Any]:
        rows = request.app.state.store.list_results(limit=limit)
        items = [
            row
            for row in rows
            if isinstance(row.get("prediction"), dict)
            and row["prediction"].get("review_required")
            and not request.app.state.store.list_reviews(row["id"])
        ]
        return {"items": items, "count": len(items)}

    @app.get("/model/info")
    async def model_info(request: Request) -> dict[str, Any]:
        candidate = request.app.state.pipeline
        if candidate is None:
            return {"ready": False, "mode": "unconfigured"}
        info = candidate.model_info() if callable(getattr(candidate, "model_info", None)) else {}
        ready = (
            bool(candidate.is_ready()) if callable(getattr(candidate, "is_ready", None)) else True
        )
        return {"ready": ready, **info}

    @app.get("/metrics")
    async def metrics(request: Request) -> PlainTextResponse:
        values = request.app.state.store.operational_metrics()
        text = (
            "\n".join(
                f"# TYPE edgesight_{name} gauge\nedgesight_{name} {value}"
                for name, value in values.items()
            )
            + "\n"
        )
        return PlainTextResponse(text, media_type="text/plain; version=0.0.4")

    return app

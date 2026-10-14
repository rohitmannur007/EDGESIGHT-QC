SHELL := /bin/bash
.DEFAULT_GOAL := help

UV ?= uv
PYTHON := PYTHONPATH=src $(UV) run python
CONFIG ?= configs/default.yaml
APP_CONFIG ?= configs/app.yaml
TRAIN_CONFIG ?= configs/training.yaml
MODEL_FACTORY ?=
ARTIFACT ?=
PREDICTIONS ?= results/evaluation/predictions.jsonl
IMAGE ?= edgesight-qc:local
PORT ?= 8080

.PHONY: help setup install install-all lock check-env lint format typecheck test check demo demo-data data-validate data-prepare train train-det fit-anom eval export quantize bench optimize-cost report api stream stream-up stream-down stream-produce stream-worker dashboard drift-demo docker-build docker-run compose-up compose-down package clean

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*## "; printf "Usage: make <target>\n\n"} /^[a-zA-Z_-]+:.*## / {printf "  %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Create the Python 3.11 environment
	$(UV) sync --frozen --no-editable

setup: install ## Alias for reproducible local setup

install-all: ## Install all optional capabilities
	$(UV) sync --frozen --all-extras

lock: ## Refresh uv.lock after dependency changes
	$(UV) lock

check-env: ## Validate local tools, paths, and optional services
	$(PYTHON) scripts/check_env.py --config $(CONFIG)

lint: ## Check source, tests, and operational scripts
	$(UV) run ruff check src tests scripts

format: ## Format operational scripts
	$(UV) run ruff format scripts
	$(UV) run ruff check --fix scripts

typecheck: ## Run strict static typing
	$(UV) run mypy scripts

test: ## Run unit and integration tests
	$(UV) run pytest

check: lint typecheck test ## Run all quality checks

demo: ## Run the explicitly synthetic demonstration
	$(PYTHON) scripts/run_demo.py --config configs/demo.yaml

demo-data: demo ## Generate deterministic synthetic data and persisted predictions

data-validate: ## Validate MVTec input (set DATA_ROOT)
	$(PYTHON) scripts/download_data.py --validate-only --source "$(DATA_ROOT)"

data-prepare: data-validate ## Validate before deterministic preparation
	$(PYTHON) scripts/download_data.py --prepare --source "$(DATA_ROOT)" --output data/splits/splits.json --processed-root data/processed

train: ## Train every configured model family
	$(PYTHON) scripts/train_all.py --config $(TRAIN_CONFIG) --detector-config configs/detector.yaml --anomaly-config configs/anomaly.yaml

train-det: ## Train detector from user-supplied MVTec-derived YOLO data
	$(PYTHON) scripts/train_all.py --detector-config configs/detector.yaml --family detector

fit-anom: ## Fit PatchCore after normal tensors have been prepared
	$(PYTHON) scripts/train_all.py --anomaly-config configs/anomaly.yaml --family anomaly

eval: ## Run real models on the untouched manifest and calculate open-set metrics
	$(PYTHON) scripts/evaluate.py --manifest data/processed/manifest.json --config $(APP_CONFIG) --output results/evaluation

export: ## Export and verify configured PyTorch model factory (MODEL_FACTORY=module:function)
	@test -n "$(MODEL_FACTORY)" || { echo "MODEL_FACTORY=module:function is required"; exit 2; }
	$(PYTHON) scripts/export_models.py --factory "$(MODEL_FACTORY)" --output-dir results/exports

quantize: ## Execute supported quantization and record unsupported paths honestly
	@test -n "$(MODEL_FACTORY)" || { echo "MODEL_FACTORY=module:function is required"; exit 2; }
	$(PYTHON) scripts/export_models.py --factory "$(MODEL_FACTORY)" --quantize --output-dir results/exports

bench: ## Benchmark a real exported ARTIFACT
	@test -n "$(ARTIFACT)" || { echo "ARTIFACT=path/to/model.onnx-or-pt is required"; exit 2; }
	$(PYTHON) scripts/benchmark_models.py "$(ARTIFACT)" --output results/benchmarks/benchmark.json --database results/edgesight.duckdb

optimize-cost: ## Select validation thresholds from labeled prediction JSON
	$(PYTHON) scripts/optimize_cost.py "$(PREDICTIONS)" --output results/cost_optimization.json --database results/edgesight.duckdb

report: ## Build evaluation artifacts without inventing results
	$(PYTHON) scripts/generate_report.py --results-dir results --output results/report.md

api: ## Start the HTTP inference service
	PYTHONPATH=src $(UV) run uvicorn edgesight.api.main:app --host 0.0.0.0 --port $(PORT)

stream: ## Start the stream processor
	$(PYTHON) -m edgesight.cli stream --config configs/stream.yaml

stream-up: compose-up ## Start the Kafka-compatible broker

stream-down: compose-down ## Stop the Kafka-compatible broker

stream-produce: ## Run the camera simulator
	$(PYTHON) scripts/camera_simulator.py "$(FRAME_DIR)" --brokers localhost:19092

stream-worker: stream ## Run the inference worker

dashboard: ## Start the operational Streamlit dashboard
	PYTHONPATH=src $(UV) run --extra ui streamlit run src/edgesight/dashboard/app.py

drift-demo: ## Run a controlled persisted-window drift experiment
	$(PYTHON) scripts/drift_demo.py --database results/drift.duckdb --output results/drift_experiment.json

docker-build: ## Build the CPU production image
	docker build --build-arg UV_LOCKED=true -t $(IMAGE) .

docker-run: ## Run the API container locally
	docker run --rm -p $(PORT):8080 --env-file .env $(IMAGE)

compose-up: ## Start Redpanda and its console
	docker compose up -d --wait

compose-down: ## Stop local streaming services
	docker compose down

package: ## Create and validate a source archive
	$(PYTHON) scripts/package_project.py

clean: ## Remove generated caches and reports
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build

# Architecture Decision Records

## ADR-001: Python 3.11 with uv

**Decision:** use a narrow Python 3.11 range, PEP 621 metadata, and a committed uv lockfile.  
**Why:** reproducible, fast local/CI environments while avoiding accidental interpreter drift.  
**Consequence:** dependency updates are explicit; releases fail if the lock is stale.

## ADR-002: CPU ONNX serving boundary

**Decision:** train in framework-specific code, export promoted models to ONNX, and serve through ONNX Runtime CPU.  
**Why:** smaller operational surface and portable edge/cloud execution.  
**Consequence:** every model requires export parity tests; unsupported operators block promotion.

## ADR-003: One immutable bundle

**Decision:** couple model bytes with preprocessing, classes, thresholds, provenance, and checksums.  
**Why:** prevents silent mismatch between training and serving.  
**Consequence:** threshold changes create a new bundle version.

## ADR-004: Shared inference core

**Decision:** HTTP and stream adapters call the same inference service.  
**Why:** consistent validation and decisions across transports.  
**Consequence:** adapters own transport retries/auth, not model logic.

## ADR-005: At-least-once stream processing

**Decision:** commit input only after publishing output and require event-ID idempotency.  
**Why:** clear failure behavior without claiming unimplemented exactly-once guarantees.  
**Consequence:** downstream consumers must deduplicate.

## ADR-006: Scripts target stable interfaces

**Decision:** operational wrappers use the existing `edgesight.api.app:create_app` factory and expect `edgesight.cli` training/stream verbs.  
**Why:** shell/CI/deployment remain thin and testable.  
**Consequence:** implementation should provide adapters if internal modules differ.

## ADR-007: No evidence fabrication

**Decision:** demo output is labeled synthetic; reporting only transcribes real artifacts; screenshots require a live URL.  
**Why:** portfolio and operational documentation must distinguish design claims from measured evidence.  
**Consequence:** empty runs produce explicit “no artifacts,” not sample metrics.

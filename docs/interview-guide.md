# Interview Guide

## Two-minute walkthrough

Start with the operational problem: defect decisions need bounded latency near image sources, while training and governance can remain offline. Explain the immutable model-bundle boundary, then show how API and streaming paths reuse inference. End with evidence discipline: synthetic demo output is plumbing-only and model promotion requires held-out evaluation.

## Likely questions

**Why ONNX Runtime?** It gives a framework-neutral CPU deployment boundary. The trade-off is export/operator compatibility, addressed with parity tests and promotion gates.

**Why one worker?** Multiple web workers duplicate model memory. Horizontal container scaling is easier to observe and limit; measured profiling can justify a change.

**Is Kafka processing exactly once?** No. The documented baseline is at-least-once: publish before commit and key by event ID. Exactly-once requires tested transactions across the relevant boundary.

**How do you prevent leakage?** Content hashing, near-duplicate checks, grouped source/lot splits, immutable manifests, validation-only threshold tuning, and one final held-out evaluation.

**What matters beyond accuracy?** False-pass cost, false-reject/review capacity, calibration, slice behavior, uncertainty, latency distribution, throughput, memory, drift, and operator workflow.

**How does rollback work?** Image digest and model checksum are separately versioned; traffic can return to the previous known-good pair without retraining.

**What are the license risks?** MVTec must be independently obtained under publisher terms. Ultralytics can trigger AGPL or commercial licensing considerations. Every pretrained weight needs provenance and terms review.

## Honest demo path

Run `make demo`, emphasize the synthetic watermark, inspect configuration and generated artifacts, then run tests. If no trained model exists, do not simulate production predictions or quote metrics. Discuss the experiment protocol and show where validated evidence would be recorded.

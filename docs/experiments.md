# Experiment Protocol

## Reproducible run

1. Freeze the dataset manifest and record its SHA-256.
2. Define the hypothesis, primary metric, slices, and acceptance threshold before training.
3. Fix seeds and grouped splits; retain an untouched test split.
4. Run `uv sync --frozen --extra ml`.
5. Set license acknowledgment only after review, then run `make train CONFIG=configs/training.yaml`.
6. Record config, commit, environment, hardware, package lock, duration, and artifact checksums.
7. Evaluate once on the held-out set after selecting thresholds on validation data.
8. Run `make report`; inspect source JSON and failure examples.

## Comparison rules

Change one planned factor at a time or declare a factorial design. Use identical splits and preprocessing. Report uncertainty and sample counts, not only point estimates. Prefer false-pass cost and review capacity over a generic accuracy target. Correct for repeated testing when many variants are compared.

## Artifact layout

```text
results/<run-id>/
  run.json
  metrics.json
  slices.json
  predictions.jsonl
  failures/
  environment.txt
  checksums.sha256
```

`generate_report.py` transcribes scalar values found in JSON; it does not compute, infer, or embellish missing metrics.

## Promotion checklist

- License/provenance approved.
- No split leakage detected.
- Acceptance criteria met on held-out and operational slices.
- Calibration and thresholds reviewed by quality owners.
- CPU load test completed on target hardware.
- Model bundle checksum and rollback candidate recorded.
- Model card completed.

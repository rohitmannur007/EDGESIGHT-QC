# Open-set experiment protocol

Compare detector-only, anomaly-only, and combined-router systems on disjoint
good, known-defect, and held-out-defect test groups. Thresholds, score
normalization, cost weights, augmentations, and model choices use validation
data only. Held-out types are `squeeze`, `print`, and `rough` for capsule,
hazelnut, and tile respectively.

Select comparable operating points against validation-good false alarms
(illustrative target 5% when sample size supports it), then freeze thresholds.
Report test counts, known and unseen recall, good false-alarm rate, precision,
F1, class correctness, AUROC where defined, latency, per-category slices, ROC
trade-offs, and 1,000-resample bootstrap intervals. Report small samples and
negative results explicitly; never guarantee router superiority.

Every run records seed 42, configuration, dataset-manifest hash, source
revision, device/runtime, versions, duration, artifacts, and errors. Completed
runs are immutable. Human-review labels retain reviewer provenance and do not
alter original metrics. The detailed protocol is in `docs/experiments.md`.

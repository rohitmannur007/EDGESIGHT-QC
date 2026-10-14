# EdgeSight-QC model card

EdgeSight-QC combines an optional Ultralytics YOLO nano detector for known
defects with a PatchCore-style nearest-neighbour anomaly model fitted only on
normal products. The router emits `PASS`, `KNOWN_DEFECT`, `UNKNOWN_DEFECT`, or
`HUMAN_REVIEW`.

The included deterministic backend is synthetic demo infrastructure, not a
trained model and not evidence of manufacturing accuracy. Real thresholds must
be selected only from validation predictions under the protocol in
`docs/experiments.md`; held-out defect types and test labels must never
influence fitting, architecture selection, augmentation, or calibration.

Intended use is portfolio research and assisted visual inspection. It is not
intended for safety-critical release decisions, autonomous retraining, medical
use, or claims that all novel defects are detectable. Expected failures include
subtle defects, domain shifts, reflections, blur, confusing known/unknown
evidence, sparse anomaly memory banks, and detector classes absent from the
training taxonomy.

Native macOS may use MPS where operations are supported; nearest-neighbour
distance can remain on CPU. Linux containers are CPU-only and do not imply MPS
or Apple Neural Engine execution. Model artifacts are not bundled. Review
Ultralytics AGPL-3.0/enterprise terms and every checkpoint's provenance before
distribution. See `docs/model-card.md` for the detailed card.

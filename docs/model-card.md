# Model Card

## Status

This is a template for a model bundle, not a claim that a trained model exists. Complete every bracketed field during promotion.

- Model/version: `[required]`
- Task/family: `[classification | detection | anomaly]`
- Training run and commit: `[required]`
- Dataset manifest hash and license review: `[required]`
- Model checksum: `[required]`
- Preprocessing and class map: `[required]`
- Decision thresholds and rationale: `[required]`

## Intended use

Assist manufacturing inspection on the validated product, camera, lighting, and defect scope. It should route uncertain cases to review. It is not a general visual inspector and must not independently control safety-critical machinery without a separately assessed interlock.

## Evaluation requirements

Report per-class confusion, precision/recall, false-pass and false-reject rates, calibration, threshold curves, confidence intervals, slice performance, CPU latency distribution, throughput, memory, and robustness under approved perturbations. Include sample counts and hardware. Never replace missing values with demo output.

## Limitations

Expected failure modes include unseen defects, domain shift, reflections, blur, occlusion, camera movement, adversarial/corrupt files, and labels that do not match the deployed quality policy.

## Licensing and provenance

Ultralytics software/models may be governed by AGPL-3.0 or a commercial/enterprise agreement. Deployment, modification, and distribution need legal review. Pretrained weights can have separate licenses and training-data concerns. Record the exact package version, checkpoint URL/hash, terms at acquisition time, and approval. Default training disables pretrained weights.

## Monitoring

Monitor input validity, score distribution, review/fail rate, drift proxies, errors, latency, and post-inspection outcomes. Alert on changes, retain privacy-safe evidence, and define rollback criteria. Retraining requires the same evaluation and promotion gate as the first release.

# Data Card

## Dataset status

No dataset is included. The training configuration uses `data/mvtec` as a local convention, not evidence that MVTec AD has been downloaded, licensed, or used.

MVTec AD is distributed under terms set by MVTec Software GmbH. Users must obtain it from the publisher, preserve attribution/terms, and determine whether research, commercial, redistribution, and derived-artifact use is allowed. Do not commit or package it.

## Intended data

Images of manufactured objects with labels for normal/defective status and, when available, defect category, localization, source, capture device, lot, and timestamp. A manifest should use stable sample IDs and content hashes without exposing personal data.

## Required validation

- Verify files decode safely and cap dimensions/decompressed pixel count.
- Detect exact and near duplicates before splitting.
- Split by source/lot/time as appropriate to prevent leakage.
- Record class and defect distributions per split.
- Review label ambiguity with domain experts.
- Document camera, lighting, line, and product drift.
- Preserve a never-tuned test set.

## Synthetic demo

`configs/demo.yaml` creates synthetic geometric examples. They test plumbing only. They cannot establish real-world accuracy, calibration, latency under production load, or defect coverage.

## Privacy and retention

Production images may reveal employees, serial numbers, facility layouts, or proprietary processes. Minimize collection, crop when possible, encrypt storage, restrict access, define deletion schedules, and avoid logging image bytes.

## Known risks

Training coverage can omit rare defects; labels can encode inspector inconsistency; acquisition changes can cause drift; random image splits can inflate results; and false passes can be safety-critical. Document mitigations and maintain human-review and stop-line procedures.

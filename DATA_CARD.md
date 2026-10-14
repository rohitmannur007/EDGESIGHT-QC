# EdgeSight-QC data card

The real-data path expects a user-acquired MVTec AD tree containing `capsule`,
`hazelnut`, and `tile`. MVTec data is not redistributed; users must obtain it
from its publisher and review the applicable license. The importer validates
image integrity, masks, dimensions, duplicate hashes, categories, and defect
names before writing `data/splits/splits.json`.

This project uses a custom open-set experiment, not the official MVTec
benchmark: `capsule/squeeze`, `hazelnut/print`, and `tile/rough` are isolated as
unseen tests. Known defect types use deterministic approximate 60/20/20
subsets; seed 42 and every path/hash are recorded. Normal training images fit
PatchCore, while separate normal validation images calibrate thresholds.

The bundled demo data consists of deterministic procedural textures and
rectangular defects. It tests software flow only and is visibly marked
synthetic. It does not represent factory variability, demographic populations,
or production economics. See `docs/data-card.md` for detailed limitations and
retention guidance.

import hashlib
import json

from PIL import Image

from edgesight.data.prepare import prepare_from_manifest


def _image(path, *, mask=False, color=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L" if mask else "RGB", (12, 10), color)
    if mask:
        for x in range(2, 7):
            for y in range(3, 8):
                image.putpixel((x, y), 255)
    image.save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prepare_creates_yolo_negatives_category_anomaly_and_hashes(tmp_path):
    normal = tmp_path / "source" / "normal.png"
    defect = tmp_path / "source" / "crack.png"
    mask = tmp_path / "source" / "crack_mask.png"
    heldout = tmp_path / "source" / "squeeze.png"
    normal_hash = _image(normal)
    defect_hash = _image(defect, color=64)
    _image(mask, mask=True)
    heldout_hash = _image(heldout, color=128)
    manifest = {
        "protocol": "test",
        "seed": 7,
        "held_out": {"capsule": "squeeze"},
        "samples": [
            {
                "sample_id": "normal",
                "path": str(normal),
                "mask_path": None,
                "category": "capsule",
                "defect_type": "good",
                "assignment": "train",
                "sha256": normal_hash,
            },
            {
                "sample_id": "crack",
                "path": str(defect),
                "mask_path": str(mask),
                "category": "capsule",
                "defect_type": "crack",
                "assignment": "known_train",
                "sha256": defect_hash,
            },
            {
                "sample_id": "heldout",
                "path": str(heldout),
                "mask_path": None,
                "category": "capsule",
                "defect_type": "squeeze",
                "assignment": "unseen_test",
                "sha256": heldout_hash,
            },
        ],
    }
    manifest_path = tmp_path / "splits.json"
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "processed"
    stats = prepare_from_manifest(manifest_path, output, minimum_component_area=2)
    assert stats["class_ids"] == {"capsule:crack": 0}
    assert stats["counts"]["yolo_train_negatives"] == 1
    assert (output / "anomaly/capsule/normal_train/normal.png").is_file()
    label = (output / "yolo/labels/train/capsule__crack__crack.txt").read_text()
    assert label.startswith("0 ")
    assert "squeeze" not in (output / "yolo/data.yaml").read_text()
    prepared = json.loads((output / "manifest.json").read_text())
    crack_record = next(item for item in prepared["samples"] if item["sample_id"] == "crack")
    assert crack_record["mask_sha256"]

"""
Split a YOLO dataset export and train a detection model.

Expected source structure:
  dataset_raw/
    images/
    labels/
    classes.txt

Usage:
  python train_YOLO.py
"""

from __future__ import annotations

import random
import shutil
from datetime import datetime
from pathlib import Path

import yaml
from ultralytics import YOLO


RAW_DIR = Path("lpn_dataset_raw")
OUTPUT_DIR = Path("lpn_dataset")
MODEL_SIZE = "yolov8n.pt"
EPOCHS = 80
IMG_SIZE = 640
PROJECT_DIR = Path("runs/train")
RUN_NAME_PREFIX = "lpn_zones"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

RANDOM_SEED = 42
OVERWRITE_DATASET = False
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_config() -> None:
    ratio_sum = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {TRAIN_RATIO} + {VAL_RATIO} + {TEST_RATIO} = {ratio_sum}"
        )


def read_classes(raw_dir: Path) -> list[str]:
    classes_path = raw_dir / "classes.txt"
    if not classes_path.exists():
        raise FileNotFoundError(f"classes.txt not found: {classes_path}")

    classes = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not classes:
        raise ValueError(f"classes.txt is empty: {classes_path}")

    print(f"Classes: {classes}")
    return classes


def validate_raw_dataset(raw_dir: Path) -> list[str]:
    images_dir = raw_dir / "images"
    labels_dir = raw_dir / "labels"

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    image_names = sorted(
        path.name for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not image_names:
        raise FileNotFoundError(f"No supported images found in: {images_dir}")

    label_names = {path.stem for path in labels_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt"}
    missing_labels = [name for name in image_names if Path(name).stem not in label_names]
    if missing_labels:
        preview = ", ".join(missing_labels[:10])
        raise ValueError(
            f"Missing label files for {len(missing_labels)} images in {labels_dir}. "
            f"Examples: {preview}"
        )

    orphan_labels = sorted(label_names - {Path(name).stem for name in image_names})
    if orphan_labels:
        preview = ", ".join(orphan_labels[:10])
        print(f"Warning: {len(orphan_labels)} label files have no matching image. Examples: {preview}")

    print(f"Validated raw dataset: {len(image_names)} images, {len(label_names)} label files")
    return image_names


def prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        if not OVERWRITE_DATASET:
            raise FileExistsError(
                f"Output dataset directory already exists: {output_dir}. "
                "Remove it first or set OVERWRITE_DATASET = True."
            )
        shutil.rmtree(output_dir)

    (output_dir / "images").mkdir(parents=True, exist_ok=False)
    (output_dir / "labels").mkdir(parents=True, exist_ok=False)


def split_dataset(raw_dir: Path, output_dir: Path, image_names: list[str]) -> dict[str, list[str]]:
    images_dir = raw_dir / "images"
    labels_dir = raw_dir / "labels"

    rng = random.Random(RANDOM_SEED)
    shuffled = list(image_names)
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    splits = {
        "train": shuffled[:train_count],
        "val": shuffled[train_count:train_count + val_count],
        "test": shuffled[train_count + val_count:],
    }

    for split_name, files in splits.items():
        print(f"  {split_name}: {len(files)} images")

    for split_name, files in splits.items():
        img_out = output_dir / "images" / split_name
        lbl_out = output_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=False)
        lbl_out.mkdir(parents=True, exist_ok=False)

        for filename in files:
            shutil.copy2(images_dir / filename, img_out / filename)
            label_name = f"{Path(filename).stem}.txt"
            shutil.copy2(labels_dir / label_name, lbl_out / label_name)

    verify_split_integrity(splits, output_dir)
    write_split_manifest(splits, output_dir)
    print(f"Dataset generated in: {output_dir}")
    return splits


def verify_split_integrity(splits: dict[str, list[str]], output_dir: Path) -> None:
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    if train_set & val_set:
        raise ValueError("Train and val splits overlap.")
    if train_set & test_set:
        raise ValueError("Train and test splits overlap.")
    if val_set & test_set:
        raise ValueError("Val and test splits overlap.")

    total_split_files = sum(len(files) for files in splits.values())
    total_unique_files = len(train_set | val_set | test_set)
    if total_split_files != total_unique_files:
        raise ValueError("Split contains duplicate filenames.")

    copied_total = 0
    for split_name, expected_files in splits.items():
        copied_files = {
            path.name
            for path in (output_dir / "images" / split_name).iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        }
        if copied_files != set(expected_files):
            raise ValueError(f"Copied files do not match the manifest for split: {split_name}")
        copied_total += len(copied_files)

    if copied_total != total_unique_files:
        raise ValueError("Copied file count does not match total unique split file count.")

    print("Verified split integrity: no overlap between train/val/test")


def write_split_manifest(splits: dict[str, list[str]], output_dir: Path) -> Path:
    manifest_path = output_dir / "split_manifest.yaml"
    manifest = {
        "seed": RANDOM_SEED,
        "ratios": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "counts": {split_name: len(files) for split_name, files in splits.items()},
        "splits": splits,
    }
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Saved split manifest: {manifest_path}")
    return manifest_path


def create_yaml(output_dir: Path, classes: list[str]) -> Path:
    data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(classes),
        "names": classes,
    }

    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Saved data.yaml: {yaml_path}")
    return yaml_path


def make_run_name() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{RUN_NAME_PREFIX}_{timestamp}"


def resolve_training_save_dir(model: YOLO, train_result) -> Path:
    save_dir = getattr(train_result, "save_dir", None)
    if save_dir is None:
        trainer = getattr(model, "trainer", None)
        save_dir = getattr(trainer, "save_dir", None)
    if save_dir is None:
        raise RuntimeError("Could not determine Ultralytics training save directory.")
    return Path(save_dir)


def train(yaml_path: Path) -> tuple[Path, Path, str]:
    run_name = make_run_name()

    print(f"\nLoading base model: {MODEL_SIZE}")
    model = YOLO(MODEL_SIZE)

    print(f"Starting training for {EPOCHS} epochs...")
    train_result = model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        project=str(PROJECT_DIR),
        name=run_name,
        exist_ok=False,
        patience=15,
        batch=-1,
        workers=2,
        verbose=True,
    )

    save_dir = resolve_training_save_dir(model, train_result)
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"best.pt was not created at the expected path: {best_weights}")

    print("\nTraining finished")
    print(f"Training run directory: {save_dir}")
    print(f"Best weights: {best_weights}")
    return best_weights, save_dir, run_name


def evaluate(best_weights: Path, yaml_path: Path, train_run_dir: Path, run_name: str):
    print("\nEvaluating model on the test split...")
    model = YOLO(str(best_weights))

    metrics = model.val(
        data=str(yaml_path),
        split="test",
        project=str(train_run_dir),
        name=f"{run_name}_test",
        exist_ok=False,
    )

    eval_dir = getattr(metrics, "save_dir", None)
    if eval_dir is not None:
        print(f"Evaluation artifacts: {eval_dir}")

    print("\nTest metrics:")
    print(f"  mAP50:     {metrics.box.map50:.3f}")
    print(f"  mAP50-95:  {metrics.box.map:.3f}")
    print(f"  Precision: {metrics.box.mp:.3f}")
    print(f"  Recall:    {metrics.box.mr:.3f}")
    return metrics


def main() -> None:
    print("=" * 55)
    print("  YOLOv8 - dataset preparation and training")
    print("=" * 55)

    validate_config()
    classes = read_classes(RAW_DIR)
    image_names = validate_raw_dataset(RAW_DIR)
    prepare_output_dir(OUTPUT_DIR)

    print(
        f"\nSplitting dataset "
        f"({int(TRAIN_RATIO * 100)}/{int(VAL_RATIO * 100)}/{int(TEST_RATIO * 100)})..."
    )
    split_dataset(RAW_DIR, OUTPUT_DIR, image_names)

    yaml_path = create_yaml(OUTPUT_DIR, classes)
    best_weights, train_run_dir, run_name = train(yaml_path)
    evaluate(best_weights, yaml_path, train_run_dir, run_name)

    print("\nArtifacts:")
    print(f"  Dataset:   {OUTPUT_DIR.resolve()}")
    print(f"  Run dir:   {train_run_dir}")
    print(f"  Best .pt:  {best_weights}")
    print("\nDone")


if __name__ == "__main__":
    main()

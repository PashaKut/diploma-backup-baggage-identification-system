from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "runs" / "train" / "detector_20260506_012226" / "weights" / "best.pt"
DEFAULT_PHOTOS_DIR = PROJECT_ROOT / "data" / "photos"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "detect" / "photo_crops"
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect regions in data/photos with YOLO and save cropped detections."
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Path to YOLO weights (.pt).")
    parser.add_argument("--photos-dir", type=Path, default=DEFAULT_PHOTOS_DIR, help="Directory with source photos.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for cropped images and reports. Defaults to a timestamped folder under runs/detect/photo_crops.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for prediction.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument(
        "--best-only",
        action="store_true",
        help="Save only the highest-confidence detection per photo instead of all detections.",
    )
    return parser.parse_args()


def resolve_output_dir(custom_dir: Path | None) -> Path:
    if custom_dir is not None:
        return custom_dir.resolve()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (DEFAULT_OUTPUT_ROOT / f"detector_crops_{timestamp}").resolve()


def list_photos(photos_dir: Path) -> list[Path]:
    if not photos_dir.exists():
        raise FileNotFoundError(f"Photos directory not found: {photos_dir}")

    photos = sorted(
        path for path in photos_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not photos:
        raise FileNotFoundError(f"No supported images found in: {photos_dir}")
    return photos


def load_model(weights_path: Path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed in the current environment. Install it before running this script."
        ) from exc

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    return YOLO(str(weights_path))


def run_inference(model, photos: list[Path], conf: float, imgsz: int):
    return model.predict(
        source=[str(path) for path in photos],
        conf=conf,
        imgsz=imgsz,
        save=False,
        verbose=False,
    )


def clamp_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[int, int, int, int]:
    left = max(0, min(int(round(x1)), width - 1))
    top = max(0, min(int(round(y1)), height - 1))
    right = max(left + 1, min(int(round(x2)), width))
    bottom = max(top + 1, min(int(round(y2)), height))
    return left, top, right, bottom


def save_crops(results, output_dir: Path, best_only: bool) -> tuple[list[dict[str, str]], int]:
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    total_crops = 0

    for result in results:
        image_path = Path(result.path)
        image_name = image_path.name
        image_stem = image_path.stem
        image = result.orig_img
        height, width = image.shape[:2]

        boxes = result.boxes
        detection_count = len(boxes)
        if detection_count == 0:
            rows.append(
                {
                    "photo_filename": image_name,
                    "crop_filename": "",
                    "detection_index": "",
                    "confidence": "",
                    "class_name": "",
                    "x1": "",
                    "y1": "",
                    "x2": "",
                    "y2": "",
                }
            )
            continue

        detections = list(
            zip(
                boxes.xyxy.tolist(),
                boxes.conf.tolist(),
                boxes.cls.tolist(),
            )
        )
        if best_only:
            detections = [max(detections, key=lambda item: float(item[1]))]

        for index, (xyxy, confidence, class_id) in enumerate(detections, start=1):
            x1, y1, x2, y2 = clamp_box(*xyxy, width=width, height=height)
            crop = image[y1:y2, x1:x2]
            class_name = result.names[int(class_id)]
            crop_filename = f"{image_stem}__det{index:02d}__{class_name}_{float(confidence):.3f}.jpg"
            crop_path = crops_dir / crop_filename

            if crop.size == 0:
                continue

            if not cv2.imwrite(str(crop_path), crop):
                raise RuntimeError(f"Failed to save crop: {crop_path}")

            rows.append(
                {
                    "photo_filename": image_name,
                    "crop_filename": crop_filename,
                    "detection_index": str(index),
                    "confidence": f"{float(confidence):.4f}",
                    "class_name": class_name,
                    "x1": str(x1),
                    "y1": str(y1),
                    "x2": str(x2),
                    "y2": str(y2),
                }
            )
            total_crops += 1

    return rows, total_crops


def write_csv(rows: list[dict[str, str]], output_dir: Path) -> Path:
    csv_path = output_dir / "crops.csv"
    fieldnames = [
        "photo_filename",
        "crop_filename",
        "detection_index",
        "confidence",
        "class_name",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def write_summary(
    rows: list[dict[str, str]],
    output_dir: Path,
    weights_path: Path,
    photos_dir: Path,
    total_photos: int,
    total_crops: int,
) -> Path:
    summary_path = output_dir / "summary.txt"
    photos_with_crops = len({row["photo_filename"] for row in rows if row["crop_filename"]})
    photos_without_crops = total_photos - photos_with_crops

    lines = [
        "YOLO crop extraction",
        "=" * 60,
        f"Weights: {weights_path}",
        f"Photos: {photos_dir}",
        f"Output: {output_dir}",
        "",
        f"Total photos: {total_photos}",
        f"Photos with detections: {photos_with_crops}",
        f"Photos without detections: {photos_without_crops}",
        f"Total crops saved: {total_crops}",
        "",
    ]

    missing = sorted({row["photo_filename"] for row in rows if not row["crop_filename"]})
    if missing:
        lines.append("Photos without detections:")
        lines.extend(f"- {name}" for name in missing)
        lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def print_console_summary(output_dir: Path, csv_path: Path, summary_path: Path, total_photos: int, total_crops: int) -> None:
    print("=" * 60)
    print("YOLO crop extraction on data/photos")
    print("=" * 60)
    print(f"Total photos:       {total_photos}")
    print(f"Total crops saved:  {total_crops}")
    print(f"Crops directory:    {output_dir / 'crops'}")
    print(f"CSV report:         {csv_path}")
    print(f"Summary report:     {summary_path}")
    print("=" * 60)


def main() -> None:
    args = parse_args()
    weights_path = args.weights.resolve()
    photos_dir = args.photos_dir.resolve()
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    photos = list_photos(photos_dir)
    model = load_model(weights_path)
    results = run_inference(model, photos, conf=args.conf, imgsz=args.imgsz)
    rows, total_crops = save_crops(results, output_dir, best_only=args.best_only)
    csv_path = write_csv(rows, output_dir)
    summary_path = write_summary(rows, output_dir, weights_path, photos_dir, len(photos), total_crops)
    print_console_summary(output_dir, csv_path, summary_path, len(photos), total_crops)


if __name__ == "__main__":
    main()

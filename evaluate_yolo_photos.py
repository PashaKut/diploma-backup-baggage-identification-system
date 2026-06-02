from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "runs" / "train" / "lpn_zones_20260509_003420" / "weights" / "best.pt"
DEFAULT_PHOTOS_DIR = PROJECT_ROOT / "check_lpn_photos" / "check2"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "detect" / "photo_eval" / "lpn_zones"
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the trained YOLO detector on every image in data/photos and save a report."
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Path to YOLO weights (.pt).")
    parser.add_argument("--photos-dir", type=Path, default=DEFAULT_PHOTOS_DIR, help="Directory with source photos.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for annotated images and reports. Defaults to a timestamped folder under runs/detect/photo_eval.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for prediction.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    return parser.parse_args()


def resolve_output_dir(custom_dir: Path | None) -> Path:
    if custom_dir is not None:
        return custom_dir.resolve()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (DEFAULT_OUTPUT_ROOT / f"detector_v1_photos_{timestamp}").resolve()


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


def run_inference(model, photos: list[Path], output_dir: Path, conf: float, imgsz: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    return model.predict(
        source=[str(path) for path in photos],
        conf=conf,
        imgsz=imgsz,
        save=True,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
        verbose=False,
    )


def build_rows(results) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for result in results:
        boxes = result.boxes
        detection_count = len(boxes)
        confidences = [float(conf) for conf in boxes.conf.tolist()] if detection_count else []
        classes = [result.names[int(cls_id)] for cls_id in boxes.cls.tolist()] if detection_count else []
        xyxy_values = boxes.xyxy.tolist() if detection_count else []

        rows.append(
            {
                "photo_filename": Path(result.path).name,
                "image_width": str(result.orig_shape[1]),
                "image_height": str(result.orig_shape[0]),
                "detections": str(detection_count),
                "best_confidence": f"{max(confidences):.4f}" if confidences else "",
                "all_confidences": "; ".join(f"{value:.4f}" for value in confidences),
                "classes": "; ".join(classes),
                "boxes_xyxy": "; ".join(
                    f"[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
                    for x1, y1, x2, y2 in xyxy_values
                ),
            }
        )
    return rows


def write_csv(rows: list[dict[str, str]], output_dir: Path) -> Path:
    csv_path = output_dir / "detections.csv"
    fieldnames = [
        "photo_filename",
        "image_width",
        "image_height",
        "detections",
        "best_confidence",
        "all_confidences",
        "classes",
        "boxes_xyxy",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def write_summary(rows: list[dict[str, str]], output_dir: Path, weights_path: Path, photos_dir: Path) -> Path:
    summary_path = output_dir / "summary.txt"
    total = len(rows)
    detected_rows = [row for row in rows if int(row["detections"]) > 0]
    undetected_rows = [row for row in rows if int(row["detections"]) == 0]
    best_conf_values = [float(row["best_confidence"]) for row in detected_rows if row["best_confidence"]]

    avg_best_conf = sum(best_conf_values) / len(best_conf_values) if best_conf_values else 0.0

    lines = [
        "YOLO photo evaluation",
        "=" * 60,
        f"Weights: {weights_path}",
        f"Photos: {photos_dir}",
        f"Annotated output: {output_dir}",
        "",
        f"Total photos: {total}",
        f"Photos with detections: {len(detected_rows)}",
        f"Photos without detections: {len(undetected_rows)}",
        f"Detection rate: {(len(detected_rows) / total * 100):.2f}%" if total else "Detection rate: 0.00%",
        f"Average best confidence: {avg_best_conf:.4f}" if detected_rows else "Average best confidence: n/a",
        "",
    ]

    if undetected_rows:
        lines.append("Photos without detections:")
        lines.extend(f"- {row['photo_filename']}" for row in undetected_rows)
        lines.append("")

    top_rows = sorted(
        detected_rows,
        key=lambda row: float(row["best_confidence"]) if row["best_confidence"] else 0.0,
        reverse=True,
    )[:10]
    if top_rows:
        lines.append("Top detections by confidence:")
        lines.extend(
            f"- {row['photo_filename']}: {row['best_confidence']} ({row['detections']} detections)"
            for row in top_rows
        )
        lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def print_console_summary(rows: list[dict[str, str]], output_dir: Path, csv_path: Path, summary_path: Path) -> None:
    total = len(rows)
    detected = sum(1 for row in rows if int(row["detections"]) > 0)
    undetected = total - detected

    print("=" * 60)
    print("YOLO detector check on data/photos")
    print("=" * 60)
    print(f"Total photos:             {total}")
    print(f"Photos with detections:   {detected}")
    print(f"Photos without detections:{undetected}")
    if total:
        print(f"Detection rate:           {detected / total * 100:.2f}%")
    print(f"Annotated images:         {output_dir}")
    print(f"CSV report:               {csv_path}")
    print(f"Summary report:           {summary_path}")
    print("=" * 60)


def main() -> None:
    args = parse_args()
    weights_path = args.weights.resolve()
    photos_dir = args.photos_dir.resolve()
    output_dir = resolve_output_dir(args.output_dir)

    photos = list_photos(photos_dir)
    model = load_model(weights_path)
    results = run_inference(model, photos, output_dir, conf=args.conf, imgsz=args.imgsz)
    rows = build_rows(results)
    csv_path = write_csv(rows, output_dir)
    summary_path = write_summary(rows, output_dir, weights_path, photos_dir)
    print_console_summary(rows, output_dir, csv_path, summary_path)


if __name__ == "__main__":
    main()

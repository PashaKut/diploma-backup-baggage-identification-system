from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "runs" / "train" / "lpn_zones_20260509_003420" / "weights" / "best.pt"
DEFAULT_PHOTOS_DIR = PROJECT_ROOT / "check_lpn_photos" / "check3"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "detect" / "ocr_checks" / "lpn_zones"
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_CLASSES = ("LPN_right", "LPN_bottom")
LPN_DIGITS_PATTERN = re.compile(r"(\d{4})\D*(\d{6})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect LPN zones, OCR them, and compare readings across both zones."
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Path to YOLO weights (.pt).")
    parser.add_argument("--photos-dir", type=Path, default=DEFAULT_PHOTOS_DIR, help="Directory with source photos.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for crops and OCR reports. Defaults to a timestamped folder under runs/detect/ocr_checks/lpn_zones.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detection.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size for YOLO.")
    parser.add_argument("--ocr-width", type=int, default=640, help="Target width for OCR preprocessing while keeping aspect ratio.")
    return parser.parse_args()


def resolve_output_dir(custom_dir: Path | None) -> Path:
    if custom_dir is not None:
        return custom_dir.resolve()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (DEFAULT_OUTPUT_ROOT / f"ocr_check_{timestamp}").resolve()


def list_photos(photos_dir: Path) -> list[Path]:
    if not photos_dir.exists():
        raise FileNotFoundError(f"Photos directory not found: {photos_dir}")

    photos = sorted(
        path for path in photos_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not photos:
        raise FileNotFoundError(f"No supported images found in: {photos_dir}")
    return photos


def load_detection_model(weights_path: Path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed in the current environment. Install it before running this script."
        ) from exc

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    return YOLO(str(weights_path))


def load_ocr_reader():
    try:
        import easyocr
    except ImportError as exc:
        raise RuntimeError(
            "EasyOCR is not installed in the current environment. Install it before running this script."
        ) from exc

    return easyocr.Reader(["en"], gpu=False)


def run_detection(model, photos: list[Path], conf: float, imgsz: int):
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


def sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value).strip("_") or "class"


def resize_for_ocr(image, target_width: int):
    height, width = image.shape[:2]
    if width >= target_width:
        return image

    scale = target_width / width
    target_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)


def build_ocr_variants(crop, target_width: int) -> list[tuple[str, object]]:
    upscaled = resize_for_ocr(crop, target_width)
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blurred = cv2.GaussianBlur(clahe, (3, 3), 0)

    return [
        ("upscaled_color", upscaled),
        ("gray", gray),
        ("clahe", clahe),
        ("otsu", otsu),
        ("clahe_blur", blurred),
    ]


def normalize_lpn(text: str) -> str:
    match = LPN_DIGITS_PATTERN.search(text)
    if not match:
        return ""
    return f"{match.group(1)} {match.group(2)}"


def score_ocr_candidate(raw_text: str, normalized_text: str, confidence: float) -> tuple[int, float, int]:
    is_valid = 1 if normalized_text else 0
    digit_count = sum(ch.isdigit() for ch in raw_text)
    return (is_valid, confidence, digit_count)


def run_zone_ocr(reader, crop, target_width: int) -> dict[str, str]:
    best = {
        "ocr_raw_text": "",
        "ocr_normalized_text": "",
        "ocr_confidence": "",
        "ocr_variant": "",
    }
    best_score = (-1, -1.0, -1)

    for variant_name, variant_image in build_ocr_variants(crop, target_width):
        readings = reader.readtext(variant_image, detail=1, paragraph=False, allowlist="0123456789 ")
        if not readings:
            continue

        combined_text = " ".join((text or "").strip() for _, text, _ in readings).strip()
        confidence = max(float(confidence) for _, _, confidence in readings)
        normalized_text = normalize_lpn(combined_text)
        score = score_ocr_candidate(combined_text, normalized_text, confidence)

        if score > best_score:
            best_score = score
            best = {
                "ocr_raw_text": combined_text,
                "ocr_normalized_text": normalized_text,
                "ocr_confidence": f"{confidence:.4f}",
                "ocr_variant": variant_name,
            }

    return best


def choose_best_zones(result) -> dict[str, dict]:
    selections: dict[str, dict] = {}
    boxes = result.boxes
    if len(boxes) == 0:
        return selections

    for xyxy, confidence, class_id in zip(
        boxes.xyxy.tolist(),
        boxes.conf.tolist(),
        boxes.cls.tolist(),
    ):
        class_name = result.names[int(class_id)]
        current = selections.get(class_name)
        candidate = {
            "xyxy": xyxy,
            "confidence": float(confidence),
            "class_name": class_name,
        }
        if current is None or candidate["confidence"] > current["confidence"]:
            selections[class_name] = candidate

    return selections


def save_crop(crop, output_dir: Path, image_stem: str, class_name: str, confidence: float) -> str:
    class_dir = output_dir / "crops" / sanitize_name(class_name)
    class_dir.mkdir(parents=True, exist_ok=True)
    crop_filename = f"{image_stem}__{sanitize_name(class_name)}__{confidence:.3f}.jpg"
    crop_path = class_dir / crop_filename
    if not cv2.imwrite(str(crop_path), crop):
        raise RuntimeError(f"Failed to save crop: {crop_path}")
    return str(crop_path.relative_to(output_dir))


def evaluate_photo_decision(zone_rows: dict[str, dict]) -> tuple[str, str]:
    right_text = zone_rows.get("LPN_right", {}).get("ocr_normalized_text", "")
    bottom_text = zone_rows.get("LPN_bottom", {}).get("ocr_normalized_text", "")

    if right_text and bottom_text:
        if right_text == bottom_text:
            return "zones_match", right_text
        return "zones_conflict", ""

    if right_text:
        return "single_zone_right", right_text
    if bottom_text:
        return "single_zone_bottom", bottom_text

    if zone_rows:
        return "zones_detected_no_valid_ocr", ""
    return "no_zones_detected", ""


def process_results(results, reader, output_dir: Path, ocr_width: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    photo_rows: list[dict[str, str]] = []
    zone_rows_output: list[dict[str, str]] = []

    for result in results:
        image_path = Path(result.path)
        image_name = image_path.name
        image_stem = image_path.stem
        image = result.orig_img
        height, width = image.shape[:2]
        selected_zones = choose_best_zones(result)

        zone_rows_for_decision: dict[str, dict] = {}
        for class_name in EXPECTED_CLASSES:
            detection = selected_zones.get(class_name)
            if detection is None:
                zone_rows_output.append(
                    {
                        "photo_filename": image_name,
                        "class_name": class_name,
                        "detected": "0",
                        "detection_confidence": "",
                        "crop_filename": "",
                        "x1": "",
                        "y1": "",
                        "x2": "",
                        "y2": "",
                        "ocr_variant": "",
                        "ocr_raw_text": "",
                        "ocr_normalized_text": "",
                        "ocr_confidence": "",
                    }
                )
                continue

            x1, y1, x2, y2 = clamp_box(*detection["xyxy"], width=width, height=height)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_filename = save_crop(crop, output_dir, image_stem, class_name, detection["confidence"])
            ocr_result = run_zone_ocr(reader, crop, ocr_width)
            zone_row = {
                "photo_filename": image_name,
                "class_name": class_name,
                "detected": "1",
                "detection_confidence": f"{detection['confidence']:.4f}",
                "crop_filename": crop_filename,
                "x1": str(x1),
                "y1": str(y1),
                "x2": str(x2),
                "y2": str(y2),
                "ocr_variant": ocr_result["ocr_variant"],
                "ocr_raw_text": ocr_result["ocr_raw_text"],
                "ocr_normalized_text": ocr_result["ocr_normalized_text"],
                "ocr_confidence": ocr_result["ocr_confidence"],
            }
            zone_rows_output.append(zone_row)
            zone_rows_for_decision[class_name] = zone_row

        decision, final_lpn = evaluate_photo_decision(zone_rows_for_decision)
        right_row = zone_rows_for_decision.get("LPN_right", {})
        bottom_row = zone_rows_for_decision.get("LPN_bottom", {})

        photo_rows.append(
            {
                "photo_filename": image_name,
                "decision": decision,
                "final_lpn": final_lpn,
                "right_detected": "1" if right_row else "0",
                "right_text": right_row.get("ocr_normalized_text", ""),
                "right_raw_text": right_row.get("ocr_raw_text", ""),
                "right_detection_confidence": right_row.get("detection_confidence", ""),
                "right_ocr_confidence": right_row.get("ocr_confidence", ""),
                "bottom_detected": "1" if bottom_row else "0",
                "bottom_text": bottom_row.get("ocr_normalized_text", ""),
                "bottom_raw_text": bottom_row.get("ocr_raw_text", ""),
                "bottom_detection_confidence": bottom_row.get("detection_confidence", ""),
                "bottom_ocr_confidence": bottom_row.get("ocr_confidence", ""),
            }
        )

    return photo_rows, zone_rows_output


def write_csv(rows: list[dict[str, str]], csv_path: Path, fieldnames: list[str]) -> Path:
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def write_summary(photo_rows: list[dict[str, str]], output_dir: Path, weights_path: Path, photos_dir: Path) -> Path:
    summary_path = output_dir / "summary.txt"
    decision_counts: dict[str, int] = {}
    for row in photo_rows:
        decision = row["decision"]
        decision_counts[decision] = decision_counts.get(decision, 0) + 1

    lines = [
        "LPN zone OCR check",
        "=" * 60,
        f"Weights: {weights_path}",
        f"Photos: {photos_dir}",
        f"Output: {output_dir}",
        "",
        f"Total photos: {len(photo_rows)}",
        "Decision counts:",
    ]
    lines.extend(f"- {decision}: {count}" for decision, count in sorted(decision_counts.items()))
    lines.append("")

    conflicts = [row for row in photo_rows if row["decision"] == "zones_conflict"]
    if conflicts:
        lines.append("Conflict photos:")
        lines.extend(
            f"- {row['photo_filename']}: right={row['right_text'] or row['right_raw_text']} / "
            f"bottom={row['bottom_text'] or row['bottom_raw_text']}"
            for row in conflicts
        )
        lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def print_console_summary(output_dir: Path, photo_csv: Path, zone_csv: Path, summary_path: Path, total_photos: int) -> None:
    print("=" * 60)
    print("LPN zone OCR check")
    print("=" * 60)
    print(f"Total photos:    {total_photos}")
    print(f"Crops root:      {output_dir / 'crops'}")
    print(f"Photo results:   {photo_csv}")
    print(f"Zone results:    {zone_csv}")
    print(f"Summary report:  {summary_path}")
    print("=" * 60)


def main() -> None:
    args = parse_args()
    weights_path = args.weights.resolve()
    photos_dir = args.photos_dir.resolve()
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    photos = list_photos(photos_dir)
    detection_model = load_detection_model(weights_path)
    ocr_reader = load_ocr_reader()
    detection_results = run_detection(detection_model, photos, conf=args.conf, imgsz=args.imgsz)
    photo_rows, zone_rows = process_results(detection_results, ocr_reader, output_dir, args.ocr_width)

    photo_csv = write_csv(
        photo_rows,
        output_dir / "photo_results.csv",
        [
            "photo_filename",
            "decision",
            "final_lpn",
            "right_detected",
            "right_text",
            "right_raw_text",
            "right_detection_confidence",
            "right_ocr_confidence",
            "bottom_detected",
            "bottom_text",
            "bottom_raw_text",
            "bottom_detection_confidence",
            "bottom_ocr_confidence",
        ],
    )
    zone_csv = write_csv(
        zone_rows,
        output_dir / "zone_results.csv",
        [
            "photo_filename",
            "class_name",
            "detected",
            "detection_confidence",
            "crop_filename",
            "x1",
            "y1",
            "x2",
            "y2",
            "ocr_variant",
            "ocr_raw_text",
            "ocr_normalized_text",
            "ocr_confidence",
        ],
    )
    summary_path = write_summary(photo_rows, output_dir, weights_path, photos_dir)
    print_console_summary(output_dir, photo_csv, zone_csv, summary_path, len(photo_rows))


if __name__ == "__main__":
    main()

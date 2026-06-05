from __future__ import annotations

import re
from pathlib import Path

import cv2
from flask_socketio import SocketIO

from config import EXPECTED_ZONE_CLASSES, MODEL_ZONES, OCR_TARGET_WIDTH, YOLO_IMAGE_SIZE, ZONES_CONFIDENCE
from subsystem import results
from subsystem.pipeline_utils import clamp_box, registration_exists


LPN_DIGITS_PATTERN = re.compile(r"(\d{4})\D*(\d{6})")
_ZONES_MODEL = None
_ZONES_MODEL_LOAD_ATTEMPTED = False
_OCR_READER = None
_OCR_READER_LOAD_ATTEMPTED = False


def run_recognition(
    tag_image,
    tag_relative_path: str,
    session_id: str,
    mode_key: str,
    mode_config: dict,
    copied_path: Path,
    original_path: str,
    session_db,
    socketio: SocketIO,
) -> None:
    socketio.emit("block_update", {"block": "crnn", "status": "running"})

    zones_model = _get_zones_model()
    ocr_reader = _get_ocr_reader()
    if zones_model is None or ocr_reader is None:
        _record_recognition_failure(
            session_id=session_id,
            mode_key=mode_key,
            mode_config=mode_config,
            copied_path=copied_path,
            original_path=original_path,
            processed_path=tag_relative_path,
            session_db=session_db,
            socketio=socketio,
        )
        return

    try:
        predictions = zones_model.predict(
            source=tag_image,
            conf=ZONES_CONFIDENCE,
            imgsz=YOLO_IMAGE_SIZE,
            save=False,
            verbose=False,
        )
    except Exception as exc:
        print(f"LPN zone detection failed for {copied_path.name}: {exc}")
        _record_recognition_failure(
            session_id=session_id,
            mode_key=mode_key,
            mode_config=mode_config,
            copied_path=copied_path,
            original_path=original_path,
            processed_path=tag_relative_path,
            session_db=session_db,
            socketio=socketio,
        )
        return

    selected_zones = _choose_best_zones(predictions[0] if predictions else None)
    zone_rows: dict[str, dict] = {}
    height, width = tag_image.shape[:2]

    for class_name in EXPECTED_ZONE_CLASSES:
        detection = selected_zones.get(class_name)
        if detection is None:
            continue

        x1, y1, x2, y2 = clamp_box(*detection["xyxy"], width=width, height=height)
        crop = tag_image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        zone_rows[class_name] = run_zone_ocr(ocr_reader, crop, OCR_TARGET_WIDTH)

    _, final_lpn = evaluate_photo_decision(zone_rows)
    if final_lpn and registration_exists(final_lpn, session_db):
        socketio.emit("block_update", {"block": "crnn", "status": "done"})
        results.record_success(
            session_id=session_id,
            mode=mode_key,
            lpn=final_lpn,
            method="crnn",
            subsystem_on=mode_config["yolo"],
            photo_filename=copied_path.name,
            original_path=original_path,
            processed_path=tag_relative_path,
            session_db=session_db,
            socketio=socketio,
            qr_strategy=None,
        )
        return

    _record_recognition_failure(
        session_id=session_id,
        mode_key=mode_key,
        mode_config=mode_config,
        copied_path=copied_path,
        original_path=original_path,
        processed_path=tag_relative_path,
        session_db=session_db,
        socketio=socketio,
    )


def _get_zones_model():
    global _ZONES_MODEL, _ZONES_MODEL_LOAD_ATTEMPTED

    if _ZONES_MODEL_LOAD_ATTEMPTED:
        return _ZONES_MODEL

    _ZONES_MODEL_LOAD_ATTEMPTED = True
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics is not installed. LPN zone detection is unavailable.")
        return None

    if not MODEL_ZONES.exists():
        print(f"LPN zones weights not found: {MODEL_ZONES}")
        return None

    try:
        _ZONES_MODEL = YOLO(str(MODEL_ZONES))
    except Exception as exc:
        print(f"Failed to load LPN zones model {MODEL_ZONES}: {exc}")
        _ZONES_MODEL = None

    return _ZONES_MODEL


def _get_ocr_reader():
    global _OCR_READER, _OCR_READER_LOAD_ATTEMPTED

    if _OCR_READER_LOAD_ATTEMPTED:
        return _OCR_READER

    _OCR_READER_LOAD_ATTEMPTED = True
    try:
        import easyocr
    except ImportError:
        print("EasyOCR is not installed. OCR recognition is unavailable.")
        return None

    try:
        _OCR_READER = easyocr.Reader(["en"], gpu=False)
    except Exception as exc:
        print(f"Failed to initialize EasyOCR: {exc}")
        _OCR_READER = None

    return _OCR_READER


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


def _choose_best_zones(result) -> dict[str, dict]:
    selections: dict[str, dict] = {}
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return selections

    for xyxy, confidence, class_id in zip(
        result.boxes.xyxy.tolist(),
        result.boxes.conf.tolist(),
        result.boxes.cls.tolist(),
    ):
        class_name = result.names[int(class_id)]
        current = selections.get(class_name)
        candidate = {
            "xyxy": xyxy,
            "confidence": float(confidence),
        }
        if current is None or candidate["confidence"] > current["confidence"]:
            selections[class_name] = candidate

    return selections


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


def _record_recognition_failure(
    session_id: str,
    mode_key: str,
    mode_config: dict,
    copied_path: Path,
    original_path: str,
    processed_path: str | None,
    session_db,
    socketio: SocketIO,
) -> None:
    socketio.emit("block_update", {"block": "crnn", "status": "failed"})
    results.record_unidentified(
        session_id=session_id,
        mode=mode_key,
        subsystem_on=mode_config["yolo"],
        photo_filename=copied_path.name,
        original_path=original_path,
        processed_path=processed_path,
        session_db=session_db,
        socketio=socketio,
    )

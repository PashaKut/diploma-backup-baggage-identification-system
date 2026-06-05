from __future__ import annotations

from pathlib import Path

import cv2
from flask_socketio import SocketIO

from config import DETECTION_CONFIDENCE, DETECTOR_INPUT_SIZE, MODEL_DETECT, YOLO_IMAGE_SIZE
from subsystem import enhancement, recognition, results
from subsystem.pipeline_utils import clamp_box, save_processed_image


_DETECTOR_MODEL = None
_DETECTOR_LOAD_ATTEMPTED = False


def run_detection(
    session_id: str,
    mode_key: str,
    mode_config: dict,
    copied_path: Path,
    original_path: str,
    session_db,
    socketio: SocketIO,
) -> None:
    socketio.emit("block_update", {"block": "detection", "status": "running"})

    image = cv2.imread(str(copied_path))
    if image is None:
        _record_detection_failure(
            session_id=session_id,
            mode_key=mode_key,
            mode_config=mode_config,
            copied_path=copied_path,
            original_path=original_path,
            session_db=session_db,
            socketio=socketio,
        )
        return

    detector_input = preprocess_for_detection(image)
    model = _get_detection_model()
    if model is None:
        _record_detection_failure(
            session_id=session_id,
            mode_key=mode_key,
            mode_config=mode_config,
            copied_path=copied_path,
            original_path=original_path,
            session_db=session_db,
            socketio=socketio,
        )
        return

    try:
        predictions = model.predict(
            source=detector_input,
            conf=DETECTION_CONFIDENCE,
            imgsz=YOLO_IMAGE_SIZE,
            save=False,
            verbose=False,
        )
    except Exception as exc:
        print(f"Tag detection failed for {copied_path.name}: {exc}")
        _record_detection_failure(
            session_id=session_id,
            mode_key=mode_key,
            mode_config=mode_config,
            copied_path=copied_path,
            original_path=original_path,
            session_db=session_db,
            socketio=socketio,
        )
        return

    detection = _choose_best_detection(predictions[0] if predictions else None)
    if detection is None:
        _record_detection_failure(
            session_id=session_id,
            mode_key=mode_key,
            mode_config=mode_config,
            copied_path=copied_path,
            original_path=original_path,
            session_db=session_db,
            socketio=socketio,
        )
        return

    height, width = detector_input.shape[:2]
    x1, y1, x2, y2 = clamp_box(*detection["xyxy"], width=width, height=height)
    tag_crop = detector_input[y1:y2, x1:x2]
    if tag_crop.size == 0:
        _record_detection_failure(
            session_id=session_id,
            mode_key=mode_key,
            mode_config=mode_config,
            copied_path=copied_path,
            original_path=original_path,
            session_db=session_db,
            socketio=socketio,
        )
        return

    try:
        tag_relative_path = save_processed_image(
            session_id,
            f"{copied_path.stem}__tag.jpg",
            tag_crop,
        )
    except Exception as exc:
        print(f"Failed to save tag crop for {copied_path.name}: {exc}")
        _record_detection_failure(
            session_id=session_id,
            mode_key=mode_key,
            mode_config=mode_config,
            copied_path=copied_path,
            original_path=original_path,
            session_db=session_db,
            socketio=socketio,
        )
        return

    socketio.emit("block_update", {"block": "detection", "status": "done"})

    if mode_config["enhanced_qr"]:
        enhancement.run_enhancement(
            tag_image=tag_crop,
            tag_relative_path=tag_relative_path,
            session_id=session_id,
            mode_key=mode_key,
            mode_config=mode_config,
            copied_path=copied_path,
            original_path=original_path,
            session_db=session_db,
            socketio=socketio,
        )
        return

    if mode_config["crnn"]:
        recognition.run_recognition(
            tag_image=tag_crop,
            tag_relative_path=tag_relative_path,
            session_id=session_id,
            mode_key=mode_key,
            mode_config=mode_config,
            copied_path=copied_path,
            original_path=original_path,
            session_db=session_db,
            socketio=socketio,
        )
        return

    results.record_unidentified(
        session_id=session_id,
        mode=mode_key,
        subsystem_on=mode_config["yolo"],
        photo_filename=copied_path.name,
        original_path=original_path,
        processed_path=tag_relative_path,
        session_db=session_db,
    )


def preprocess_for_detection(image):
    square = crop_to_square_center(image)
    if square.shape[0] == DETECTOR_INPUT_SIZE and square.shape[1] == DETECTOR_INPUT_SIZE:
        return square
    return cv2.resize(
        square,
        (DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE),
        interpolation=cv2.INTER_AREA,
    )


def crop_to_square_center(image):
    height, width = image.shape[:2]
    side = min(height, width)

    y1 = (height - side) // 2
    x1 = (width - side) // 2
    y2 = y1 + side
    x2 = x1 + side
    return image[y1:y2, x1:x2]


def _get_detection_model():
    global _DETECTOR_MODEL, _DETECTOR_LOAD_ATTEMPTED

    if _DETECTOR_LOAD_ATTEMPTED:
        return _DETECTOR_MODEL

    _DETECTOR_LOAD_ATTEMPTED = True
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics is not installed. Tag detection is unavailable.")
        return None

    if not MODEL_DETECT.exists():
        print(f"Tag detector weights not found: {MODEL_DETECT}")
        return None

    try:
        _DETECTOR_MODEL = YOLO(str(MODEL_DETECT))
    except Exception as exc:
        print(f"Failed to load tag detector model {MODEL_DETECT}: {exc}")
        _DETECTOR_MODEL = None

    return _DETECTOR_MODEL


def _choose_best_detection(result) -> dict | None:
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return None

    best_confidence = -1.0
    best_detection = None
    for xyxy, confidence in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
        confidence_value = float(confidence)
        if confidence_value > best_confidence:
            best_confidence = confidence_value
            best_detection = {
                "xyxy": xyxy,
                "confidence": confidence_value,
            }
    return best_detection


def _record_detection_failure(
    session_id: str,
    mode_key: str,
    mode_config: dict,
    copied_path: Path,
    original_path: str,
    session_db,
    socketio: SocketIO,
) -> None:
    socketio.emit("block_update", {"block": "detection", "status": "failed"})
    results.record_unidentified(
        session_id=session_id,
        mode=mode_key,
        subsystem_on=mode_config["yolo"],
        photo_filename=copied_path.name,
        original_path=original_path,
        session_db=session_db,
        socketio=socketio,
    )

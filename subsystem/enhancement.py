from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from flask_socketio import SocketIO

from config import YOLO_IMAGE_SIZE
from read_qr import read_qr_from_ndarray
from subsystem import recognition, results
from subsystem.pipeline_utils import extract_valid_lpn, registration_exists, save_processed_image


def run_enhancement(
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
    socketio.emit("block_update", {"block": "enhanced_qr", "status": "running"})

    last_failure_path = tag_relative_path
    for variant_name, variant_image in _build_qr_variants(tag_image):
        qr_result = read_qr_from_ndarray(variant_image)
        decoded_lpn = extract_valid_lpn(qr_result)
        if not decoded_lpn or not registration_exists(decoded_lpn, session_db):
            continue

        processed_path = save_processed_image(
            session_id,
            f"{copied_path.stem}__qr_{variant_name}.jpg",
            variant_image,
        )
        qr_strategy = f"{variant_name}:{qr_result.get('strategy', 'direct')}"
        socketio.emit("block_update", {"block": "enhanced_qr", "status": "done"})
        results.record_success(
            session_id=session_id,
            mode=mode_key,
            lpn=decoded_lpn,
            method="qr_enhanced",
            subsystem_on=mode_config["yolo"],
            photo_filename=copied_path.name,
            original_path=original_path,
            processed_path=processed_path,
            session_db=session_db,
            socketio=socketio,
            qr_strategy=qr_strategy,
        )
        return

    socketio.emit("block_update", {"block": "enhanced_qr", "status": "failed"})
    if mode_config["crnn"]:
        recognition.run_recognition(
            tag_image=tag_image,
            tag_relative_path=last_failure_path,
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
        processed_path=last_failure_path,
        session_db=session_db,
        socketio=socketio,
    )


def _build_qr_variants(tag_image) -> list[tuple[str, np.ndarray]]:

    upscaled = _scale_for_qr(tag_image, 2.0)
    gray = cv2.cvtColor(tag_image, cv2.COLOR_BGR2GRAY)
    gray_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    _, gray_otsu = cv2.threshold(gray_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    scale_4x = _scale_for_qr(tag_image, 4.0)

    return [
        ("gray", gray),
        ("gray_clahe", gray_clahe),
        ("gray_otsu", gray_otsu),
        ("scale_2x", upscaled),
        ("scale_4x", scale_4x),
    ]


def _scale_for_qr(tag_image, scale: float) -> np.ndarray:
    height, width = tag_image.shape[:2]
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    return cv2.resize(tag_image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

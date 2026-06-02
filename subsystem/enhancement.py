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
    )


def _build_qr_variants(tag_image) -> list[tuple[str, np.ndarray]]:
    upscaled = _upscale_for_qr(tag_image)
    gaussian = cv2.GaussianBlur(upscaled, (3, 3), 0)
    unsharp = cv2.addWeighted(upscaled, 1.5, cv2.GaussianBlur(upscaled, (0, 0), 1.0), -0.5, 0)

    variants = [
        ("upscale", upscaled),
        ("gaussian", gaussian),
        ("unsharp", unsharp),
    ]

    perspective = _attempt_perspective_correction(upscaled)
    if perspective is not None:
        variants.append(("perspective", perspective))

    return variants


def _upscale_for_qr(tag_image) -> np.ndarray:
    height, width = tag_image.shape[:2]
    longest_side = max(height, width)
    if longest_side >= YOLO_IMAGE_SIZE:
        return tag_image.copy()

    scale = YOLO_IMAGE_SIZE / longest_side
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    return cv2.resize(tag_image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)


def _attempt_perspective_correction(image) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(polygon) != 4:
            continue

        ordered_points = _order_points(polygon.reshape(4, 2).astype("float32"))
        target_width = int(
            max(
                np.linalg.norm(ordered_points[1] - ordered_points[0]),
                np.linalg.norm(ordered_points[2] - ordered_points[3]),
            )
        )
        target_height = int(
            max(
                np.linalg.norm(ordered_points[3] - ordered_points[0]),
                np.linalg.norm(ordered_points[2] - ordered_points[1]),
            )
        )
        if target_width < 20 or target_height < 20:
            continue

        destination = np.array(
            [
                [0, 0],
                [target_width - 1, 0],
                [target_width - 1, target_height - 1],
                [0, target_height - 1],
            ],
            dtype="float32",
        )
        matrix = cv2.getPerspectiveTransform(ordered_points, destination)
        return cv2.warpPerspective(image, matrix, (target_width, target_height))

    return None


def _order_points(points: np.ndarray) -> np.ndarray:
    ordered = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1)

    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered

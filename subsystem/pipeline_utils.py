from __future__ import annotations

import re
from pathlib import Path

import cv2
from sqlalchemy import select

from config import SESSIONS_DIR
from database.models import BaggageRegistration


LPN_PATTERN = re.compile(r"^\d{4} \d{6}$")


def extract_valid_lpn(qr_result: dict | None) -> str | None:
    if not qr_result or not qr_result.get("success"):
        return None

    parsed = qr_result.get("parsed") or {}
    lpn = (parsed.get("lpn") or "").strip()
    if not LPN_PATTERN.fullmatch(lpn):
        return None
    return lpn


def registration_exists(lpn: str, session_db) -> bool:
    return (
        session_db.execute(select(BaggageRegistration.id).where(BaggageRegistration.lpn == lpn))
        .scalar_one_or_none()
        is not None
    )


def ensure_processed_dir(session_id: str) -> Path:
    processed_dir = SESSIONS_DIR / session_id / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir


def save_processed_image(session_id: str, filename: str, image) -> str:
    processed_dir = ensure_processed_dir(session_id)
    image_path = processed_dir / filename
    if not cv2.imwrite(str(image_path), image):
        raise RuntimeError(f"Failed to save processed image: {image_path}")
    return image_path.relative_to(SESSIONS_DIR).as_posix()


def clamp_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[int, int, int, int]:
    left = max(0, min(int(round(x1)), width - 1))
    top = max(0, min(int(round(y1)), height - 1))
    right = max(left + 1, min(int(round(x2)), width))
    bottom = max(top + 1, min(int(round(y2)), height))
    return left, top, right, bottom


def sanitize_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value).strip("_") or "image"

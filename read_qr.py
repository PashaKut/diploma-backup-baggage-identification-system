"""
Baseline QR reader for baggage tag images.

Usage:
    python read_qr.py <path_to_image>

The QR payload is expected to contain only the LPN, for example:
    0014 123456
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np


MAX_WORKING_LONGEST_SIDE = 1800
MAX_UPSCALE_OUTPUT_PIXELS = 12_000_000


def parse_qr_data(raw: str) -> dict:
    lpn = raw.strip()
    if not lpn:
        return {"raw": raw, "error": "QR code is empty"}
    return {"lpn": lpn}


def try_decode(img: np.ndarray, detector: cv2.QRCodeDetector) -> tuple[str | None, np.ndarray | None]:
    data, points, _ = detector.detectAndDecode(img)
    if data:
        return data, points

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        data, points, _ = detector.detectAndDecode(gray)
        if data:
            return data, points

    return None, None


def prepare_working_image(img: np.ndarray) -> np.ndarray:
    """
    Bound processing cost for large camera photos.
    Very large images can hang or exhaust memory when blindly upscaled.
    """
    height, width = img.shape[:2]
    longest_side = max(height, width)
    if longest_side <= MAX_WORKING_LONGEST_SIDE:
        return img

    scale = MAX_WORKING_LONGEST_SIDE / longest_side
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)


def read_qr_from_ndarray(img: np.ndarray) -> dict:
    if img is None or img.size == 0:
        return {"success": False, "error": "Invalid image array"}

    working_img = prepare_working_image(img)
    detector = cv2.QRCodeDetector()

    data, points = try_decode(working_img, detector)
    if data:
        return {
            "success": True,
            "raw": data,
            "parsed": parse_qr_data(data),
            "points": points.tolist() if points is not None else None,
            "strategy": "direct",
        }

    height, width = working_img.shape[:2]
    for scale in [2, 4]:
        if (height * scale) * (width * scale) > MAX_UPSCALE_OUTPUT_PIXELS:
            continue

        upscaled = cv2.resize(
            working_img,
            (width * scale, height * scale),
            interpolation=cv2.INTER_CUBIC,
        )
        data, points = try_decode(upscaled, detector)
        if data:
            normalized_points = (points / scale).tolist() if points is not None else None
            return {
                "success": True,
                "raw": data,
                "parsed": parse_qr_data(data),
                "points": normalized_points,
                "strategy": f"upscale_x{scale}",
            }

    gray = cv2.cvtColor(working_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    data, points = try_decode(enhanced, detector)
    if data:
        return {
            "success": True,
            "raw": data,
            "parsed": parse_qr_data(data),
            "points": points.tolist() if points is not None else None,
            "strategy": "clahe",
        }

    return {"success": False, "error": "QR code not found or could not be decoded"}


def read_qr_from_image(image_path: str) -> dict:
    if not os.path.exists(image_path):
        return {"success": False, "error": f"File not found: {image_path}"}

    img = cv2.imread(image_path)
    if img is None:
        return {"success": False, "error": "Failed to load image"}

    return read_qr_from_ndarray(img)


def print_result(result: dict, filename: str = "") -> None:
    print("\n" + "=" * 50)
    if filename:
        print(f"  File: {filename}")
    print("  QR READ RESULT")
    print("=" * 50)

    if not result["success"]:
        print(f"  ERROR: {result['error']}")
        print("=" * 50)
        return

    parsed = result["parsed"]
    strategy = result.get("strategy", "-")

    if "error" in parsed:
        print(f"  Raw value: {result['raw']}")
        print(f"  Error:     {parsed['error']}")
    else:
        print(f"  Strategy:  {strategy}")
        print(f"  LPN:       {parsed['lpn']}")

    print("=" * 50)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python read_qr.py <path_to_image>")
        sys.exit(1)

    result = read_qr_from_image(sys.argv[1])
    print_result(result, filename=os.path.basename(sys.argv[1]))
    sys.exit(0 if result["success"] else 1)

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


def read_qr_from_ndarray(img: np.ndarray) -> dict:
    if img is None or img.size == 0:
        return {"success": False, "error": "Invalid image array"}

    detector = cv2.QRCodeDetector()

    data, points = try_decode(img, detector)
    if data:
        return {
            "success": True,
            "raw": data,
            "parsed": parse_qr_data(data),
            "points": points.tolist() if points is not None else None,
            "strategy": "direct",
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

# Baggage Tag Fallback Identification Subsystem

A web-based demonstrator of a fallback identification pipeline for airport baggage handling systems (BHS). When a primary QR scan fails, the system attempts to recover the baggage LPN (License Plate Number) using computer vision and OCR — without operator intervention.

## Pipeline

1. **Direct QR decode** — fast path, covers most images
2. **YOLO tag detection + image enhancement** — crops and improves the tag, retries QR decode
3. **YOLO LPN zone detection + EasyOCR** — reads the printed LPN text directly

Each stage activates only if the previous one fails.

## Usage

1. Import a CSV file with registered baggage LPNs
2. Select a pipeline mode: **basic**, **enhanced**, or **full**
3. Upload a batch of tag photos and start a session
4. Watch results update in real time
5. Compare sessions on the results page

## Stack

Python, Flask, YOLOv8n, EasyOCR, OpenCV, SQLite, SQLAlchemy, Socket.IO

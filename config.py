from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATABASE_URL = "sqlite:///baggage.db"

DATA_DIR = BASE_DIR / "data"
PHOTOS_DIR = DATA_DIR / "photos"
CSV_PATH = DATA_DIR / "registrations.csv"
SESSIONS_DIR = DATA_DIR / "sessions"
MODELS_DIR = BASE_DIR / "models"
MODEL_DETECT = (
    BASE_DIR
    / "runs"
    / "detect"
    / "runs"
    / "train"
    / "detector_20260506_012226"
    / "weights"
    / "best.pt"
)
MODEL_ZONES = (
    BASE_DIR
    / "runs"
    / "detect"
    / "runs"
    / "train"
    / "lpn_zones_20260509_003420"
    / "weights"
    / "best.pt"
)

DETECTION_CONFIDENCE = 0.25
ZONES_CONFIDENCE = 0.25
YOLO_IMAGE_SIZE = 640
DETECTOR_INPUT_SIZE = 640
OCR_TARGET_WIDTH = 640
EXPECTED_ZONE_CLASSES = ("LPN_right", "LPN_bottom")

MODES = {
    "basic": {"yolo": False, "enhanced_qr": False, "crnn": False, "direct_qr": True},
    "enhanced": {"yolo": True, "enhanced_qr": True, "crnn": False, "direct_qr": True},
    "full": {"yolo": True, "enhanced_qr": True, "crnn": True, "direct_qr": True},
    "qr_enhanced_only": {"yolo": True, "enhanced_qr": True, "crnn": False, "direct_qr": False},
    "ocr_only": {"yolo": True, "enhanced_qr": False, "crnn": True, "direct_qr": False},
}

SESSION_IMAGE_ROUTE = "/static/sessions"
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

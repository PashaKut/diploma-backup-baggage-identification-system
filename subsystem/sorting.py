import shutil
import time
from pathlib import Path

from flask_socketio import SocketIO
from sqlalchemy import select

from config import ALLOWED_IMAGE_SUFFIXES, PHOTOS_DIR, SESSIONS_DIR
from database.models import BaggageRegistration
from read_qr import read_qr_from_image
from subsystem import detection
from subsystem import results
from subsystem.pipeline_utils import extract_valid_lpn, registration_exists


def run_sorting(session_id: str, mode_key: str, mode_config: dict, session_db, socketio: SocketIO) -> int:
    socketio.emit("block_update", {"block": "sorting", "status": "running"})
    registrations = session_db.execute(
        select(BaggageRegistration).order_by(BaggageRegistration.id.asc())
    ).scalars().all()
    results.initialize_session_results(
        session_id=session_id,
        mode=mode_key,
        subsystem_on=mode_config["yolo"],
        registrations=registrations,
        session_db=session_db,
    )

    photo_paths = _list_photo_files()
    total = len(photo_paths)

    for index, photo_path in enumerate(photo_paths, start=1):
        socketio.emit(
            "block_update",
            {"block": "sorting", "status": "running", "current": index, "total": total},
        )
        copied_path = _copy_original_photo(session_id, photo_path)
        time.sleep(0.1)
        try:
            _process_photo(
                session_id=session_id,
                mode_key=mode_key,
                mode_config=mode_config,
                copied_path=copied_path,
                session_db=session_db,
                socketio=socketio,
            )
        except Exception as exc:
            print(f"Error while processing {copied_path.name}: {exc}")
            results.record_unidentified(
                session_id=session_id,
                mode=mode_key,
                subsystem_on=mode_config["yolo"],
                photo_filename=copied_path.name,
                original_path=copied_path.relative_to(SESSIONS_DIR).as_posix(),
                session_db=session_db,
            )

    results.finalize_session_results(session_id=session_id, session_db=session_db, socketio=socketio)
    socketio.emit("block_update", {"block": "sorting", "status": "done", "count": total})
    return total


def _list_photo_files() -> list[Path]:
    if not PHOTOS_DIR.exists():
        return []
    return sorted(
        path for path in PHOTOS_DIR.iterdir() if path.is_file() and path.suffix.lower() in ALLOWED_IMAGE_SUFFIXES
    )


def _copy_original_photo(session_id: str, photo_path: Path) -> Path:
    session_original_dir = SESSIONS_DIR / session_id / "original"
    session_original_dir.mkdir(parents=True, exist_ok=True)
    target = session_original_dir / photo_path.name
    shutil.copy2(photo_path, target)
    return target


def _process_photo(
    session_id: str,
    mode_key: str,
    mode_config: dict,
    copied_path: Path,
    session_db,
    socketio: SocketIO,
) -> None:
    relative_original_path = copied_path.relative_to(SESSIONS_DIR).as_posix()
    if mode_config["direct_qr"]:
        qr_result = read_qr_from_image(str(copied_path))
        qr_strategy = qr_result.get("strategy")
        decoded_lpn = extract_valid_lpn(qr_result)

        if decoded_lpn and registration_exists(decoded_lpn, session_db):
            results.record_success(
                session_id=session_id,
                mode=mode_key,
                lpn=decoded_lpn,
                method="qr_direct",
                subsystem_on=mode_config["yolo"],
                photo_filename=copied_path.name,
                original_path=relative_original_path,
                processed_path=None,
                session_db=session_db,
                socketio=socketio,
                qr_strategy=qr_strategy,
            )
            return

        if decoded_lpn:
            results.record_unidentified(
                session_id=session_id,
                mode=mode_key,
                subsystem_on=mode_config["yolo"],
                photo_filename=copied_path.name,
                original_path=relative_original_path,
                session_db=session_db,
            )
            return

    if mode_config["yolo"]:
        detection.run_detection(
            session_id=session_id,
            mode_key=mode_key,
            mode_config=mode_config,
            copied_path=copied_path,
            original_path=relative_original_path,
            session_db=session_db,
            socketio=socketio,
        )
        return

    results.record_unidentified(
        session_id=session_id,
        mode=mode_key,
        subsystem_on=mode_config["yolo"],
        photo_filename=copied_path.name,
        original_path=relative_original_path,
        session_db=session_db,
    )

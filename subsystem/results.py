from flask_socketio import SocketIO

from database.models import MatchedImageResult, SortingResult, UnidentifiedImageResult


def initialize_session_results(
    session_id: str,
    mode: str,
    subsystem_on: bool,
    registrations,
    session_db,
) -> None:
    session_db.add_all(
        SortingResult(
            session_id=session_id,
            mode=mode,
            lpn=registration.lpn,
            status="failed",
            method="none",
            subsystem_on=subsystem_on,
            photo_filename=None,
            original_image_path=None,
            processed_image_path=None,
        )
        for registration in registrations
    )
    session_db.commit()


def record_success(
    session_id: str,
    mode: str,
    lpn: str,
    method: str,
    subsystem_on: bool,
    photo_filename: str,
    original_path: str,
    session_db,
    socketio: SocketIO,
    processed_path: str | None = None,
    qr_strategy: str | None = None,
) -> None:
    session_db.add(
        MatchedImageResult(
            session_id=session_id,
            lpn=lpn,
            method=method,
            qr_strategy=qr_strategy,
            photo_filename=photo_filename,
            original_image_path=original_path,
            processed_image_path=processed_path,
        )
    )

    session_result = session_db.query(SortingResult).filter_by(session_id=session_id, lpn=lpn).one()
    session_result.status = "success"
    session_result.method = method
    session_result.subsystem_on = subsystem_on
    if session_result.photo_filename is None:
        session_result.photo_filename = photo_filename
        session_result.original_image_path = original_path
        session_result.processed_image_path = processed_path

    session_db.commit()

    socketio.emit(
        "counter_update",
        {"block": "identified", "delta": 1, "method": method},
    )


def record_unidentified(
    session_id: str,
    mode: str,
    subsystem_on: bool,
    photo_filename: str,
    original_path: str,
    session_db,
    socketio: SocketIO,
    processed_path: str | None = None,
) -> None:
    session_db.add(
        UnidentifiedImageResult(
            session_id=session_id,
            mode=mode,
            subsystem_on=subsystem_on,
            photo_filename=photo_filename,
            original_image_path=original_path,
            processed_image_path=processed_path,
        )
    )
    session_db.commit()
    socketio.emit("counter_update", {"block": "lost", "delta": 1})


def finalize_session_results(session_id: str, session_db, socketio: SocketIO) -> None:
    return None

from threading import Thread
from uuid import uuid4

from flask import Flask, jsonify, request, send_from_directory
from flask_socketio import SocketIO
from sqlalchemy import func, select

from config import MODES, SESSIONS_DIR
from database.connection import SessionLocal, init_db
from database.models import (
    BaggageRegistration,
    MatchedImageResult,
    SortingResult,
    UnidentifiedImageResult,
)
from state import app_state
from subsystem.registration import run_registration
from subsystem.sorting import run_sorting


app = Flask(__name__, static_folder="static", static_url_path="/static")
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/results")
def results_page():
    return app.send_static_file("results.html")


@app.get("/api/modes")
def get_modes():
    return jsonify(
        {
            "modes": list(MODES.keys()),
            "current_mode": app_state["current_mode"],
            "mode_details": MODES,
        }
    )


@app.post("/api/set_mode")
def set_mode():
    if app_state["is_running"]:
        return jsonify({"error": "Cannot change mode while a session is running."}), 400

    payload = request.get_json(silent=True) or {}
    mode = payload.get("mode")
    if mode not in MODES:
        return jsonify({"error": "Unknown mode."}), 400

    app_state["current_mode"] = mode
    return jsonify({"mode": mode})


@app.post("/api/start")
def start_session():
    if app_state["is_running"]:
        return jsonify({"error": "A session is already running."}), 400

    session_id = str(uuid4())
    mode = app_state["current_mode"]
    app_state["is_running"] = True
    app_state["session_id"] = session_id

    socketio.emit("session_started", {"session_id": session_id, "mode": mode})

    worker = Thread(target=_run_session_pipeline, args=(session_id, mode), daemon=True)
    worker.start()
    return jsonify({"session_id": session_id, "mode": mode}), 202


@app.get("/api/results/sessions")
def get_session_summaries():
    session_db = SessionLocal()
    try:
        matched_rows = session_db.execute(
            select(
                MatchedImageResult.session_id.label("session_id"),
                MatchedImageResult.method.label("method"),
                MatchedImageResult.qr_strategy.label("qr_strategy"),
                func.count(MatchedImageResult.id).label("count"),
            )
            .group_by(
                MatchedImageResult.session_id,
                MatchedImageResult.method,
                MatchedImageResult.qr_strategy,
            )
        ).all()
        unidentified_rows = session_db.execute(
            select(
                UnidentifiedImageResult.session_id.label("session_id"),
                func.count(UnidentifiedImageResult.id).label("count"),
            )
            .group_by(UnidentifiedImageResult.session_id)
        ).all()

        sessions_by_id: dict[str, dict] = {}
        for row in matched_rows:
            session = sessions_by_id.setdefault(
                row.session_id,
                {
                    "session_id": row.session_id,
                    "session_id_short": row.session_id[:8],
                    "mode": None,
                    "total": 0,
                    "identified": 0,
                    "lost": 0,
                    "qr_direct": 0,
                    "qr_enhanced": 0,
                    "crnn": 0,
                },
            )
            session["identified"] += row.count or 0
            session["total"] += row.count or 0
            if row.method == "qr_direct":
                session["qr_direct"] += row.count or 0
            elif row.method == "qr_enhanced":
                session["qr_enhanced"] += row.count or 0
            elif row.method == "crnn":
                session["crnn"] += row.count or 0

        for row in unidentified_rows:
            session = sessions_by_id.setdefault(
                row.session_id,
                {
                    "session_id": row.session_id,
                    "session_id_short": row.session_id[:8],
                    "mode": None,
                    "total": 0,
                    "identified": 0,
                    "lost": 0,
                    "qr_direct": 0,
                    "qr_enhanced": 0,
                    "crnn": 0,
                },
            )
            session["lost"] += row.count or 0
            session["total"] += row.count or 0

        mode_rows = session_db.execute(
            select(
                SortingResult.session_id,
                SortingResult.mode,
            )
            .group_by(SortingResult.session_id, SortingResult.mode)
        ).all()
        for row in mode_rows:
            session = sessions_by_id.setdefault(
                row.session_id,
                {
                    "session_id": row.session_id,
                    "session_id_short": row.session_id[:8],
                    "mode": None,
                    "total": 0,
                    "identified": 0,
                    "lost": 0,
                    "qr_direct": 0,
                    "qr_enhanced": 0,
                    "crnn": 0,
                },
            )
            session["mode"] = row.mode

        sessions = sorted(
            sessions_by_id.values(),
            key=lambda item: item["session_id"],
            reverse=True,
        )
        return jsonify({"sessions": sessions})
    finally:
        session_db.close()


@app.get("/api/results/session/<session_id>")
def get_session_results(session_id: str):
    session_db = SessionLocal()
    try:
        registration_rows = session_db.execute(
            select(SortingResult, BaggageRegistration)
            .join(BaggageRegistration, SortingResult.lpn == BaggageRegistration.lpn)
            .where(SortingResult.session_id == session_id)
            .order_by(BaggageRegistration.id.asc())
        ).all()
        matched_images = session_db.execute(
            select(MatchedImageResult)
            .where(MatchedImageResult.session_id == session_id)
            .order_by(MatchedImageResult.id.asc())
        ).scalars().all()
        unidentified_images = session_db.execute(
            select(UnidentifiedImageResult)
            .where(UnidentifiedImageResult.session_id == session_id)
            .order_by(UnidentifiedImageResult.id.asc())
        ).scalars().all()

        matched_images_by_lpn: dict[str, list[dict]] = {}
        for image in matched_images:
            matched_images_by_lpn.setdefault(image.lpn, []).append(
                {
                    "id": image.id,
                    "method": image.method,
                    "qr_strategy": image.qr_strategy,
                    "photo_filename": image.photo_filename,
                    "original_image_path": image.original_image_path,
                    "processed_image_path": image.processed_image_path,
                }
            )

        registration_results = [
            {
                "id": result.id,
                "session_id": result.session_id,
                "mode": result.mode,
                "lpn": result.lpn,
                "status": result.status,
                "method": result.method,
                "route_from": registration.route_from,
                "route_to": registration.route_to,
                "flight": registration.flight,
                "date": registration.date,
                "passenger": registration.passenger,
                "baggage_class": registration.baggage_class,
                "pieces": registration.pieces,
                "baggage_type": registration.baggage_type,
                "matched_images": matched_images_by_lpn.get(result.lpn, []),
            }
            for result, registration in registration_rows
        ]
        unidentified_payload = [
            {
                "id": image.id,
                "session_id": image.session_id,
                "mode": image.mode,
                "photo_filename": image.photo_filename,
                "original_image_path": image.original_image_path,
                "processed_image_path": image.processed_image_path,
            }
            for image in unidentified_images
        ]
        return jsonify(
            {
                "registration_results": registration_results,
                "unidentified_images": unidentified_payload,
            }
        )
    finally:
        session_db.close()


@app.get("/static/sessions/<path:asset_path>")
def serve_session_asset(asset_path: str):
    return send_from_directory(SESSIONS_DIR, asset_path)


def _run_session_pipeline(session_id: str, mode: str) -> None:
    session_db = SessionLocal()
    try:
        run_registration(session_db, socketio)
        run_sorting(session_id, mode, MODES[mode], session_db, socketio)
    finally:
        session_db.close()
        SessionLocal.remove()
        app_state["is_running"] = False
        app_state["session_id"] = None
        socketio.emit("session_finished", {"session_id": session_id})


def bootstrap() -> None:
    init_db()


if __name__ == "__main__":
    bootstrap()
    socketio.run(app, debug=True)

import csv

from flask_socketio import SocketIO
from sqlalchemy import func, select

from config import CSV_PATH
from database.models import BaggageRegistration


EXPECTED_COLUMNS = ["lpn", "from", "to", "flight", "date", "passenger", "class", "pieces", "type"]


def run_registration(session_db, socketio: SocketIO) -> int:
    socketio.emit("block_update", {"block": "registration", "status": "running"})

    existing_count = session_db.scalar(select(func.count(BaggageRegistration.id))) or 0
    if existing_count:
        socketio.emit(
            "block_update",
            {"block": "registration", "status": "done", "count": existing_count},
        )
        return existing_count

    rows = _load_csv_rows()
    for row in rows:
        session_db.add(
            BaggageRegistration(
                lpn=row["lpn"].strip(),
                route_from=row["from"].strip(),
                route_to=row["to"].strip(),
                flight=row["flight"].strip(),
                date=row["date"].strip(),
                passenger=row["passenger"].strip(),
                baggage_class=row["class"].strip(),
                pieces=row["pieces"].strip(),
                baggage_type=row["type"].strip(),
            )
        )

    session_db.commit()
    socketio.emit(
        "block_update",
        {"block": "registration", "status": "done", "count": len(rows)},
    )
    return len(rows)


def _load_csv_rows() -> list[dict[str, str]]:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(
                f"CSV columns must be exactly {EXPECTED_COLUMNS}, got {reader.fieldnames!r}"
            )

        rows = []
        for line_number, row in enumerate(reader, start=2):
            cleaned_row = {key: (value or "").strip() for key, value in row.items()}
            if not all(cleaned_row.values()):
                raise ValueError(f"CSV row {line_number} contains empty values.")
            rows.append(cleaned_row)

    return rows

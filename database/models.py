from sqlalchemy import Boolean, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class BaggageRegistration(Base):
    __tablename__ = "baggage_registrations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lpn: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    route_from: Mapped[str] = mapped_column(String(5), nullable=False)
    route_to: Mapped[str] = mapped_column(String(5), nullable=False)
    flight: Mapped[str] = mapped_column(String(15), nullable=False)
    date: Mapped[str] = mapped_column(String(10), nullable=False)
    passenger: Mapped[str] = mapped_column(String(50), nullable=False)
    baggage_class: Mapped[str] = mapped_column(String(5), nullable=False)
    pieces: Mapped[str] = mapped_column(String(10), nullable=False)
    baggage_type: Mapped[str] = mapped_column(String(15), nullable=False)


class SortingResult(Base):
    __tablename__ = "sorting_results"
    __table_args__ = (UniqueConstraint("session_id", "lpn", name="uq_sorting_results_session_lpn"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    mode: Mapped[str] = mapped_column(String(20), nullable=False)
    lpn: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(10), nullable=False)
    method: Mapped[str] = mapped_column(String(20), nullable=False)
    subsystem_on: Mapped[bool] = mapped_column(Boolean, nullable=False)
    photo_filename: Mapped[str | None] = mapped_column(String(100), nullable=True)
    original_image_path: Mapped[str | None] = mapped_column(String(200), nullable=True)
    processed_image_path: Mapped[str | None] = mapped_column(String(200), nullable=True)


class MatchedImageResult(Base):
    __tablename__ = "matched_image_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    lpn: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    method: Mapped[str] = mapped_column(String(20), nullable=False)
    qr_strategy: Mapped[str | None] = mapped_column(String(32), nullable=True)
    photo_filename: Mapped[str] = mapped_column(String(100), nullable=False)
    original_image_path: Mapped[str] = mapped_column(String(200), nullable=False)
    processed_image_path: Mapped[str | None] = mapped_column(String(200), nullable=True)


class UnidentifiedImageResult(Base):
    __tablename__ = "unidentified_image_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    mode: Mapped[str] = mapped_column(String(20), nullable=False)
    subsystem_on: Mapped[bool] = mapped_column(Boolean, nullable=False)
    photo_filename: Mapped[str] = mapped_column(String(100), nullable=False)
    original_image_path: Mapped[str] = mapped_column(String(200), nullable=False)
    processed_image_path: Mapped[str | None] = mapped_column(String(200), nullable=True)

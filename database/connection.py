import shutil

from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

from config import BASE_DIR, DATABASE_URL, SESSIONS_DIR
from database.models import Base


engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    future=True,
)
SessionLocal = scoped_session(
    sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
)


def init_db() -> None:
    SessionLocal.remove()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    _reset_sessions_dir()


def _reset_sessions_dir() -> None:
    if SESSIONS_DIR.exists():
        shutil.rmtree(SESSIONS_DIR)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

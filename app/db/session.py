from collections.abc import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings

settings = get_settings()

# Gunakan pool_pre_ping=True agar koneksi terputus otomatis dipulihkan
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Kelas dasar deklaratif SQLAlchemy 2.x untuk seluruh model ORM."""
    pass


def get_db() -> Generator[Session, None, None]:
    """Dependency FastAPI untuk menyediakan sesi database per request dan menutupnya dengan aman."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
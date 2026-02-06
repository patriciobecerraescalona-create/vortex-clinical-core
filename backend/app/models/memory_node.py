import uuid
import os
from sqlalchemy import Column, DateTime, Text
# from sqlalchemy.dialects.postgresql import UUID <-- Reemplazado
from sqlalchemy.sql import func
from sqlalchemy import String

from backend.app.db.base import Base

# ==========================================
# Compatibilidad LAB (SQLite) vs PROD (Postgres)
# ==========================================
is_sqlite = "sqlite" in os.getenv("DATABASE_URL", "") or not os.getenv("DATABASE_URL")

if is_sqlite:
    # SQLite fallback
    UUID_TYPE = String(36)
    def uuid_gen():
        return str(uuid.uuid4())
else:
    from sqlalchemy.dialects.postgresql import UUID
    UUID_TYPE = UUID(as_uuid=True)
    def uuid_gen():
        return uuid.uuid4()


class MemoryNode(Base):
    __tablename__ = "memory_nodes"

    id = Column(UUID_TYPE, primary_key=True, default=uuid_gen)
    user_id = Column(UUID_TYPE, nullable=False, index=True)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

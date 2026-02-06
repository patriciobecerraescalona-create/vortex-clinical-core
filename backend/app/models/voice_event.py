import uuid
import os
from sqlalchemy import Column, String, DateTime, Text
# from sqlalchemy.dialects.postgresql import UUID, JSONB  <-- Reemplazado por lógica condicional
from sqlalchemy.sql import func

from backend.app.db.base import Base

# ==========================================
# Compatibilidad LAB (SQLite) vs PROD (Postgres)
# ==========================================
is_sqlite = "sqlite" in os.getenv("DATABASE_URL", "") or not os.getenv("DATABASE_URL")

if is_sqlite:
    from sqlalchemy import JSON
    JSONB = JSON
    # En SQLite usamos String(36) para UUIDs
    # y lambda para generar el string
    UUID_TYPE = String(36)
    def uuid_gen():
        return str(uuid.uuid4())
else:
    from sqlalchemy.dialects.postgresql import UUID, JSONB
    # En Postgres usamos UUID nativo
    UUID_TYPE = UUID(as_uuid=True)
    def uuid_gen():
        return uuid.uuid4()


class VoiceEvent(Base):
    __tablename__ = "voice_events"

    id = Column(UUID_TYPE, primary_key=True, default=uuid_gen)
    procedure_id = Column(UUID_TYPE, nullable=False, index=True)
    user_id = Column(UUID_TYPE, nullable=False)

    intent = Column(String, nullable=False)
    confidence = Column(String, nullable=False)
    raw_text = Column(Text, nullable=False)
    feedback = Column(JSONB, nullable=False)

    source = Column(String, nullable=False)
    user_role = Column(String, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

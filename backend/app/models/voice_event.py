import uuid
from sqlalchemy import Column, String, DateTime, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func

from backend.app.db.base import Base


class VoiceEvent(Base):
    __tablename__ = "voice_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    procedure_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), nullable=False)

    intent = Column(String, nullable=False)
    confidence = Column(String, nullable=False)
    raw_text = Column(Text, nullable=False)
    feedback = Column(JSONB, nullable=False)

    source = Column(String, nullable=False)
    user_role = Column(String, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

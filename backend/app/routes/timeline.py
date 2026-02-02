from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from uuid import UUID

from backend.app.db.session import SessionLocal
from backend.app.models.voice_event import VoiceEvent

router = APIRouter(prefix="/procedures", tags=["Timeline"])


# =========================
# DB DEPENDENCY
# =========================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =========================
# TIMELINE ENDPOINT
# =========================

@router.get("/{procedure_id}/timeline")
def get_procedure_timeline(
    procedure_id: UUID,
    db: Session = Depends(get_db),
):
    events = (
        db.query(VoiceEvent)
        .filter(VoiceEvent.procedure_id == procedure_id)
        .order_by(VoiceEvent.created_at.asc())
        .all()
    )

    if not events:
        raise HTTPException(
            status_code=404,
            detail="No timeline events found for this procedure"
        )

    return {
        "procedure_id": str(procedure_id),
        "count": len(events),
        "timeline": [
            {
                "id": str(e.id),
                "intent": e.intent,
                "confidence": e.confidence,
                "raw_text": e.raw_text,
                "feedback": e.feedback,
                "source": e.source,
                "user_role": e.user_role,
                "created_at": e.created_at.isoformat(),
            }
            for e in events
        ],
    }

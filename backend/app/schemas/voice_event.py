from pydantic import BaseModel
from uuid import UUID
from typing import Optional, Dict, Any


class VoiceEventIn(BaseModel):
    user_id: UUID
    procedure_id: Optional[UUID] = None

    source: str                 # "text" | "voice"
    user_role: str              # "leader" | "staff"

    raw_text: str

    # 👇 NUEVO: opciones de ejecución (LAB / Front)
    options: Optional[Dict[str, Any]] = {}

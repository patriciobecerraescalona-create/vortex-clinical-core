from uuid import UUID, uuid4
from datetime import datetime

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.voice_event import VoiceEvent
from app.models.memory_node import MemoryNode

from app.services.kai_engine import process_kai_activation
from app.services.agent_context import build_layer1_context
from app.services.llm_agent import run_llm


def handle_voice_event(payload, db: Session) -> dict:
    """
    Handler central de eventos (voz / texto).
    """

    # -------------------------
    # 1. KAI
    # -------------------------
    kai = process_kai_activation(
        user_text=payload.raw_text,
        user_context=payload.options or {},
    )

    agent = kai.get("agent")
    kai_called = kai.get("kai_called", False)
    clean_text = kai.get("clean_text", payload.raw_text)
    warning = kai.get("warning")

    # -------------------------
    # 2. Capa 1 cognitiva
    # -------------------------
    layer1 = build_layer1_context(agent)
    mode = layer1.get("mode", "WORK")

    # -------------------------
    # 3. LIFE → memoria personal
    # -------------------------
    if mode == "LIFE":
        node = MemoryNode(
            content=clean_text,
            created_at=datetime.utcnow(),
        )
        db.add(node)
        db.commit()

        return {
            "mode": "LIFE",
            "agent": agent,
            "kai_called": kai_called,
            "warning": warning,
            "input": clean_text,
            "response": "📌 Información guardada en tu memoria personal.",
        }

    # -------------------------
    # 4. WORK → timeline clínico
    # -------------------------
    # En LAB, si no viene procedure_id, lo generamos
    procedure_id = getattr(payload, "procedure_id", None)

    if not procedure_id:
        procedure_id = str(uuid4())

    procedure_uuid = UUID(procedure_id)

    event = VoiceEvent(
        procedure_id=procedure_uuid,
        intent="CLINICAL_NOTE",
        raw_text=clean_text,
    )
    db.add(event)
    db.commit()
    db.refresh(event)

    # -------------------------
    # 5. LLM (solo WORK)
    # -------------------------
    llm_response = run_llm(
        text=clean_text,
        agent=agent,
        layer1_context=layer1,
    )

    return {
        "mode": "WORK",
        "agent": agent,
        "kai_called": kai_called,
        "warning": warning,
        "procedure_id": str(procedure_uuid),
        "timeline_event_id": str(event.id),
        "input": clean_text,
        "response": llm_response,
    }

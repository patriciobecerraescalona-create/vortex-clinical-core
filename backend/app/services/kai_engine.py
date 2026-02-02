from typing import Optional, Dict

from backend.app.services.agent_permissions import (
    resolve_agent,
    AgentPermissionError,
)

# =========================
# Configuración global
# =========================

KAI_NAME = "kai"

# Activadores explícitos (evita activación semántica)
KAI_WAKE_PHRASES = [
    "oye kai",
    "hola kai",
]

# Palabras clave → agentes
AGENT_KEYWORDS = {
    "soporte": "support",
    "support": "support",
    "medico": "medical",
    "médico": "medical",
    "auditor": "auditor",
    "comercial": "commercial",
}

# =========================
# Utilidades internas
# =========================

def _normalize(text: str) -> str:
    return text.lower().strip()


def _contains_wake_phrase(text: str) -> bool:
    return any(phrase in text for phrase in KAI_WAKE_PHRASES)


def _extract_requested_agent(text: str) -> Optional[str]:
    """
    Detecta frases tipo:
    - 'modo soporte'
    - 'modo medico'
    """
    for keyword, agent in AGENT_KEYWORDS.items():
        if f"modo {keyword}" in text:
            return agent
    return None


def _strip_wake_phrase(text: str) -> str:
    cleaned = text
    for phrase in KAI_WAKE_PHRASES:
        cleaned = cleaned.replace(phrase, "")
    return cleaned.strip()

# =========================
# API pública
# =========================

def detect_agent_switch(user_text: str) -> Optional[str]:
    """
    Función liviana usada por procedures.py.
    NO valida permisos.
    SOLO detecta si el texto pide cambio de agente.
    """
    text = _normalize(user_text)
    return _extract_requested_agent(text)


def process_kai_activation(
    user_text: str,
    user_context: Dict,
) -> Dict:
    """
    Procesa activación de KAI + permisos.

    user_context esperado:
    {
        "user_id": str,
        "user_role": str,
        "allowed_agents": list[str],
        "default_agent": str
    }
    """

    text = _normalize(user_text)

    kai_called = _contains_wake_phrase(text)
    requested_agent = _extract_requested_agent(text)

    try:
        final_agent = resolve_agent(
            requested_agent=requested_agent,
            allowed_agents=user_context["allowed_agents"],
            default_agent=user_context["default_agent"],
        )
        warning = None

    except AgentPermissionError as e:
        final_agent = user_context["default_agent"]
        warning = str(e)

    clean_text = _strip_wake_phrase(text)

    return {
        "kai_called": kai_called,
        "agent": final_agent,
        "warning": warning,
        "clean_text": clean_text,
    }

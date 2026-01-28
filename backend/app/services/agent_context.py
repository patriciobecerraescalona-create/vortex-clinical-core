"""
agent_context.py

Fuente única de verdad para:
- permisos por rol
- agentes disponibles
- capa 1 cognitiva por agente

El LLM NO decide nada aquí.
"""

# =========================
# Contexto por rol (SGMI)
# =========================

def get_user_context(role: str) -> dict:
    """
    Contexto de identidad y permisos.
    En producción vendrá desde SGMI.
    """

    ROLE_AGENT_MAP = {
        "secretary": {
            "allowed_agents": [
                "support",
                "commercial",
            ],
            "default_agent": "commercial",
        },

        "clinician": {
            "allowed_agents": [
                "medical",
                "support",
                "life",
                "commercial",
            ],
            "default_agent": "commercial",
        },

        "admin": {
            "allowed_agents": [
                "support",
                "auditor",
                "commercial",
            ],
            "default_agent": "commercial",
        },

        "manager": {
            "allowed_agents": [
                "medical",
                "support",
                "auditor",
                "life",
                "commercial",
            ],
            "default_agent": "commercial",
        },

        "anonymous": {
            "allowed_agents": [
                "commercial",
            ],
            "default_agent": "commercial",
        },
    }

    return ROLE_AGENT_MAP.get(
        role,
        ROLE_AGENT_MAP["anonymous"]
    )


# =========================
# Capa 1 cognitiva por agente
# =========================

def build_layer1_context(agent: str) -> dict:
    """
    Retorna el contexto base (capa 1) que SIEMPRE
    se entrega al agente antes de cualquier LLM.
    """

    AGENT_LAYER1 = {
        "medical": {
            "domain": "clinical",
            "mode": "WORK",
            "rules": [
                "No emitir diagnóstico definitivo",
                "Apoyo al razonamiento clínico",
                "Lenguaje profesional de salud",
                "Derivar a evaluación médica presencial",
            ],
        },

        "support": {
            "domain": "operational",
            "mode": "WORK",
            "rules": [
                "Soporte de sistemas",
                "No acceso a información clínica",
                "Responder de forma clara y operativa",
            ],
        },

        "auditor": {
            "domain": "compliance",
            "mode": "WORK",
            "rules": [
                "Auditoría y cumplimiento",
                "No modificar datos",
                "Enfoque normativo y trazable",
            ],
        },

        "commercial": {
            "domain": "general",
            "mode": "WORK",
            "rules": [
                "Información general",
                "Lenguaje neutro",
                "No asumir contexto clínico",
            ],
        },

        "life": {
            "domain": "personal",
            "mode": "LIFE",
            "rules": [
                "Memoria personal",
                "Recordatorios",
                "No mezclar con trabajo",
                "No acceder a datos clínicos",
            ],
        },
    }

    return AGENT_LAYER1.get(
        agent,
        AGENT_LAYER1["commercial"]
    )

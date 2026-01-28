"""
Definición de agentes Vortex.
Un agente = reglas + capa 1 + políticas.
"""

AGENTS = {
    "commercial": {
        "id": "commercial",
        "label": "Agente Comercial",
        "default": True,
        "allowed_domains": ["WORK"],
        "layer1_sources": [
            "products",
            "pricing",
            "client_context"
        ],
        "allow_life_memory": False,
        "allow_work_timeline": False,
        "llm_policy": {
            "allow_diagnosis": False,
            "allow_prescription": False,
            "style": "commercial_assistant"
        }
    },

    "medical": {
        "id": "medical",
        "label": "Agente Médico",
        "default": False,
        "allowed_domains": ["WORK"],
        "layer1_sources": [
            "clinical_timeline",
            "patient_context"
        ],
        "allow_life_memory": False,
        "allow_work_timeline": True,
        "llm_policy": {
            "allow_diagnosis": False,
            "allow_prescription": False,
            "style": "clinical_support"
        }
    },

    "support": {
        "id": "support",
        "label": "Agente Soporte",
        "default": False,
        "allowed_domains": ["WORK"],
        "layer1_sources": [
            "support_kb",
            "system_status",
            "known_issues"
        ],
        "allow_life_memory": False,
        "allow_work_timeline": False,
        "llm_policy": {
            "allow_diagnosis": False,
            "allow_prescription": False,
            "style": "technical_support"
        }
    },

    "observer": {
        "id": "observer",
        "label": "Agente Observador",
        "default": False,
        "allowed_domains": ["READ_ONLY"],
        "layer1_sources": [],
        "allow_life_memory": False,
        "allow_work_timeline": True,
        "llm_policy": {
            "allow_diagnosis": False,
            "allow_prescription": False,
            "style": "observer"
        }
    }
}


def get_default_agent_id() -> str:
    for agent in AGENTS.values():
        if agent.get("default"):
            return agent["id"]
    return "commercial"


def get_agent(agent_id: str):
    return AGENTS.get(agent_id) or AGENTS[get_default_agent_id()]

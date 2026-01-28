from enum import Enum
from typing import List, Dict
import re


class CognitiveSignal(str, Enum):
    OBSERVATION = "OBSERVATION"
    CLINICAL_ACTION = "CLINICAL_ACTION"
    REMINDER_INTENT = "REMINDER_INTENT"
    REFLECTION = "REFLECTION"
    DECISION_SEED = "DECISION_SEED"
    SOCIAL_CONTEXT = "SOCIAL_CONTEXT"


def detect_cognitive_signals(text: str) -> Dict:
    """
    Analiza el texto y devuelve señales cognitivas.
    NO decide modo vida/trabajo.
    NO usa LLM.
    """

    t = text.lower()
    signals: List[CognitiveSignal] = []

    # Recordatorios / memoria futura
    if re.search(r"(recu[eé]rdame|no se me olvide|acu[eé]rdame)", t):
        signals.append(CognitiveSignal.REMINDER_INTENT)

    # Solicitudes clínicas claras
    if re.search(r"(solicitar|indicar|ordenar).*(ex[aá]men|tratamiento|rx|radiograf)", t):
        signals.append(CognitiveSignal.CLINICAL_ACTION)

    # Síntomas / observaciones
    if re.search(r"(dolor|molestia|náuseas|cefalea|mareo|fiebre)", t):
        signals.append(CognitiveSignal.OBSERVATION)

    # Reflexión personal
    if re.search(r"(creo que|pienso que|me parece que)", t):
        signals.append(CognitiveSignal.REFLECTION)

    # Contexto social
    if re.search(r"(mi sobrino|mi papá|mi mamá|un amigo)", t):
        signals.append(CognitiveSignal.SOCIAL_CONTEXT)

    if not signals:
        signals.append(CognitiveSignal.OBSERVATION)

    return {
        "signals": [s.value for s in signals],
        "confidence": round(min(1.0, 0.4 + 0.15 * len(signals)), 2),
    }

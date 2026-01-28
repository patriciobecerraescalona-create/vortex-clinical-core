from enum import Enum


class CognitiveMode(str, Enum):
    MEMORY = "memory"
    WORK = "work"


def normalize(text: str) -> str:
    return text.lower().strip()


def detect_mode(text: str, context_active: bool) -> CognitiveMode:
    """
    Decide si el texto corresponde a:
    - memoria personal (vida)
    - contexto de trabajo (clínico)

    context_active:
      - True  -> estamos en procedimiento activo
      - False -> modo libre
    """

    t = normalize(text)

    # Palabras típicas de memoria / vida
    memory_triggers = [
        "me olvidé",
        "se me olvidó",
        "recuerda",
        "aniversario",
        "cumpleaños",
        "nota mental",
        "acuérdate",
    ]

    for trigger in memory_triggers:
        if trigger in t:
            return CognitiveMode.MEMORY

    # Si hay contexto activo, por defecto es trabajo
    if context_active:
        return CognitiveMode.WORK

    # Default conservador
    return CognitiveMode.WORK

from dataclasses import dataclass

RULES = [
    ("AUTHORITY", "Autoridad de la fuente"),
    ("OBLIGATION_EXPLICIT", "Obligación explícita"),
    ("CONSEQUENCE", "Consecuencia de incumplimiento"),
    ("STABILITY", "Estabilidad/Versionable"),
    ("AUDITABLE", "Auditabilidad"),
    ("APPLICABILITY", "Aplicabilidad general"),
    ("LEVEL_SEPARATION", "Separación de niveles")
]

@dataclass
class RuleResult:
    rule_code: str
    rule_name: str
    result: bool
    justification: str

def evaluate_document(meta: dict) -> list[RuleResult]:
    """
    MVP: evaluamos en base a:
    - authority_source (si viene vacío, falla)
    - document_type (si viene vacío, falla)
    - status recomendado por heurística simple
    Luego lo reforzamos con parsing real.
    """
    out: list[RuleResult] = []

    authority = (meta.get("authority_source") or "").strip()
    doc_type = (meta.get("document_type") or "").strip().lower()

    # Regla: autoridad
    out.append(RuleResult(
        "AUTHORITY", "Autoridad de la fuente",
        bool(authority),
        "Fuente definida" if authority else "Falta authority_source (MINSAL/BCN/ISP/Interno)."
    ))

    # Heurística simple por tipo
    is_normative = doc_type in {"ley", "norma", "reglamento", "protocolo", "politica interna", "estandar"}

    # Las otras reglas en MVP: asumimos verdaderas si es normativo, falsas si no.
    for code, name in RULES[1:]:
        if is_normative:
            out.append(RuleResult(code, name, True, "MVP: marcado como documento normativo por tipo."))
        else:
            out.append(RuleResult(code, name, False, "MVP: tipo no normativo; requiere revisión manual."))

    return out

def recommend_status(results: list[RuleResult]) -> str:
    passed = sum(1 for r in results if r.result)
    if passed >= 5:
        return "CORE_ACTIVE"
    if passed >= 3:
        return "EXTERNAL_REFERENCE"
    return "REJECTED"

def classify_intent(text: str) -> dict:
    """
    Clasificador básico de intención (MVP).
    Luego se reemplaza por reglas + LLM.
    """
    return {
        "intent": "CLINICAL_NOTE",
        "confidence": 0.5
    }

def classify_intent(text: str) -> dict:
    t = text.lower()

    if any(w in t for w in ["dolor", "náusea", "fiebre"]):
        return {"intent": "SYMPTOM", "confidence": 0.6}

    if any(w in t for w in ["recetar", "indicar", "prescribir"]):
        return {"intent": "TREATMENT_INTENT", "confidence": 0.7}

    if any(w in t for w in ["examen", "laboratorio", "imagen"]):
        return {"intent": "EXAM_REQUEST", "confidence": 0.65}

    return {"intent": "CLINICAL_NOTE", "confidence": 0.4}

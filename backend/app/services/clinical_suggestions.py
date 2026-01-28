from typing import List, Dict

def generate_suggestions(context: Dict) -> List[Dict]:
    suggestions = []

    # 1. Vacunas
    if context.get("patient_age") and context.get("vaccines_missing"):
        suggestions.append({
            "type": "PREVENTIVE_ALERT",
            "message": "Paciente presenta vacunas pendientes según edad.",
            "requires_confirmation": True
        })

    # 2. Antecedentes repetidos
    if context.get("recent_visits_same_symptom", 0) >= 3:
        suggestions.append({
            "type": "PATTERN_ALERT",
            "message": "Se detectan múltiples consultas recientes por síntomas similares.",
            "requires_confirmation": False
        })

    # 3. Fármaco desactualizado (ejemplo OMS)
    if context.get("medication") == "omeprazol":
        suggestions.append({
            "type": "PHARMA_UPDATE",
            "message": "Existen alternativas vigentes según guías recientes (ej. esomeprazol).",
            "requires_confirmation": True
        })

    return suggestions

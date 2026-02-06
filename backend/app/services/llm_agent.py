import time
import os
import httpx
from typing import Dict, Any, Optional

# =========================
# SYSTEM PROMPTS (ROLES)
# =========================

def _build_system_prompt(role: str, context: Dict[str, Any]) -> str:
    """Construye el Prompt del Sistema basado en Rol y Contexto SGMI."""
    
    # 1. Definición de Roles
    roles = {
        "clinical": (
            "Eres un ASISTENTE CLÍNICO EXPERTO (Vortex Core). "
            "Tu objetivo es apoyar el razonamiento médico con precisión técnica. "
            "Usa terminología médica avanzada. Sé conciso y directo. "
            "NO des disclaimers (ya estamos en entorno clínico). "
            "Asume que hablas con un médico."
        ),
        "administrative": (
            "Eres un ASISTENTE ADMINISTRATIVO de salud. "
            "Tu objetivo es optimizar procesos de gestión, códigos CIE-10/FONASA, y documentación. "
            "Sé formal, eficiente y orientado a procesos."
        ),
        "commercial": (
            "Eres un ASISTENTE COMERCIAL de la clínica. "
            "Tu objetivo es explicar valor, planes de salud y beneficios. "
            "Usa un tono persuasivo pero ético y profesional."
        ),
        "personal": (
            "Eres un ASISTENTE PERSONAL del médico (Vortex Life). "
            "Tu tono es cercano, empático y servicial, pero siempre profesional. "
            "Ayudas a equilibrar la carga laboral y el bienestar."
        ),
        "support": (
            "Eres SOPORTE TÉCNICO del sistema Vortex. "
            "Ayudas a resolver dudas sobre el uso de la plataforma. "
            "Sé didáctico y paciente."
        )
    }

    base_role = roles.get(role, roles["clinical"])

    # 2. Reglas de Comportamiento (Globales)
    global_rules = (
        "IDIOMA: RESPONDE SIEMPRE EN ESPAÑOL (Neutro/Chile). NUNCA en inglés ni portugués.\n"
        "TONO: Profesional, serio y objetivo. Sin saludos innecesarios.\n"
        "DISCLAIMERS: ELIMINADOS. No digas 'No soy médico'. Asume el rol asignado.\n"
        "FORMATO: Usa Markdown para estructurar la respuesta."
    )

    # 3. Inyección de Contexto SGMI
    # Formateamos el contexto para que el LLM lo "vea" como memoria inmutable
    ctx_str = ""
    if context:
        ctx_parts = []
        if context.get("patient_name") or context.get("patient_age"):
            p_data = f"PACIENTE: {context.get('patient_name', 'Anon')} | {context.get('patient_age', '?')} años | {context.get('patient_sex', '?')}"
            ctx_parts.append(p_data)
        
        if context.get("medical_history"):
            ctx_parts.append(f"ANTECEDENTES: {context['medical_history']}")
        
        if context.get("clinical_text"):
            ctx_parts.append(f"ANAMNESIS ACTUAL: {context['clinical_text']}")
            
        if ctx_parts:
            ctx_str = "\nCONTEXTO ACTUAL DEL PACIENTE:\n" + "\n".join(ctx_parts) + "\n"

    # 4. Prompt Final
    return f"{base_role}\n\nREGLAS:\n{global_rules}\n{ctx_str}\n\nINSTRUCCIÓN: Actúa según tu rol y el contexto proporcionado."


def run_llm(*, provider: str = "openai", **kwargs) -> Dict[str, Any]:
    """
    Ejecuta el LLM con soporte para Roles y Contexto SGMI.
    Kwargs: user_text, role, context (dict)
    """

    t0 = time.perf_counter()

    user_text = kwargs.get("user_text", "")
    role = kwargs.get("role", "clinical")
    context = kwargs.get("context", {}) or {}
    
    # Construir System Prompt
    system_prompt = _build_system_prompt(role, context)
    
    # -----------------------------
    # LAB / mock / deshabilitado
    # -----------------------------
    if provider in ("none", "mock"):
        return {
            "answer": f"[MOCK {role}] Respuesta simulada en español.",
            "tokens": 0,
            "provider": provider,
            "llm_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    # -----------------------------
    # CORE: Ollama (Active Agent)
    # -----------------------------
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
    timeout = float(os.getenv("OLLAMA_TIMEOUT", "60.0"))

    url = f"{ollama_url}/api/generate"
    
    # Combinamos System + User para modelos que no soportan system prompt explícito en generate
    # O usamos el formato de prompt de Ollama si el modelo lo permite.
    # Para máxima compatibilidad en raw text completion:
    final_prompt = f"SYSTEM: {system_prompt}\n\nUSER: {user_text}\n\nASSISTANT:"
    
    payload = {
        "model": ollama_model,
        "prompt": final_prompt,
        "stream": False,
        "options": {
            "temperature": 0.5, # Un poco más determinista para roles profesionales
            "num_predict": 1000,
        }
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            
        answer = data.get("response", "")
        eval_count = data.get("eval_count", 0)

    except Exception as e:
        answer = f"Error cognitivo: {str(e)}"
        eval_count = 0

    return {
        "answer": answer,
        "tokens": eval_count,
        "provider": "ollama",
        "llm_ms": round((time.perf_counter() - t0) * 1000, 2),
    }

"""
LLM Agent de Vortex
Contrato estable entre el sistema cognitivo y el proveedor LLM.
Este módulo NO debe romper aunque cambien los argumentos.
"""

import time
from typing import Dict, Any


def run_llm(*, provider: str = "openai", **kwargs) -> Dict[str, Any]:
    """
    Ejecuta el LLM configurado.

    Acepta argumentos abiertos (**kwargs) para evitar roturas
    cuando el flujo cognitivo evoluciona.

    kwargs puede incluir:
    - user_text
    - prompt
    - layer1_context
    - agent_id
    - mode
    - intent
    - confidence
    - etc.
    """

    t0 = time.perf_counter()

    user_text = kwargs.get("user_text")
    prompt = kwargs.get("prompt")

    base_text = prompt or user_text or ""

    # -----------------------------
    # LAB / mock / deshabilitado
    # -----------------------------
    if provider in ("none", "mock"):
        return {
            "answer": "[LLM deshabilitado]",
            "tokens": 0,
            "provider": provider,
            "llm_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    # -----------------------------
    # OPENAI (placeholder seguro)
    # -----------------------------
    answer = (
        "Respuesta generada por el motor cognitivo como apoyo al razonamiento.\n"
        "No constituye diagnóstico ni tratamiento.\n\n"
        f"Entrada procesada: {base_text}"
    )

    return {
        "answer": answer,
        "tokens": len(answer.split()),
        "provider": provider,
        "llm_ms": round((time.perf_counter() - t0) * 1000, 2),
    }

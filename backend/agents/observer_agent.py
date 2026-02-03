"""
ObserverAgent - Agente observador para análisis cognitivo clínico.

MODO EXPERIMENTAL: LLM Libre
- Sin restricciones clínicas
- Captura estructurada de razonamiento
- Logging de comportamiento cognitivo
- Objetivo: mapear capacidades reales del LLM
"""

import json
import os
import time
import re
import httpx
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

# =========================
# CONFIGURACIÓN OLLAMA
# =========================
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://192.168.1.8:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "30.0"))  # 30s max, objetivo <4s

# =========================
# PROMPT OBSERVADOR CLÍNICO DEFINITIVO
# =========================
PROMPT_OBSERVER = """NUNCA repitas información del contexto. El médico ya la conoce.
Tu rol comienza donde termina la anamnesis.

Eres OBSERVADOR DE CONTRASTE clínico senior.
NO resumes. NO confirmas obviedades. NO prescribes.

PROHIBIDO usar frases como:
- "Paciente con..."
- "Se observa..."
- "Cuadro compatible con..."
- "Según la anamnesis..."

ESTRUCTURA EXACTA (JSON):
{
  "high_impact": [{"scenario": "dx", "rationale": "por qué considerar"}],
  "alternatives": [{"scenario": "dx", "rationale": "si evolución atípica"}],
  "discriminators": [{"test": "estudio", "differentiates": "qué distingue"}],
  "management_paths": [{"path": "conducta", "when": "en qué escenario"}],
  "pivot_triggers": ["hallazgo que obliga a re-priorizar"]
}

REGLAS:
- Máximo 5 items por categoría
- Frases de máximo 15 palabras
- Solo información NO OBVIA
- Lenguaje de pase de guardia

Si contexto insuficiente: {"insufficient": true, "missing": ["qué falta"]}"""

# Patrones prohibidos en respuesta (redundancia)
FORBIDDEN_PATTERNS = [
    r"^paciente (con|de|que|presenta)",
    r"^se (observa|evidencia|aprecia|presenta)",
    r"^cuadro (compatible|sugestivo|clínico)",
    r"^según (la anamnesis|el examen|los datos)",
    r"^el paciente",
    r"^la paciente",
    r"^presenta",
    r"^refiere",
]

# =========================
# PATRONES DE COMPORTAMIENTO COGNITIVO
# =========================
COGNITIVE_PATTERNS = {
    "uncertainty": [
        r"no estoy seguro",
        r"podría ser",
        r"es posible que",
        r"no tengo certeza",
        r"difícil determinar",
        r"insuficiente información",
        r"requiere más datos",
        r"no puedo confirmar",
        r"possibly",
        r"uncertain",
        r"maybe",
    ],
    "contradiction": [
        r"sin embargo.*pero",
        r"aunque.*no obstante",
        r"por un lado.*por otro",
        r"contradictorio",
    ],
    "overgeneralization": [
        r"siempre",
        r"nunca",
        r"todos los pacientes",
        r"en todos los casos",
        r"invariablemente",
        r"sin excepción",
    ],
    "fabrication_markers": [
        r"según estudios recientes",
        r"investigaciones demuestran",
        r"está comprobado que",
        r"la literatura indica",
        r"estudios han mostrado",
    ],
}


class CognitiveLogger:
    """Registra comportamiento cognitivo del LLM para análisis."""

    def __init__(self):
        self.logs: List[Dict[str, Any]] = []

    def analyze_response(self, response_text: str, context: dict) -> Dict[str, Any]:
        """Analiza la respuesta buscando patrones cognitivos."""
        findings = {
            "timestamp": datetime.now().isoformat(),
            "uncertainty_markers": [],
            "contradiction_markers": [],
            "overgeneralization_markers": [],
            "fabrication_markers": [],
            "confidence_assessment": "unknown",
        }

        text_lower = response_text.lower()

        # Buscar patrones
        for pattern in COGNITIVE_PATTERNS["uncertainty"]:
            matches = re.findall(pattern, text_lower)
            if matches:
                findings["uncertainty_markers"].extend(matches)

        for pattern in COGNITIVE_PATTERNS["contradiction"]:
            matches = re.findall(pattern, text_lower)
            if matches:
                findings["contradiction_markers"].extend(matches)

        for pattern in COGNITIVE_PATTERNS["overgeneralization"]:
            matches = re.findall(pattern, text_lower)
            if matches:
                findings["overgeneralization_markers"].extend(matches)

        for pattern in COGNITIVE_PATTERNS["fabrication_markers"]:
            matches = re.findall(pattern, text_lower)
            if matches:
                findings["fabrication_markers"].extend(matches)

        # Evaluar confianza general
        uncertainty_count = len(findings["uncertainty_markers"])
        if uncertainty_count == 0:
            findings["confidence_assessment"] = "high_confidence"
        elif uncertainty_count <= 2:
            findings["confidence_assessment"] = "moderate_confidence"
        else:
            findings["confidence_assessment"] = "low_confidence"

        # Agregar al log
        log_entry = {
            **findings,
            "context_hash": hash(json.dumps(context, sort_keys=True)),
            "response_length": len(response_text),
        }
        self.logs.append(log_entry)

        return findings

    def get_summary(self) -> Dict[str, Any]:
        """Resumen del comportamiento cognitivo observado."""
        if not self.logs:
            return {"total_analyses": 0}

        return {
            "total_analyses": len(self.logs),
            "avg_uncertainty_markers": sum(len(l["uncertainty_markers"]) for l in self.logs) / len(self.logs),
            "total_fabrication_flags": sum(len(l["fabrication_markers"]) for l in self.logs),
            "confidence_distribution": {
                "high": sum(1 for l in self.logs if l["confidence_assessment"] == "high_confidence"),
                "moderate": sum(1 for l in self.logs if l["confidence_assessment"] == "moderate_confidence"),
                "low": sum(1 for l in self.logs if l["confidence_assessment"] == "low_confidence"),
            }
        }


class ObserverAgent:
    """
    Agente observador para análisis cognitivo clínico.
    MODO EXPERIMENTAL: Sin restricciones, captura estructurada.
    """

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        timeout: float = OLLAMA_TIMEOUT,
    ):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.cognitive_logger = CognitiveLogger()

    def analyze(self, patient_context: dict) -> dict:
        """
        Contraste clínico: amplía el razonamiento del médico.
        Solo se activa con contexto suficiente (anamnesis + impresión).
        """
        # Verificar contexto suficiente
        if not self._has_sufficient_context(patient_context):
            missing = []
            if not patient_context.get("clinical_text", "").strip():
                missing.append("Anamnesis")
            if not patient_context.get("clinical_impression", "").strip():
                missing.append("Impresión clínica inicial")
            return {
                "insufficient": True,
                "missing": missing,
                "metrics": {"response_time_ms": 0, "eval_count": 0},
                "visual_indicator": "gray",
                "llm_status": "waiting",
                "mode": "observer",
            }

        # Preparar contexto
        context_text = self._format_context(patient_context)

        # Llamar a Ollama
        try:
            raw_response, metrics = self._call_ollama(context_text)

            # Validar respuesta (sin patrones prohibidos)
            is_valid, validation_error = self._validate_response(raw_response)
            if not is_valid:
                result = {
                    "validation_error": validation_error,
                    "high_impact": [],
                    "alternatives": [],
                    "discriminators": [],
                    "management_paths": [],
                    "pivot_triggers": [],
                    "visual_indicator": "yellow",
                }
            else:
                result = self._parse_observer_response(raw_response)

            # Análisis cognitivo
            cognitive_analysis = self.cognitive_logger.analyze_response(
                raw_response, patient_context
            )
            result["cognitive_behavior"] = cognitive_analysis
            result["metrics"] = metrics
            result["llm_status"] = "connected"
            result["mode"] = "observer"

        except httpx.ConnectError:
            result = self._error_response(
                "disconnected",
                f"No se pudo conectar a Ollama en {self.base_url}"
            )
        except httpx.TimeoutException:
            result = self._error_response(
                "timeout",
                "El modelo tardó demasiado en responder."
            )
        except Exception as e:
            result = self._error_response("error", str(e))

        result["clinical_phase"] = patient_context.get("clinical_phase", "experimental")
        return result

    def _error_response(self, status: str, message: str) -> dict:
        """Genera respuesta de error estructurada."""
        return {
            "high_impact": [],
            "alternatives": [],
            "discriminators": [],
            "management_paths": [],
            "pivot_triggers": [],
            "metrics": {"response_time_ms": 0, "eval_count": 0},
            "cognitive_behavior": {},
            "visual_indicator": "gray",
            "llm_status": status,
            "llm_error": message,
            "mode": "observer",
        }

    def _format_context(self, patient_context: dict) -> str:
        """Formatea el contexto del paciente para contraste."""
        parts = []

        # Datos básicos (compacto)
        basic = []
        if patient_context.get("age"):
            basic.append(f"{patient_context['age']}a")
        if patient_context.get("sex"):
            basic.append(patient_context['sex'])
        if basic:
            parts.append(" ".join(basic))

        if patient_context.get("medical_history"):
            parts.append(f"APP: {patient_context['medical_history']}")

        if patient_context.get("reason_for_visit"):
            parts.append(f"MC: {patient_context['reason_for_visit']}")

        if patient_context.get("clinical_text"):
            parts.append(f"Anamnesis: {patient_context['clinical_text']}")

        # CLAVE: Impresión clínica del médico
        if patient_context.get("clinical_impression"):
            parts.append(f"IMPRESIÓN DEL COLEGA: {patient_context['clinical_impression']}")

        return "\n".join(parts) if parts else "Sin información."

    def _has_sufficient_context(self, patient_context: dict) -> bool:
        """Verifica si hay contexto suficiente para activar contraste."""
        # Requiere: anamnesis + impresión clínica inicial
        has_clinical_text = bool(patient_context.get("clinical_text", "").strip())
        has_impression = bool(patient_context.get("clinical_impression", "").strip())
        return has_clinical_text and has_impression

    def _call_ollama(self, context_text: str) -> Tuple[str, dict]:
        """Llama a Ollama API. Retorna (response, metrics)."""
        url = f"{self.base_url}/api/generate"

        prompt = f"{PROMPT_OBSERVER}\n\n---\n{context_text}\n---\n\nJSON:"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "num_predict": 400,  # Límite ~400 tokens
                "temperature": 0.3,  # Más determinista
            }
        }

        start_time = time.time()
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        elapsed = time.time() - start_time
        metrics = {
            "response_time_ms": int(elapsed * 1000),
            "eval_count": data.get("eval_count", 0),
            "prompt_eval_count": data.get("prompt_eval_count", 0),
        }

        return data.get("response", "{}"), metrics

    def _validate_response(self, response_text: str) -> Tuple[bool, str]:
        """Valida que la respuesta no contenga patrones prohibidos."""
        text_lower = response_text.lower()
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, text_lower):
                return False, f"Patrón prohibido: {pattern}"
        return True, ""

    def _normalize_list(self, value) -> list:
        """Normaliza un valor a lista."""
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [value] if value.strip() else []
        return [str(value)]

    def _parse_observer_response(self, response_text: str) -> dict:
        """Parsea respuesta del observador clínico."""
        try:
            result = json.loads(response_text)

            # Check if insufficient context
            if result.get("insufficient"):
                return {
                    "insufficient": True,
                    "missing": self._normalize_list(result.get("missing", [])),
                    "visual_indicator": "gray",
                }

            return {
                "high_impact": self._normalize_list(result.get("high_impact", [])),
                "alternatives": self._normalize_list(result.get("alternatives", [])),
                "discriminators": self._normalize_list(result.get("discriminators", [])),
                "management_paths": self._normalize_list(result.get("management_paths", [])),
                "pivot_triggers": self._normalize_list(result.get("pivot_triggers", [])),
                "visual_indicator": self._determine_observer_indicator(result),
            }
        except json.JSONDecodeError:
            return {
                "high_impact": [],
                "alternatives": [],
                "discriminators": [],
                "management_paths": [],
                "pivot_triggers": [],
                "visual_indicator": "yellow",
                "parse_error": response_text[:300] if response_text else "Sin respuesta",
            }

    def _determine_observer_indicator(self, result: dict) -> str:
        """Determina indicador visual basado en análisis del observador."""
        high_impact = result.get("high_impact", [])
        pivot_triggers = result.get("pivot_triggers", [])

        if len(high_impact) > 0:
            return "red"  # Rojo = escenarios de alto impacto a descartar
        elif len(pivot_triggers) > 0:
            return "yellow"  # Amarillo = hay triggers a vigilar
        return "green"  # Verde = sin alertas críticas

    def get_cognitive_summary(self) -> Dict[str, Any]:
        """Obtiene resumen del comportamiento cognitivo."""
        return self.cognitive_logger.get_summary()


# =========================
# THROTTLING HELPER
# =========================
class ThrottledObserver:
    """Wrapper con throttling para el ObserverAgent."""

    def __init__(
        self,
        observer: Optional[ObserverAgent] = None,
        min_interval: float = 2.0,
        min_chars_change: int = 20,
    ):
        self.observer = observer or ObserverAgent()
        self.min_interval = min_interval
        self.min_chars_change = min_chars_change
        self._last_analysis_time: float = 0
        self._last_context_str: str = ""
        self._cached_result: Optional[Dict[str, Any]] = None

    def analyze(self, patient_context: dict, force: bool = False) -> dict:
        """Analiza con throttling."""
        current_time = time.time()
        context_str = json.dumps(patient_context, sort_keys=True)

        time_elapsed = current_time - self._last_analysis_time
        chars_changed = abs(len(context_str) - len(self._last_context_str))

        should_execute = force or (
            time_elapsed >= self.min_interval or chars_changed >= self.min_chars_change
        )

        if should_execute and (context_str != self._last_context_str or force):
            self._cached_result = self.observer.analyze(patient_context)
            self._last_analysis_time = current_time
            self._last_context_str = context_str

        return self._cached_result or {
            "insufficient": True,
            "missing": ["Esperando contexto clínico"],
            "high_impact": [],
            "alternatives": [],
            "discriminators": [],
            "management_paths": [],
            "pivot_triggers": [],
            "metrics": {"response_time_ms": 0, "eval_count": 0},
            "visual_indicator": "gray",
            "mode": "observer",
        }

    def get_cognitive_summary(self) -> Dict[str, Any]:
        """Obtiene resumen cognitivo del observer interno."""
        return self.observer.get_cognitive_summary()


# =========================
# INSTANCIA GLOBAL
# =========================
_default_observer: Optional[ThrottledObserver] = None


def get_observer() -> ThrottledObserver:
    """Obtiene la instancia global del observer."""
    global _default_observer
    if _default_observer is None:
        _default_observer = ThrottledObserver()
    return _default_observer

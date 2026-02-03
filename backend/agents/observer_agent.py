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
from typing import Dict, Any, Optional, List

# =========================
# CONFIGURACIÓN OLLAMA
# =========================
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://192.168.1.8:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120.0"))

# =========================
# PROMPT CLÍNICO SENIOR
# =========================
PROMPT_CLINICAL = """Eres un internista senior en discusión de caso con colegas.
NO repitas la información del caso. Aporta criterio clínico.

ESTRUCTURA DE RESPUESTA:

1. ESCENARIOS CLÍNICOS (ordenados por urgencia)
   - CRÍTICO (rojo): Condiciones que requieren acción inmediata
   - INTERMEDIO (amarillo): Requieren evaluación prioritaria
   - MANEJABLE (verde): Abordaje ambulatorio razonable

2. HALLAZGOS DE ALERTA
   - Signos/síntomas que modifican la conducta
   - Combinaciones semiológicas de riesgo

3. ORIENTACIÓN DIAGNÓSTICA
   - Estudios clave (no exhaustivos)
   - Qué buscar en cada uno
   - Secuencia lógica

4. REFERENCIAS
   - Guías o criterios clínicos relevantes (nombre corto)
   - Sin URLs

REGLAS:
- Frases cortas, lenguaje de pase de visita
- Prioriza lo que cambia conducta
- No describas lo obvio
- Si faltan datos críticos, dilo primero

Responde en JSON:
{
  "scenarios": {
    "critical": [{"condition": "nombre", "why": "razón breve", "action": "conducta"}],
    "intermediate": [{"condition": "nombre", "why": "razón breve", "action": "conducta"}],
    "manageable": [{"condition": "nombre", "why": "razón breve", "action": "conducta"}]
  },
  "alert_findings": ["hallazgo que modifica conducta"],
  "workup": [{"study": "nombre", "target": "qué buscar"}],
  "clinical_keys": ["perla clínica o criterio importante"],
  "references": ["Guía/Criterio relevante"],
  "data_gaps": ["dato crítico faltante"]
}

CONTEXTO: Investigación cognitiva LLM. No es consejo médico real."""

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
        Análisis clínico experimental sin restricciones.
        Captura razonamiento completo del LLM.
        """
        # Preparar contexto
        context_text = self._format_context(patient_context)

        # Llamar a Ollama
        try:
            raw_response = self._call_ollama(context_text)
            result = self._parse_clinical_response(raw_response)

            # Análisis cognitivo
            cognitive_analysis = self.cognitive_logger.analyze_response(
                raw_response, patient_context
            )
            result["cognitive_behavior"] = cognitive_analysis
            result["llm_status"] = "connected"
            result["mode"] = "clinical"
            result["raw_response"] = raw_response[:1000]  # Guardar muestra

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
            "scenarios": {"critical": [], "intermediate": [], "manageable": []},
            "alert_findings": [],
            "workup": [],
            "clinical_keys": [],
            "references": [],
            "data_gaps": [],
            "cognitive_behavior": {},
            "visual_indicator": "gray",
            "llm_status": status,
            "llm_error": message,
            "mode": "clinical",
        }

    def _format_context(self, patient_context: dict) -> str:
        """Formatea el contexto del paciente."""
        parts = []

        if patient_context.get("patient_name"):
            parts.append(f"Paciente: {patient_context['patient_name']}")

        if patient_context.get("age"):
            parts.append(f"Edad: {patient_context['age']} años")

        if patient_context.get("sex"):
            sex_map = {"M": "Masculino", "F": "Femenino"}
            parts.append(f"Sexo: {sex_map.get(patient_context['sex'], patient_context['sex'])}")

        if patient_context.get("medical_history"):
            parts.append(f"Antecedentes médicos: {patient_context['medical_history']}")

        if patient_context.get("socio_cultural"):
            parts.append(f"Contexto socio-cultural: {patient_context['socio_cultural']}")

        if patient_context.get("reason_for_visit"):
            parts.append(f"Motivo de consulta: {patient_context['reason_for_visit']}")

        if patient_context.get("clinical_text"):
            parts.append(f"Descripción clínica: {patient_context['clinical_text']}")

        return "\n".join(parts) if parts else "Sin información disponible."

    def _call_ollama(self, context_text: str) -> str:
        """Llama a Ollama API."""
        url = f"{self.base_url}/api/generate"

        prompt = f"{PROMPT_CLINICAL}\n\n--- CASO ---\n{context_text}\n---\n\nJSON:"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "{}")

    def _normalize_list(self, value) -> list:
        """Normaliza un valor a lista."""
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [value] if value.strip() else []
        return [str(value)]

    def _parse_clinical_response(self, response_text: str) -> dict:
        """Parsea respuesta clínica jerarquizada."""
        try:
            result = json.loads(response_text)
            scenarios = result.get("scenarios", {})

            return {
                "scenarios": {
                    "critical": self._normalize_list(scenarios.get("critical", [])),
                    "intermediate": self._normalize_list(scenarios.get("intermediate", [])),
                    "manageable": self._normalize_list(scenarios.get("manageable", [])),
                },
                "alert_findings": self._normalize_list(result.get("alert_findings", [])),
                "workup": self._normalize_list(result.get("workup", [])),
                "clinical_keys": self._normalize_list(result.get("clinical_keys", [])),
                "references": self._normalize_list(result.get("references", [])),
                "data_gaps": self._normalize_list(result.get("data_gaps", [])),
                "visual_indicator": self._determine_indicator_v2(scenarios),
            }
        except json.JSONDecodeError:
            return {
                "scenarios": {"critical": [], "intermediate": [], "manageable": []},
                "alert_findings": [],
                "workup": [],
                "clinical_keys": [],
                "references": [],
                "data_gaps": ["Respuesta no estructurada"],
                "visual_indicator": "yellow",
                "parse_error": response_text[:300] if response_text else "Sin respuesta",
            }

    def _determine_indicator_v2(self, scenarios: dict) -> str:
        """Determina indicador visual basado en escenarios clínicos."""
        critical = scenarios.get("critical", [])
        intermediate = scenarios.get("intermediate", [])

        if critical and len(critical) > 0:
            return "red"
        elif intermediate and len(intermediate) > 0:
            return "yellow"
        return "green"

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
            "scenarios": {"critical": [], "intermediate": [], "manageable": []},
            "alert_findings": [],
            "workup": [],
            "clinical_keys": [],
            "references": [],
            "data_gaps": [],
            "visual_indicator": "gray",
            "mode": "clinical",
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

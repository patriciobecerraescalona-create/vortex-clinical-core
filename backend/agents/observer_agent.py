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
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "60.0"))  # 60s LAB, objetivo <4s


def safe_str(value) -> str:
    """Convierte cualquier valor a string seguro (nunca None)."""
    if value is None:
        return ""
    return str(value).strip()

# =========================
# PROMPT OBSERVADOR CLÍNICO DEFINITIVO
# =========================
PROMPT_OBSERVER = """Nunca resumas ni repitas información clínica ya presente en el contexto.
Asume que el médico ya comprende completamente el caso.
Tu rol comienza después de finalizada la anamnesis y el examen físico.
Si no puedes aportar información clínica adicional relevante, responde:
{"no_additional": true}

Eres OBSERVADOR DE CONTRASTE CLÍNICO senior.
No realizas anamnesis, no resumes casos, no confirmas lo evidente.
Tu función es ampliar el razonamiento del médico tratante y reducir sesgos.

PROHIBIDO:
- Resumir el caso
- Repetir información explícita del contexto
- Validar lo obvio ("cuadro gastrointestinal", "dolor abdominal")
- Prescribir tratamientos
- Emitir diagnósticos definitivos
- Usar frases como "Paciente con...", "Se observa...", "Cuadro compatible..."

ESTRUCTURA JSON:
{
  "high_impact": [{"scenario": "dx", "rationale": "justificación breve"}],
  "alternatives": [{"scenario": "dx", "rationale": "si evolución atípica"}],
  "discriminators": [{"test": "estudio", "differentiates": "qué información aporta"}],
  "management_paths": [{"path": "conducta general", "when": "en qué escenario"}],
  "pivot_triggers": ["hallazgo que obliga a re-priorizar"]
}

REGLAS:
- Máximo 5 items por categoría
- Frases breves, nivel médico senior
- Solo información NO OBVIA que amplíe el razonamiento
- Prioriza síntesis y jerarquía, no volumen

Objetivo: Que el médico piense "Esto no lo había considerado y es clínicamente relevante"."""

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
        Solo se activa con contexto suficiente (anamnesis).
        """
        # Sanitizar entrada
        if not patient_context or not isinstance(patient_context, dict):
            patient_context = {}

        # Verificar contexto suficiente
        if not self._has_sufficient_context(patient_context):
            missing = []
            if not safe_str(patient_context.get("clinical_text")):
                missing.append("Anamnesis")
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
            # llm_status ya viene del parser - solo asegurar que existe
            if "llm_status" not in result:
                result["llm_status"] = "ok"
            result["mode"] = "observer"

        except httpx.ConnectError:
            result = self._error_response(
                "error",
                f"Sin conexión a Ollama ({self.base_url})"
            )
        except httpx.TimeoutException:
            result = self._error_response(
                "error",
                "Timeout del modelo"
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
        if not patient_context:
            return "Sin información."

        parts = []

        # Datos básicos (compacto)
        basic = []
        age = patient_context.get("age")
        sex = safe_str(patient_context.get("sex"))
        if age:
            basic.append(f"{age}a")
        if sex:
            basic.append(sex)
        if basic:
            parts.append(" ".join(basic))

        history = safe_str(patient_context.get("medical_history"))
        if history:
            parts.append(f"APP: {history}")

        reason = safe_str(patient_context.get("reason_for_visit"))
        if reason:
            parts.append(f"MC: {reason}")

        clinical = safe_str(patient_context.get("clinical_text"))
        if clinical:
            parts.append(f"Anamnesis: {clinical}")

        return "\n".join(parts) if parts else "Sin información."

    def _has_sufficient_context(self, patient_context: dict) -> bool:
        """Verifica si hay contexto suficiente para activar contraste."""
        # Requiere: solo anamnesis
        has_clinical_text = bool(safe_str(patient_context.get("clinical_text")))
        return has_clinical_text

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
            "model_name": self.model,
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
        """Normaliza un valor a lista (seguro para None)."""
        if value is None:
            return []
        if isinstance(value, list):
            # Filtrar None de la lista
            return [v for v in value if v is not None]
        if isinstance(value, str):
            return [value] if value.strip() else []
        return [str(value)] if value else []

    def _parse_observer_response(self, response_text: str) -> dict:
        """Parsea respuesta del observador clínico."""
        try:
            result = json.loads(response_text)

            # Sin aporte adicional del LLM
            if result.get("no_additional"):
                return {
                    "no_additional": True,
                    "llm_status": "ok",  # LLM respondió, pero sin aporte
                    "visual_indicator": "green",
                }

            # Check if insufficient context (respuesta del LLM)
            if result.get("insufficient"):
                return {
                    "insufficient": True,
                    "missing": self._normalize_list(result.get("missing", [])),
                    "llm_status": "ok",  # LLM respondió
                    "visual_indicator": "gray",
                }

            # Respuesta con contenido clínico
            high_impact = self._normalize_list(result.get("high_impact", []))
            alternatives = self._normalize_list(result.get("alternatives", []))
            discriminators = self._normalize_list(result.get("discriminators", []))
            management_paths = self._normalize_list(result.get("management_paths", []))
            pivot_triggers = self._normalize_list(result.get("pivot_triggers", []))

            # Determinar si hay contenido útil
            has_content = any([high_impact, alternatives, discriminators, management_paths, pivot_triggers])

            return {
                "high_impact": high_impact,
                "alternatives": alternatives,
                "discriminators": discriminators,
                "management_paths": management_paths,
                "pivot_triggers": pivot_triggers,
                "llm_status": "ok" if has_content else "ok",  # LLM respondió
                "visual_indicator": self._determine_observer_indicator(result),
            }
        except json.JSONDecodeError:
            return {
                "high_impact": [],
                "alternatives": [],
                "discriminators": [],
                "management_paths": [],
                "pivot_triggers": [],
                "llm_status": "error",  # LLM falló, no hay texto útil
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
_warmup_done: bool = False


def get_observer() -> ThrottledObserver:
    """Obtiene la instancia global del observer."""
    global _default_observer
    if _default_observer is None:
        _default_observer = ThrottledObserver()
    return _default_observer


def warmup_ollama() -> dict:
    """
    Calienta el modelo Ollama con una consulta simple.
    Llamar al iniciar el backend para reducir latencia del primer request real.
    """
    global _warmup_done
    if _warmup_done:
        return {"status": "already_warm"}

    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": "Responde OK",
        "stream": False,
        "options": {"num_predict": 5}
    }

    try:
        start = time.time()
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
        elapsed = time.time() - start
        _warmup_done = True
        print(f"[WARMUP] Ollama {OLLAMA_MODEL} listo en {elapsed:.1f}s")
        return {"status": "ok", "time_s": round(elapsed, 1)}
    except Exception as e:
        print(f"[WARMUP] Error: {e}")
        return {"status": "error", "error": str(e)}


def get_warmup_status() -> bool:
    """Retorna si el warmup de Ollama ha finalizado."""
    return _warmup_done

import time
import os
import httpx
from typing import Dict, Any, Optional

# =========================
# SYSTEM PROMPTS (ROLES)
# =========================
from backend.app.services.prompts_service import PromptsService
from backend.app.db.session import SessionLocal

def _build_context_string(context: Dict[str, Any]) -> str:
    """Inyección de Contexto SGMI (Datos del Paciente)."""
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
    return ctx_str

def run_llm(*, provider: str = "openai", **kwargs) -> Dict[str, Any]:
    """
    Ejecuta el LLM con soporte para Roles, Contexto SGMI y Multi-Provider (LAB).
    Ahora usa Prompts desde BD (PromptsService).
    """

    t0 = time.perf_counter()

    user_text = kwargs.get("user_text", "")
    role = kwargs.get("role", "clinical")
    context = kwargs.get("context", {}) or {}
    
    # Lab parameters
    model = kwargs.get("model", "")
    raw = kwargs.get("raw", False)
    mode = kwargs.get("mode", "work") 
    
    # Cost defaults
    cost_inference = 0.0
    cost_infra = 0.0001 
    
    # 1. System Prompt Construction
    # RAW mode check: Only allowed for 'clinical' role
    use_raw = raw and (role == "clinical")
    
    # Build System Messages
    system_messages = []
    
    # DB Access for Prompts
    try:
        with SessionLocal() as db:
            system_prompts = PromptsService.get_system_prompts(db, role, mode, use_raw)
            for p in system_prompts:
                system_messages.append({"role": "system", "content": p})
    except Exception as e:
        # Fallback de emergencia si DB falla
        print(f"Error fetching prompts: {e}")
        system_messages.append({"role": "system", "content": f"You are a {role} assistant."})

    # Add Context
    ctx_str = _build_context_string(context)
    if ctx_str:
        system_messages.append({"role": "system", "content": ctx_str})

    # Si es RAW y no es clinical (warning already handled in UI, but clean here)
    # logic is handled in PromptsService regarding what to include.

    # -----------------------------
    # MOCK / NONE
    # -----------------------------
    if provider in ("none", "mock"):
        return {
            "answer": f"[MOCK {role}] Respuesta simulada en español. (RAW={use_raw})",
            "tokens_prompt": 0,
            "tokens_completion": 0,
            "tokens_total": 0,
            "provider": provider,
            "llm_ms": round((time.perf_counter() - t0) * 1000, 2),
            "cost_inference": 0.0,
            "cost_infra": 0.0,
            "cost_total": 0.0,
        }

    # -----------------------------
    # OPENAI (LAB ONLY)
    # -----------------------------
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {
                "answer": "Error: OPENAI_API_KEY no configurada en entorno.",
                "tokens_total": 0,
                "provider": "openai",
                "llm_ms": 0,
                "cost_total": 0
            }
            
        if not model:
            model = "gpt-3.5-turbo"

        url = "https://api.openai.com/v1/chat/completions"
        
        messages = []
        # Inyectamos prompts (DB + Context)
        messages.extend(system_messages)
        messages.append({"role": "user", "content": user_text})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.5 if not use_raw else 0.7,
        }
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                
            choice = data["choices"][0]
            answer = choice["message"]["content"]
            usage = data.get("usage", {})
            
            t_prompt = usage.get("prompt_tokens", 0)
            t_completion = usage.get("completion_tokens", 0)
            t_total = usage.get("total_tokens", 0)
            
            # Cost estimation (approx)
            # gpt-3.5-turbo: $0.50 / $1.50
            price_in = 0.50 / 1_000_000
            price_out = 1.50 / 1_000_000
            
            if "gpt-4o" in model:
                # gpt-4o: $5.00 / $15.00
                price_in = 5.00 / 1_000_000
                price_out = 15.00 / 1_000_000
            elif "gpt-4-turbo" in model:
                # gpt-4-turbo: $10.00 / $30.00
                price_in = 10.00 / 1_000_000
                price_out = 30.00 / 1_000_000
            elif "gpt-4" in model:
                 # gpt-4 (original): $30.00 / $60.00
                price_in = 30.00 / 1_000_000
                price_out = 60.00 / 1_000_000
            
            cost_inference = (t_prompt * price_in) + (t_completion * price_out)

        except Exception as e:
            return {
                "answer": f"Error OpenAI: {str(e)}",
                "provider": "openai",
                "llm_ms": round((time.perf_counter() - t0) * 1000, 2),
                "cost_total": 0
            }

        return {
            "answer": answer,
            "tokens_prompt": t_prompt,
            "tokens_completion": t_completion,
            "tokens_total": t_total,
            "provider": f"openai/{model}",
            "real_model": model,
            "llm_ms": round((time.perf_counter() - t0) * 1000, 2),
            "cost_inference": cost_inference,
            "cost_infra": cost_infra,
            "cost_total": cost_inference + cost_infra,
            "raw_mode": use_raw
        }

    # -----------------------------
    # OLLAMA (Local)
    # -----------------------------
    # -----------------------------
    # OLLAMA (Local)
    # -----------------------------
    # Default fallback
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    # Si viene modelo explícito, usarlo, sino el del env, sino llama3
    ollama_model = model if model else os.getenv("OLLAMA_MODEL", "llama3")
    timeout = float(os.getenv("OLLAMA_TIMEOUT", "60.0"))
    
    # -----------------------------------
    # VALIDACIÓN Y FALLBACK (Lab Only)
    # -----------------------------------
    fallback_warning = ""
    try:
        # Validamos si el modelo existe
        is_valid, available_models = validate_ollama_model(ollama_model, ollama_url)
        
        if not is_valid:
            allow_fallback = os.getenv("LAB_ALLOW_MODEL_FALLBACK", "true").lower() == "true"
            if allow_fallback and available_models:
                fallback_model = available_models[0]
                fallback_warning = f"[WARN] Model '{ollama_model}' not found. Using '{fallback_model}'."
                ollama_model = fallback_model
            else:
                 # Error controlado
                 avail_str = ", ".join(available_models[:3]) + ("..." if len(available_models)>3 else "")
                 return {
                    "answer": f"Error: Modelo '{ollama_model}' no encontrado en Ollama. Disponibles: {avail_str}",
                    "provider": "ollama/error",
                    "llm_ms": 0, "cost_total": 0
                 }
    except Exception as e:
         # Si falla la validación (ej: Ollama caído), atrapamos aquí o dejamos fallar el request principal?
         # Mejor dejar fallar el request principal para manejar timeout/conn error unificado.
         pass

    url = f"{ollama_url}/api/chat"
    
    
    # Construcción de mensajes para Chat API
    messages = []
    # Inyectamos prompts (DB + Context)
    messages.extend(system_messages)
    
    messages.append({"role": "user", "content": user_text})
    
    payload = {
        "model": ollama_model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.5,
            "num_predict": 1000,
        }
    }

    t_prompt = 0
    t_completion = 0
    t_total = 0

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            
        # Parse Chat Response
        # Ollama API returns 'message': {'role': 'assistant', 'content': '...'}
        msg = data.get("message", {})
        answer = msg.get("content", "")
        
        if fallback_warning:
            answer = f"⚠️ {fallback_warning}\n\n{answer}"
        
        # Ollama returns explicit token counts in some versions, or eval_count
        t_completion = data.get("eval_count", 0)
        t_prompt = data.get("prompt_eval_count", 0)
        t_total = t_prompt + t_completion

    except httpx.ConnectError:
        answer = "Error: No se pudo conectar a Ollama. Verifique que el servicio esté corriendo."
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
             answer = f"Error 404: Endpoint o Modelo no encontrado en Ollama ({ollama_model})."
        else:
             answer = f"Error HTTP Ollama: {e}"
    except Exception as e:
        answer = f"Error cognitivo (Ollama): {str(e)}"
        
    # Calculate Costs
    # Ollama is "free" but we track tokens
    # t_total = t_prompt + t_completion
    
    return {
        "answer": answer,
        "tokens_prompt": t_prompt,
        "tokens_completion": t_completion,
        "tokens_total": t_total,
        "provider": f"ollama/{ollama_model}",
        "real_model": ollama_model, # Explicit for UI
        "llm_ms": round((time.perf_counter() - t0) * 1000, 2),
        "cost_inference": 0.0, 
        "cost_infra": cost_infra,
        "cost_total": cost_infra,
        "raw_mode": use_raw
    }

# =========================
# OLLAMA HELPERS
# =========================
def get_ollama_models(base_url: str) -> list:
    """Retorna lista de nombres de modelos disponibles."""
    try:
        with httpx.Client(timeout=5.0) as client:
            res = client.get(f"{base_url}/api/tags")
            if res.status_code == 200:
                models = [m["name"] for m in res.json().get("models", [])]
                return models
    except:
        pass
    return []

def validate_ollama_model(model_name: str, base_url: str):
    """Retorna (Exists: bool, AvailableModels: list)"""
    models = get_ollama_models(base_url)
    # Check exact match or match with tag
    if model_name in models:
        return True, models
    # Check if model_name is substring or prefix? No, strict for now but fuzzy on tag "latest"
    if f"{model_name}:latest" in models:
        return True, models
    return False, models

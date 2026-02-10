from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# DB: Base y engine para inicialización de tablas
from backend.app.db.base import Base
from backend.app.db.session import engine

# Modelos (importar para registrar en Base.metadata)
from backend.app.models import voice_event, memory_node, core, agent_prompts  # noqa: F401

# =========================
# APP INIT
# =========================
app = FastAPI(
    title="Vortex Clinical Core",
    description="Sistema Cognitivo Clínico con separación LIFE / WORK",
    version="0.1.0",
)

# =========================
# CORS (LAB / FUTURO FRONT)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # en producción se restringe
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# STARTUP: Warmup Ollama
# =========================
@app.on_event("startup")
def on_startup():
    # Warmup Ollama en background (no bloquea startup)
    import threading
    from backend.agents.observer_agent import warmup_ollama
    threading.Thread(target=warmup_ollama, daemon=True).start()

    # Iniciar Supervisor de Tasks (12s loop)
    from backend.app.routes.lab import start_supervisor
    start_supervisor()

# =========================
# ROUTERS
# =========================

# LAB Cognitivo
from backend.app.routes.lab import router as lab_router
app.include_router(lab_router)

# Timeline Clínico
from backend.app.routes.timeline import router as timeline_router
app.include_router(timeline_router)

# (futuro)
# from backend.app.routes.procedures import router as procedures_router
# app.include_router(procedures_router)

# =========================
# ROOT
# =========================
@app.get("/")
def root():
    return {
        "service": "Vortex Clinical Core",
        "status": "running",
        "mode": "LAB",
        "endpoints": {
            "docs": "/docs",
            "lab": "/lab",
            "timeline": "/procedures/{procedure_id}/timeline",
        },
    }

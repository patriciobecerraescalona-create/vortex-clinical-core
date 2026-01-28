from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# DB: Base y engine para inicialización de tablas
from app.db.base import Base
from app.db.session import engine

# Modelos (importar para registrar en Base.metadata)
from app.models import voice_event, memory_node, core  # noqa: F401

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
# STARTUP: Crear tablas si no existen
# =========================
@app.on_event("startup")
def on_startup():
    """Crea las tablas en la BD si no existen (checkfirst=True por defecto)."""
    Base.metadata.create_all(bind=engine)

# =========================
# ROUTERS
# =========================

# LAB Cognitivo
from app.routes.lab import router as lab_router
app.include_router(lab_router)

# Timeline Clínico
from app.routes.timeline import router as timeline_router
app.include_router(timeline_router)

# (futuro)
# from app.routes.procedures import router as procedures_router
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

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vortex Clinical Core is a cognitive clinical system that separates LIFE (personal memory) and WORK (clinical timeline) contexts. Built with FastAPI + SQLAlchemy, PostgreSQL with pgvector, and a plain HTML/JS frontend (MVP stage).

## Running the Application

```bash
# Start full stack (PostgreSQL + FastAPI backend)
docker-compose up

# Backend only (if DB already running)
uvicorn app.main:app --host=0.0.0.0 --port=8000 --reload

# STT service (separate)
uvicorn stt.main:app --port=8001
```

- API: http://localhost:8000
- Swagger docs: http://localhost:8000/docs
- Test UI: http://localhost:8000/lab
- Frontend: Open `frontend/index.html` or `lab/front_lab.html` directly in browser

## Architecture

### Request Flow

```
User Input → POST /lab
    → KAI Engine (wake phrase + agent detection)
    → Agent Permission Validation (role-based)
    → Layer 1 Context Building (agent rules/domain)
    → Mode Detection (LIFE vs WORK)
    → Database Persistence (VoiceEvent or MemoryNode)
    → LLM Processing (WORK mode only, currently placeholder)
    → Response
```

### Key Services (backend/app/services/)

| Service | Purpose |
|---------|---------|
| `kai_engine.py` | Wake phrase detection, agent keyword extraction |
| `agent_context.py` | Role→Agent mapping, Layer 1 cognitive context |
| `agent_permissions.py` | Access control validation |
| `mode_detector.py` | LIFE vs WORK classification |
| `cognitive_detector.py` | Signal detection (observation, action, reminder) |
| `llm_agent.py` | LLM provider abstraction (placeholder) |

### Role-Agent Permissions (from agent_context.py)

- `secretary` → support, commercial
- `clinician` → medical, support, life, commercial
- `admin` → support, auditor, commercial
- `manager` → medical, support, auditor, life, commercial
- `anonymous` → commercial only

### Database Models (backend/app/models/)

- `VoiceEvent` - Clinical timeline entries (WORK mode)
- `MemoryNode` - Personal memory entries (LIFE mode)
- `Domain/Subdomain/Document` - Clinical knowledge structure
- `DocumentRuleEvaluation` - Compliance tracking

### Database Connection

PostgreSQL 16 + pgvector at `postgresql+psycopg://vortex_user:vortex_pass@db:5432/vortex`

## Code Conventions

- Comments and variable names are in Spanish
- LLM does not make permission/routing decisions - these are deterministic in code
- CORS is permissive (`*`) for development - must be restricted in production
- Agent definitions in `backend/app/config/agents.py`
- Future SGMI integration will provide real identity context (currently mocked)

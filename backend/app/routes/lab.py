from uuid import UUID

from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.routes.procedures import handle_voice_event
from app.services.agent_context import get_user_context

router = APIRouter()


# =========================
# Modelos de entrada LAB
# =========================

class LabPayload(BaseModel):
    raw_text: str
    user_id: UUID | None = None
    role: str | None = "anonymous"
    options: dict | None = None


# =========================
# UI simple LAB (GET)
# =========================

@router.get("/lab", response_class=HTMLResponse)
def lab_ui():
    """
    UI mínima de pruebas LAB.
    El frontend real puede reemplazar esto sin romper el backend.
    """
    html = """
    <html>
        <head>
            <title>Vortex LAB</title>
        </head>
        <body>
            <h2>Vortex LAB — Observabilidad Cognitiva</h2>

            <form method="post">
                <label>Rol:</label>
                <select name="role">
                    <option value="anonymous">anonymous</option>
                    <option value="secretary">secretary</option>
                    <option value="clinician">clinician</option>
                    <option value="admin">admin</option>
                    <option value="manager">manager</option>
                </select>
                <br><br>

                <label>Input:</label><br>
                <textarea name="raw_text" rows="4" cols="80"></textarea>
                <br><br>

                <button type="submit">Enviar</button>
            </form>
        </body>
    </html>
    """
    return html


# =========================
# Endpoint principal LAB
# =========================

@router.post("/lab")
async def lab_post(request: Request, db: Session = Depends(get_db)):
    """
    Punto único de entrada LAB.
    Aquí se simula el contexto SGMI.
    """

    # -------------------------
    # Parseo flexible (HTML / JSON)
    # -------------------------
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        body = await request.json()
        payload = LabPayload(**body)
    else:
        form = await request.form()
        payload = LabPayload(
            raw_text=form.get("raw_text", ""),
            role=form.get("role", "anonymous"),
        )

    # -------------------------
    # Validación LAB: user_id obligatorio
    # -------------------------
    if not payload.user_id:
        return JSONResponse(
            status_code=400,
            content={
                "error": "LAB_MISSING_USER_ID",
                "detail": "user_id es obligatorio en entorno LAB. Provéelo desde el frontend.",
            },
        )

    # -------------------------
    # Contexto SGMI simulado
    # -------------------------
    user_context = get_user_context(payload.role or "anonymous")

    # IMPORTANTE:
    # payload.options es el CONTRATO SGMI
    payload.options = user_context

    # -------------------------
    # Flujo principal
    # -------------------------
    try:
        result = handle_voice_event(payload, db=db)
    except Exception as e:
        # No caemos: devolvemos error controlado
        return JSONResponse(
            status_code=500,
            content={
                "error": "LAB_EXECUTION_ERROR",
                "detail": str(e),
            },
        )

    return JSONResponse(content=result)

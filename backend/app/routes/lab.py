from uuid import UUID
from typing import Optional

from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.routes.procedures import handle_voice_event
from backend.app.services.agent_context import get_user_context

# ObserverAgent para análisis pasivo
from backend.agents.observer_agent import get_observer, get_warmup_status
from backend.agents.observer_agent import get_observer, get_warmup_status
from backend.app.services.llm_agent import run_llm, get_ollama_models

router = APIRouter()


# =========================
# Modelos de entrada LAB
# =========================

class LabPayload(BaseModel):
    raw_text: str
    user_id: UUID | None = None
    role: str | None = "anonymous"
    options: dict | None = None


class PatientContext(BaseModel):
    """Contexto del paciente para el contraste clínico."""
    patient_name: Optional[str] = None
    age: Optional[int] = None
    sex: Optional[str] = None
    medical_history: Optional[str] = None
    socio_cultural: Optional[str] = None
    reason_for_visit: Optional[str] = None
    clinical_text: Optional[str] = None
    clinical_phase: Optional[str] = "anamnesis"


class ObserverRequest(BaseModel):
    """Request para el endpoint del observer."""
    patient_context: PatientContext
    force: bool = False  # Forzar análisis ignorando throttling


class AgentRequest(BaseModel):
    """Request para el endpoint del agente activo."""
    user_text: str
    role: str = "clinical"
    patient_context: dict = {}
    options: Optional[dict] = None
    # Lab Params
    provider: str = "ollama"
    model: str = ""
    raw: bool = False
    mode: str = "work"


# =========================
# UI Unificada LAB (GET)
# =========================

# =========================
# UI Unificada LAB (GET)
# =========================

@router.get("/lab/models")
def get_lab_models():
    """Proxy para obtener modelos de Ollama."""
    import os
    base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    models = get_ollama_models(base_url)
    return {"models": models, "base_url": base_url}

@router.get("/lab/agents", response_class=HTMLResponse)
def agents_lab_ui():
    """
    UI Dedicada para Experimentación de Agentes (Benchmarking).
    Separada del flujo clínico SGMI.
    """
    html = """
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <title>Vortex Agents LAB</title>
  <style>
    * { box-sizing: border-box; }
    body {
      background: #0b1020;
      color: #e5e7eb;
      font-family: 'Segoe UI', Arial, sans-serif;
      margin: 0;
      padding: 0;
    }

    /* Header */
    .lab-header {
      background: #1e293b;
      padding: 12px 24px;
      border-bottom: 1px solid #334155;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .lab-title {
      font-size: 18px;
      font-weight: 600;
      color: #f8fafc;
      margin: 0;
    }
    .lab-subtitle {
      font-size: 11px;
      color: #64748b;
      margin-top: 2px;
    }

    /* Controls Bar (Top) */
    .controls-bar {
      display: flex;
      gap: 12px;
      align-items: center;
      background: #0f172a;
      padding: 10px 24px;
      border-bottom: 1px solid #1e293b;
      flex-wrap: wrap;
    }
    .ctrl-group {
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    .ctrl-label {
        font-size: 10px;
        color: #64748b;
        text-transform: uppercase;
    }
    .ctrl-select, .ctrl-input {
        background: #1e293b;
        border: 1px solid #334155;
        color: #cbd5e1;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
    }
    
    /* Toggle Switch */
    .toggle-switch {
        position: relative;
        width: 36px;
        height: 20px;
    }
    .toggle-checkbox { opacity: 0; width: 0; height: 0; }
    .toggle-slider {
        position: absolute;
        cursor: pointer;
        top: 0; left: 0; right: 0; bottom: 0;
        background-color: #334155;
        transition: .4s;
        border-radius: 20px;
    }
    .toggle-slider:before {
        position: absolute;
        content: "";
        height: 14px;
        width: 14px;
        left: 3px;
        bottom: 3px;
        background-color: white;
        transition: .4s;
        border-radius: 50%;
    }
    .toggle-checkbox:checked + .toggle-slider { background-color: #3b82f6; }
    .toggle-checkbox:checked + .toggle-slider:before { transform: translateX(16px); }


    /* Main Layout */
    .lab-container {
      display: flex;
      height: calc(100vh - 115px); /* Header + Controls approx */
    }
    .lab-sidebar {
      width: 260px;
      background: #0f172a;
      border-right: 1px solid #1e293b;
      display: flex;
      flex-direction: column;
    }
    .lab-main {
      flex: 1;
      display: flex;
      flex-direction: column;
      background: #0b1020;
    }
    
    /* Chat Area */
    .chat-history {
        flex: 1;
        overflow-y: auto;
        padding: 20px;
        display: flex;
        flex-direction: column;
        gap: 16px;
    }
    .chat-input-area {
        padding: 20px;
        background: #1e293b;
        border-top: 1px solid #334155;
        display: flex;
        gap: 10px;
        align-items: flex-start;
    }
    .chat-input {
        flex: 1;
        background: #0f172a;
        border: 1px solid #334155;
        color: #e5e7eb;
        padding: 12px;
        border-radius: 8px;
        font-family: inherit;
        resize: none;
        min-height: 50px;
    }
    .btn-send {
        background: #3b82f6;
        color: white;
        border: none;
        padding: 0 20px;
        height: 50px;
        border-radius: 8px;
        cursor: pointer;
        font-weight: 600;
        transition: background 0.2s;
    }
    .btn-send:hover { background: #2563eb; }
    .btn-send:disabled {
        background: #475569;
        cursor: not-allowed;
    }
    
    /* Bubbles */
    .bubble {
        max-width: 80%;
        padding: 12px 16px;
        border-radius: 12px;
        line-height: 1.5;
        font-size: 14px;
        position: relative;
    }
    .bubble.user {
        align-self: flex-end;
        background: #2563eb;
        color: white;
        border-top-right-radius: 2px;
    }
    .bubble.agent {
        align-self: flex-start;
        background: #1e293b;
        color: #e2e8f0;
        border: 1px solid #334155;
        border-top-left-radius: 2px;
    }
    
    /* Metrics Footer */
    .msg-footer {
        font-size: 10px;
        color: #64748b;
        margin-top: 6px;
        display: flex;
        gap: 10px;
        border-top: 1px solid #334155;
        padding-top: 4px;
        flex-wrap: wrap;
    }
    .metric-tag {
        display: flex;
        gap: 4px;
    }
    
    /* Sidebar List */
    .history-list {
        overflow-y: auto;
        flex: 1;
    }
    .history-item {
        padding: 12px 16px;
        border-bottom: 1px solid #1e293b;
        cursor: pointer;
        transition: background 0.2s;
    }
    .history-item:hover { background: #1e293b; }
    .history-item.active { background: #334155; border-left: 3px solid #3b82f6; }
    .h-title { font-size: 13px; color: #e2e8f0; margin-bottom: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-weight: 500; }
    .h-meta { font-size: 11px; color: #64748b; display: flex; justify-content: space-between; }

    /* Utility */
    .hidden { display: none !important; }
    .spinner {
      width: 14px;
      height: 14px;
      border: 2px solid #64748b;
      border-top-color: #3b82f6;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>

<!-- Header -->
<header class="lab-header">
  <div>
    <h1 class="lab-title">Vortex Agents LAB</h1>
    <p class="lab-subtitle">Benchmarking Cognitivo Multi-LLM</p>
  </div>
  <div>
       <!-- Future status indicators -->
  </div>
</header>

<div id="vortexInterface">
    <!-- Controls Bar -->
    <div class="controls-bar">
      <div class="ctrl-group">
        <label class="ctrl-label">Provider</label>
        <select id="PROVIDER" class="ctrl-select" onchange="toggleModelInput()">
            <option value="ollama" selected>Ollama (Local)</option>
            <option value="openai">OpenAI (Cloud)</option>
        </select>
      </div>
      
      <div class="ctrl-group">
        <label class="ctrl-label">Model <span id="modelBadge" style="font-size:9px; border-radius:3px; padding:1px 3px; display:none;"></span></label>
        <div style="position:relative;">
            <input type="text" id="MODEL" class="ctrl-input" placeholder="qwen2.5:3b" value="" list="ollamaModels" style="width:140px;" oninput="validateModel()">
            <datalist id="ollamaModels"></datalist>
            <datalist id="openaiModels">
                <option value="gpt-4o"></option>
                <option value="gpt-4.1"></option>
                <option value="gpt-4-turbo"></option>
                <option value="gpt-3.5-turbo"></option>
            </datalist>
        </div>
      </div>
      
      <div class="ctrl-group">
        <label class="ctrl-label">Agent Role</label>
        <select id="AGENT" class="ctrl-select">
            <option value="clinical" selected>🏥 Clínico</option>
            <option value="administrative">📋 Administrativo</option>
            <option value="commercial">💼 Comercial</option>
            <option value="personal">🧘 Asistente</option>
            <option value="support">🛠️ Soporte</option>
        </select>
      </div>
      
      <div class="ctrl-group">
        <label class="ctrl-label">Cognitive Mode</label>
        <select id="MODE" class="ctrl-select">
            <option value="work" selected>Work (Professional)</option>
            <option value="life">Life (Personal)</option>
        </select>
      </div>
      
      <div class="ctrl-group" style="align-items:center;">
        <label class="ctrl-label" style="margin-bottom:4px" id="rawLabel">RAW MODE: OFF</label>
        <label class="toggle-switch">
            <input type="checkbox" id="RAW" class="toggle-checkbox" onchange="updateRawLabel()">
            <span class="toggle-slider"></span>
        </label>
      </div>
      
      <div style="flex:1;"></div>
      
      <div class="ctrl-group" style="text-align:right;">
        <label class="ctrl-label">Session Cost</label>
        <div style="font-size:14px; color:#10b981; font-weight:bold;">$ <span id="sessionCost">0.0000</span></div>
      </div>
    </div>

    <!-- Main Container -->
    <div class="lab-container">
      
      <!-- Sidebar (History) -->
      <aside class="lab-sidebar">
         <div style="padding:16px; border-bottom:1px solid #1e293b; font-size:12px; color:#94a3b8; font-weight:600; text-transform:uppercase;">
            HISTORY (IN-MEMORY)
         </div>
         <div id="historyList" class="history-list">
            <!-- Items added via JS -->
         </div>
         <div style="padding:12px; border-top:1px solid #1e293b;">
            <button onclick="createNewSession()" class="btn-send" style="width:100%; height:40px; font-size:13px; background:#475569;">+ New Chat</button>
         </div>
      </aside>

      <!-- Main Chat -->
      <main class="lab-main">
         <div id="chatHistory" class="chat-history">
            <!-- Bubbles go here -->
         </div>
         
         <div class="chat-input-area">
             <textarea id="userInput" class="chat-input" placeholder="Escribe tu mensaje aquí..." rows="2"></textarea>
             <button id="btnSend" class="btn-send" onclick="sendMessage()">ENVIAR</button>
         </div>
      </main>
      
    </div>
</div>

<script>
// =============================
// STATE MANAGEMENT (Client Side)
// =============================
// Structure: [ { id, title, timestamp, messages: [], totalCost: 0.0 } ]
let sessions = [];
let currentSessionId = null;

// =============================
// INIT
// =============================
let availableOllamaModels = [];

function init() {
    createNewSession();
    fetchModels();
}

async function fetchModels() {
    try {
        const res = await fetch('/lab/models');
        if (res.ok) {
            const data = await res.json();
            availableOllamaModels = data.models || [];
            
            // Populate Datalist
            const dl = document.getElementById('ollamaModels');
            dl.innerHTML = '';
            availableOllamaModels.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m;
                dl.appendChild(opt);
            });
            
            validateModel();
        }
    } catch (e) {
        console.error("Error fetching models", e);
    }
}

function toggleModelInput() {
    const prov = document.getElementById('PROVIDER').value;
    const modelInput = document.getElementById('MODEL');
    const badge = document.getElementById('modelBadge');
    
    if (prov === 'ollama') {
        modelInput.setAttribute('list', 'ollamaModels');
        modelInput.placeholder = "qwen2.5:3b";
        validateModel();
    } else {
        modelInput.setAttribute('list', 'openaiModels'); // Use datalist for OpenAI too
        modelInput.placeholder = "gpt-4o";
        badge.style.display = 'none';
        document.getElementById('btnSend').disabled = false;
    }
}

function updateRawLabel() {
    const raw = document.getElementById('RAW').checked;
    const label = document.getElementById('rawLabel');
    if (raw) {
        label.textContent = "RAW MODE: ON";
        label.style.color = "#f59e0b"; // Amber
    } else {
        label.textContent = "RAW MODE: OFF";
        label.style.color = "#64748b";
    }
}

function validateModel() {
    const prov = document.getElementById('PROVIDER').value;
    if (prov !== 'ollama') return;

    const val = document.getElementById('MODEL').value;
    const badge = document.getElementById('modelBadge');
    const btn = document.getElementById('btnSend');
    
    if (availableOllamaModels.includes(val) || val === "") { // Empty means default, or matched
        if (val !== "") {
            badge.textContent = "OK";
            badge.style.background = "#059669"; // Green
            badge.style.color = "white";
            badge.style.display = "inline-block";
        } else {
             badge.style.display = "none";
        }
        btn.disabled = false;
    } else {
        badge.textContent = "unverified";
        badge.style.background = "#d97706"; // Amber
        badge.style.color = "white";
        badge.style.display = "inline-block";
        // We don't block send, we just warn (amber) because it might be a new tag not yet fetched
        // or user knows better. Real validation happens on backend.
    }
}

function createNewSession() {
    const id = 'sess-' + Date.now();
    const newSession = {
        id: id,
        title: "New Chat",
        timestamp: new Date(),
        messages: [], // { type, content, isHtml, metrics }
        totalCost: 0.0
    };
    
    sessions.unshift(newSession); // Add to top
    currentSessionId = id;
    
    renderSidebar();
    renderChat();
    updateCostDisplay();
}

function switchSession(id) {
    currentSessionId = id;
    renderSidebar();
    renderChat();
    updateCostDisplay();
}

function updateSessionTitle(text) {
    const sess = sessions.find(s => s.id === currentSessionId);
    if (sess && sess.title === "New Chat") {
        sess.title = text.substring(0, 30) + (text.length > 30 ? "..." : "");
        renderSidebar();
    }
}

// =============================
// RENDERING
// =============================
function renderSidebar() {
    const container = document.getElementById('historyList');
    container.innerHTML = '';
    
    sessions.forEach(sess => {
        const item = document.createElement('div');
        item.className = `history-item ${sess.id === currentSessionId ? 'active' : ''}`;
        item.onclick = () => switchSession(sess.id);
        
        // Format time
        const timeStr = sess.timestamp.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        
        item.innerHTML = `
            <div class="h-title">${sess.title}</div>
            <div class="h-meta">
                <span>${timeStr}</span>
                <span>$${sess.totalCost.toFixed(4)}</span>
            </div>
        `;
        container.appendChild(item);
    });
}

function renderChat() {
    const container = document.getElementById('chatHistory');
    container.innerHTML = '';
    
    const sess = sessions.find(s => s.id === currentSessionId);
    if (!sess) return;
    
    if (sess.messages.length === 0) {
        container.innerHTML = `
            <div style="text-align:center; color:#475569; margin-top:60px; font-size:13px;">
               <p style="font-size:16px; color:#e2e8f0; font-weight:600;">Vortex Cognitive Lab</p>
               <p>Configura los parámetros arriba y comienza.</p>
               <p style="margin-top:20px; font-size:11px;">Providers: Ollama (Local) | OpenAI (Cloud)</p>
            </div>
        `;
        return;
    }
    
    sess.messages.forEach(msg => {
        createBubbleDOM(msg, container);
    });
    
    container.scrollTop = container.scrollHeight;
}

function createBubbleDOM(msg, container) {
    const div = document.createElement('div');
    div.className = `bubble ${msg.type}`;
    
    if (msg.isHtml) {
        div.innerHTML = msg.content;
    } else {
        // Simple markdown-ish
        div.innerHTML = msg.content.replace(/\\n/g, '<br>');
    }
    
    if (msg.metrics) {
        const footer = document.createElement('div');
        footer.className = 'msg-footer';
        footer.innerHTML = `
            <span class="metric-tag">🤖 ${msg.metrics.real_model || msg.metrics.prov}</span>
            <span class="metric-tag" style="${msg.metrics.raw_mode ? 'color:#f59e0b;' : 'color:#3b82f6;'}">
                ${msg.metrics.raw_mode ? '[Base]' : '[Base+Ext]'}
            </span>
            <span class="metric-tag">⚡ ${msg.metrics.ms}ms</span>
            <span class="metric-tag">📝 ${msg.metrics.tok} tok</span>
            <span class="metric-tag" style="color:#10b981;">💰 $${msg.metrics.cost.toFixed(5)}</span>
        `;
        div.appendChild(footer);
    }
    
    container.appendChild(div);
}

function updateCostDisplay() {
    const sess = sessions.find(s => s.id === currentSessionId);
    if (sess) {
        document.getElementById('sessionCost').innerText = sess.totalCost.toFixed(5);
    }
}

// =============================
// ACTIONS
// =============================
async function sendMessage() {
    const inputEl = document.getElementById('userInput');
    const text = inputEl.value.trim();
    if (!text) return;

    const sess = sessions.find(s => s.id === currentSessionId);
    if (!sess) return;

    // Update title if first message
    updateSessionTitle(text);

    // Add User Message
    sess.messages.push({ type: 'user', content: text, isHtml: false });
    renderChat();
    
    inputEl.value = '';
    
    // Lock UI
    const btn = document.getElementById('btnSend');
    btn.disabled = true;
    btn.innerText = '...';

    // Params
    const provider = document.getElementById('PROVIDER').value;
    const model = document.getElementById('MODEL').value;
    const agent = document.getElementById('AGENT').value;
    const raw = document.getElementById('RAW').checked;
    const mode = document.getElementById('MODE').value;

    // Loading State (Temporary visual)
    const loadingMsg = { type: 'agent', content: '<div class="spinner"></div>', isHtml: true };
    createBubbleDOM(loadingMsg, document.getElementById('chatHistory')); // Just append visual
    const chatContainer = document.getElementById('chatHistory');
    chatContainer.scrollTop = chatContainer.scrollHeight;

    try {
        // 1. Submit Task
        const res = await fetch('/lab/agent', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                user_text: text,
                role: agent,
                provider: provider,
                model: model,
                raw: raw,
                mode: mode,
                options: {}
            })
        });
        
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        
        const data = await res.json();
        const taskId = data.task_id;
        
        // 2. Poll
        pollAgent(taskId, { provider, model, agent, raw });
        
    } catch (err) {
        // Remove loading visual by re-rendering
        renderChat(); 
        
        // Add Error Msg
        sess.messages.push({ type: 'agent', content: `Error: ${err.message}`, isHtml: false });
        renderChat();
        
        btn.disabled = false;
        btn.innerText = 'ENVIAR';
    }
}

async function pollAgent(taskId, metadata) {
    try {
        const res = await fetch(`/lab/agent/${taskId}`);
        if (!res.ok) throw new Error("Polling error");
        
        const data = await res.json();
        
        if (data.status === 'processing') {
            setTimeout(() => pollAgent(taskId, metadata), 1000);
            return;
        }
        
        // Finalize
        document.getElementById('btnSend').disabled = false;
        document.getElementById('btnSend').innerText = 'ENVIAR';
        
        const sess = sessions.find(s => s.id === currentSessionId);
        
        // Remove loading visual by re-rendering (the loading msg wasn't pushed to sess.messages)
        // renderChat(); // Will be called after push
        
        if (data.status === 'ok') {
            const result = data.result || {};
            const answer = result.answer || "No answer";
            const metrics = {
                 ms: result.llm_ms || 0,
                 tok: result.tokens_total || 0,
                 cost: result.cost_total || 0,
                 prov: result.provider || metadata.provider
            };
            
            // Add Agent Message to Session
            sess.messages.push({ 
                type: 'agent', 
                content: answer, 
                isHtml: false, 
                metrics: metrics 
            });
            
            // Warn RAW
            if (metadata.raw && metadata.agent !== 'clinical') {
                 sess.messages.push({ 
                    type: 'agent', 
                    content: "⚠️ [WARN] RAW mode was requested but ignored because agent is not 'clinical'.", 
                    isHtml: false
                });
            }
            
            // Update Cost
            sess.totalCost += metrics.cost;
            updateCostDisplay();
            
        } else {
             sess.messages.push({ type: 'agent', content: "Server Error: " + JSON.stringify(data), isHtml: false });
        }
        
        renderChat();
        renderSidebar(); // Update cost in sidebar
        
    } catch (err) {
        document.getElementById('btnSend').disabled = false;
        document.getElementById('btnSend').innerText = 'ENVIAR';
        
        const sess = sessions.find(s => s.id === currentSessionId);
        sess.messages.push({ type: 'agent', content: "Network Error", isHtml: false });
        renderChat();
    }
}

// Allow Enter to send
document.getElementById('userInput').addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Start
init();

</script>
</body>
</html>
    """
    return html

@router.get("/lab", response_class=HTMLResponse)
def lab_ui():
    """
    SGMI Dashboard Placeholder.
    La UI de experimentación se movió a /lab/agents.
    """
    html = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
      <meta charset="UTF-8">
      <title>Vortex SGMI Lab</title>
      <style>
        body { background: #0b1020; color: #fff; font-family: sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }
        .card { background: #1e293b; padding: 40px; border-radius: 12px; text-align: center; border: 1px solid #334155; }
        h1 { color: #3b82f6; margin-bottom: 8px; }
        p { color: #94a3b8; margin-bottom: 24px; }
        a { display: inline-block; background: #3b82f6; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: bold; }
        a:hover { background: #2563eb; }
      </style>
    </head>
    <body>
      <div class="card">
         <h1>Vortex SGMI Lab</h1>
         <p>Sistema de Gestión Médica Inteligente (SGMI)</p>
         <div style="margin: 20px 0; font-size: 12px; color: #64748b;">
            STATUS: <span style="color: #10b981;">ONLINE</span>
         </div>
         <p>Para experimentación con agentes y benchmarking, usa el nuevo laboratorio dedicado:</p>
         <a href="/lab/agents">Ir a Vortex Agents LAB →</a>
      </div>
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


# =========================
# System Status Endpoint
# =========================
@router.get("/lab/status")
def get_lab_status():
    """Retorna estado del sistema (warmup, db, etc)."""
    return {
        "status": "running",
        "warmup_done": get_warmup_status()
    }


# =========================
# Observer Agent Endpoint
# =========================

# =========================
# Async Task Store & Models
# =========================
from fastapi import BackgroundTasks
import uuid
from datetime import datetime

# Simple in-memory task store
# Structure: { task_id: { "status": "processing"|"done"|"error", "result": ..., "error": ... } }
# Structure: { task_id: { "status": "processing"|"done"|"error", "result": ..., "error": ... } }
tasks = {}
tasks_agent = {}

import asyncio

async def supervisor_loop():
    """Calcula y limpia timeouts cada 12s."""
    print("[SUPERVISOR] Iniciando loop de monitoreo (12s)...")
    while True:
        try:
            await asyncio.sleep(12)
            now = datetime.now()
            
            # Revisar Observer Tasks
            for tid, t in tasks.items():
                if t["status"] == "processing":
                    started = datetime.fromisoformat(t["started_at"])
                    if (now - started).total_seconds() > 60:
                        tasks[tid] = {
                            "status": "error",
                            "error": "TIMEOUT_SUPERVISOR",
                            "result": {"llm_error": "Timeout forzado por supervisor"}
                        }
                        print(f"[SUPERVISOR] Task {tid} timed out.")

            # Revisar Agent Tasks
            for tid, t in tasks_agent.items():
                if t["status"] == "processing":
                    started = datetime.fromisoformat(t["started_at"])
                    if (now - started).total_seconds() > 60:
                        tasks_agent[tid] = {
                            "status": "error",
                            "error": "TIMEOUT_SUPERVISOR",
                            "result": {"answer": "Error: Timeout forzado por supervisor"}
                        }
                        print(f"[SUPERVISOR] Agent Task {tid} timed out.")
                        
        except Exception as e:
            print(f"[SUPERVISOR] Error en loop: {e}")
            await asyncio.sleep(5)  # Backoff ante error interno

def start_supervisor():
    """Inicia el supervisor en background (sin bloquear)."""
    asyncio.create_task(supervisor_loop())

def run_observer_background(task_id: str, patient_context: dict, force: bool):
    """Background worker wrapper"""
    try:
        observer = get_observer()
        result = observer.analyze(
            patient_context=patient_context,
            force=force
        )
        tasks[task_id] = {
            "status": "done",
            "result": result,
            "updated_at": datetime.now().isoformat()
        }
    except Exception as e:
        tasks[task_id] = {
            "status": "error",
            "result": {
                "llm_status": "error",
                "llm_error": str(e),
                "visual_indicator": "gray",
                "mode": "observer",
                "metrics": {"response_time_ms": 0, "eval_count": 0}
            },
            "error": str(e)
        }

@router.post("/lab/observer", status_code=202)
async def observer_analyze(request: ObserverRequest, background_tasks: BackgroundTasks):
    """
    Endpoint ASYNC para el ObserverAgent.
    Retorna inmediatamente un task_id.
    """
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "processing", "started_at": datetime.now().isoformat()}

    background_tasks.add_task(
        run_observer_background, 
        task_id, 
        request.patient_context.model_dump(), 
        request.force
    )

    return {"task_id": task_id, "status": "processing"}

@router.get("/lab/observer/{task_id}")
async def get_observer_result(task_id: str):
    """Polling endpoint"""
    task = tasks.get(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"error": "Task not found"})
    
    if task["status"] == "processing":
        return {"status": "processing"}
    
    # Return result consistent with previous schema
    return {
        "status": "ok", # API status ok
        "task_status": task["status"],
        "analysis": task["result"]
    }


# =========================
# Agent Core Endpoint (Async)
# =========================

def run_agent_background(task_id: str, user_text: str, role: str, context: dict, options: dict, provider: str, model: str, raw: bool, mode: str):
    """Background worker para Agent Core"""
    try:
        # Llamada al núcleo cognitivo real con ROL y CONTEXTO
        result = run_llm(
            user_text=user_text,
            role=role,
            context=context,
            options=options,
            provider=provider,
            model=model,
            raw=raw,
            mode=mode
        )
        tasks_agent[task_id] = {
            "status": "done",
            "result": result,
            "updated_at": datetime.now().isoformat()
        }
    except Exception as e:
        tasks_agent[task_id] = {
            "status": "error",
            "result": {
                "answer": f"Error del agente: {str(e)}",
                "tokens": 0,
                "provider": "error"
            },
            "error": str(e)
        }

@router.post("/lab/agent", status_code=202)
async def agent_analyze(request: AgentRequest, background_tasks: BackgroundTasks):
    """
    Endpoint ASYNC para el Agente Activo (Core).
    """
    task_id = str(uuid.uuid4())
    tasks_agent[task_id] = {"status": "processing", "started_at": datetime.now().isoformat()}

    background_tasks.add_task(
        run_agent_background, 
        task_id, 
        request.user_text, 
        request.role,
        request.patient_context,
        request.options or {},
        provider=request.provider,
        model=request.model,
        raw=request.raw,
        mode=request.mode
    )

    return {"task_id": task_id, "status": "processing"}

@router.get("/lab/agent/{task_id}")
async def get_agent_result(task_id: str):
    """Polling endpoint para Agent"""
    task = tasks_agent.get(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"error": "Task not found"})
    
    if task["status"] == "processing":
        return {"status": "processing"}
    
    return {
        "status": "ok",
        "task_status": task["status"],
        "result": task["result"]
    }

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
from backend.app.services.llm_agent import run_llm

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


# =========================
# UI Unificada LAB (GET)
# =========================

@router.get("/lab", response_class=HTMLResponse)
def lab_ui():
    """
    LAB Unificado con dos modos exclusivos:
    1. SGMI · Observador pasivo
    2. Agentes Vortex (placeholder)
    """
    html = """
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <title>Vortex Clinical LAB</title>
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
      padding: 16px 24px;
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
      font-size: 12px;
      color: #64748b;
      margin-top: 2px;
    }

    /* Mode Switcher */
    .mode-switcher {
      display: flex;
      gap: 8px;
    }
    .mode-btn {
      padding: 8px 16px;
      border: 1px solid #475569;
      background: transparent;
      color: #94a3b8;
      border-radius: 6px;
      cursor: pointer;
      font-size: 13px;
      transition: all 0.2s;
    }
    .mode-btn:hover {
      background: #334155;
      color: #e5e7eb;
    }
    .mode-btn.active {
      background: #3b82f6;
      border-color: #3b82f6;
      color: #fff;
    }
    .mode-btn:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    /* Main Layout */
    .lab-container {
      display: flex;
      height: calc(100vh - 65px);
    }
    .lab-main {
      flex: 1;
      padding: 20px;
      overflow-y: auto;
    }
    .lab-sidebar {
      width: 340px;
      background: #0f172a;
      border-left: 1px solid #1e293b;
      padding: 20px;
      overflow-y: auto;
    }

    /* Sections */
    .section {
      background: #1e293b;
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 16px;
    }
    .section-title {
      font-size: 12px;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .section-title::before {
      content: '';
      width: 3px;
      height: 12px;
      background: #3b82f6;
      border-radius: 2px;
    }

    /* Form Grid */
    .form-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    .form-group {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .form-group.full-width {
      grid-column: 1 / -1;
    }
    .form-group label {
      font-size: 11px;
      color: #64748b;
      text-transform: uppercase;
    }
    .form-group input,
    .form-group select,
    .form-group textarea {
      background: #0f172a;
      border: 1px solid #334155;
      color: #e5e7eb;
      padding: 10px 12px;
      border-radius: 6px;
      font-size: 14px;
      transition: border-color 0.2s;
    }
    .form-group input:focus,
    .form-group select:focus,
    .form-group textarea:focus {
      outline: none;
      border-color: #3b82f6;
    }
    .form-group textarea {
      resize: vertical;
      min-height: 80px;
    }

    /* Observer Panel */
    .observer-panel {
      background: #0f172a;
      border: 1px solid #1e293b;
      border-radius: 8px;
    }
    .observer-header {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 16px;
      border-bottom: 1px solid #1e293b;
    }
    .semaforo {
      width: 20px;
      height: 20px;
      border-radius: 50%;
      background: #22c55e;
      box-shadow: 0 0 10px rgba(34, 197, 94, 0.4);
      transition: all 0.3s;
    }
    .semaforo.yellow {
      background: #eab308;
      box-shadow: 0 0 10px rgba(234, 179, 8, 0.4);
    }
    .semaforo.red {
      background: #ef4444;
      box-shadow: 0 0 10px rgba(239, 68, 68, 0.4);
    }
    .semaforo.gray {
      background: #64748b;
      box-shadow: none;
    }
    .observer-title {
      font-size: 13px;
      color: #94a3b8;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .observer-phase {
      margin-left: auto;
      font-size: 11px;
      color: #22d3ee;
      background: #164e63;
      padding: 4px 8px;
      border-radius: 4px;
    }
    .observer-status {
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 8px 16px;
      background: #0a0f1a;
      border-bottom: 1px solid #1e293b;
      font-size: 11px;
      color: #94a3b8;
    }
    .observer-status .spinner {
      width: 10px;
      height: 10px;
      border: 2px solid #334155;
      border-top-color: #3b82f6;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    .observer-content {
      padding: 16px;
    }
    .observer-section {
      margin-bottom: 16px;
    }
    .observer-section:last-child {
      margin-bottom: 0;
    }
    .observer-label {
      font-size: 10px;
      color: #64748b;
      text-transform: uppercase;
      margin-bottom: 6px;
    }
    .observer-summary {
      font-size: 14px;
      color: #e5e7eb;
      line-height: 1.5;
    }
    .observer-patterns {
      list-style: none;
      padding: 0;
      margin: 0;
    }
    .observer-patterns li {
      padding: 8px 12px;
      background: #1e293b;
      border-radius: 4px;
      margin-bottom: 6px;
      font-size: 13px;
      color: #cbd5e1;
      border-left: 3px solid #3b82f6;
    }
    .observer-notes {
      font-size: 12px;
      color: #94a3b8;
      font-style: italic;
      line-height: 1.5;
    }
    .observer-loading {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      padding: 30px;
      color: #64748b;
    }
    .spinner {
      width: 16px;
      height: 16px;
      border: 2px solid #334155;
      border-top-color: #3b82f6;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    .observer-empty {
      text-align: center;
      padding: 30px;
      color: #475569;
      font-size: 13px;
    }

    /* Placeholder Mode */
    .placeholder-mode {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 100%;
      color: #475569;
      text-align: center;
    }
    .placeholder-icon {
      font-size: 48px;
      margin-bottom: 16px;
      opacity: 0.5;
    }
    .placeholder-text {
      font-size: 14px;
      max-width: 300px;
      line-height: 1.6;
    }

    /* Observer - Insufficient */
    .insufficient-section {
      text-align: center;
      padding: 24px 12px;
    }
    .insufficient-msg {
      font-size: 12px;
      color: #64748b;
    }
    .missing-list {
      font-size: 11px;
      color: #94a3b8;
      margin-top: 8px;
    }

    /* Observer Labels */
    .label-red { color: #f87171 !important; }
    .label-yellow { color: #fbbf24 !important; }
    .label-green { color: #4ade80 !important; }
    .label-orange { color: #fb923c !important; }

    /* Item Lists */
    .item-list {
      list-style: none;
      padding: 0;
      margin: 0;
    }
    .item-list li {
      padding: 6px 0;
      font-size: 12px;
      color: #e5e7eb;
      border-bottom: 1px solid #1e293b;
    }
    .item-list li:last-child {
      border-bottom: none;
    }
    .item-list li strong {
      color: #f8fafc;
    }
    .rationale {
      display: block;
      font-size: 11px;
      color: #94a3b8;
      margin-top: 2px;
    }
    .diff-info {
      display: block;
      font-size: 11px;
      color: #67e8f9;
      margin-top: 2px;
    }

    /* High Impact (red) */
    .high-impact-section {
      background: rgba(239, 68, 68, 0.1);
      border-left: 3px solid #ef4444;
      border-radius: 4px;
      padding: 8px 10px;
    }

    /* Alternatives (yellow) */
    .alternatives-section {
      background: rgba(234, 179, 8, 0.08);
      border-left: 3px solid #eab308;
      border-radius: 4px;
      padding: 8px 10px;
    }

    /* Discriminators */
    .discriminators-section {
      background: #0f172a;
      border-left: 3px solid #3b82f6;
      border-radius: 4px;
      padding: 8px 10px;
    }
    .disc-list li {
      border-bottom: none;
      padding: 4px 0;
    }

    /* Management (green) */
    .management-section {
      background: rgba(34, 197, 94, 0.08);
      border-left: 3px solid #22c55e;
      border-radius: 4px;
      padding: 8px 10px;
    }

    /* Triggers (orange) */
    .triggers-section {
      background: rgba(251, 146, 60, 0.1);
      border-left: 3px solid #f97316;
      border-radius: 4px;
      padding: 8px 10px;
    }
    .trigger-list {
      list-style: none;
      padding: 0;
      margin: 0;
    }
    .trigger-list li {
      padding: 4px 0;
      font-size: 11px;
      color: #fdba74;
    }

    /* Metrics Bar */
    .metrics-bar {
      display: flex;
      justify-content: space-between;
      padding: 6px 8px;
      background: #0a0f1a;
      border-radius: 4px;
      font-size: 10px;
      color: #64748b;
      margin-top: 8px;
    }

    /* Hidden */
    .hidden { display: none !important; }
  </style>
</head>
<body>

<!-- Header -->
<header class="lab-header">
  <div>
    <h1 class="lab-title">Vortex Clinical LAB</h1>
    <p class="lab-subtitle">Entorno de pruebas cognitivas</p>
  </div>
  <div class="mode-switcher">
    <button id="btnModeSGMI" class="mode-btn active" onclick="setMode('sgmi')">
      SGMI · Observador
    </button>
    <button id="btnModeVortex" class="mode-btn" onclick="setMode('vortex')">
      Agentes Vortex
    </button>
  </div>
</header>

<!-- Main Container -->
<div class="lab-container">

  <!-- =====================
       SGMI MODE (Active)
       ===================== -->
  <div id="sgmiMode" class="lab-main">

    <!-- Patient Context -->
    <div class="section">
      <div class="section-title">Contexto del Paciente</div>
      <div class="form-grid">
        <div class="form-group">
          <label for="patientName">Nombre</label>
          <input type="text" id="patientName" value="Juan Pérez" />
        </div>
        <div class="form-group">
          <label for="patientAge">Edad</label>
          <input type="number" id="patientAge" value="45" min="0" max="150" />
        </div>
        <div class="form-group">
          <label for="patientSex">Sexo</label>
          <select id="patientSex">
            <option value="M" selected>Masculino</option>
            <option value="F">Femenino</option>
          </select>
        </div>
        <div class="form-group">
          <label for="clinicalPhase">Fase clínica</label>
          <select id="clinicalPhase">
            <option value="chief_complaint" selected>1. Motivo de consulta</option>
            <option value="history" disabled>2. Historia clínica</option>
            <option value="physical_exam" disabled>3. Examen físico</option>
            <option value="hypothesis" disabled>4. Hipótesis</option>
          </select>
        </div>
        <div class="form-group full-width">
          <label for="reasonForVisit">Motivo de consulta</label>
          <input type="text" id="reasonForVisit" value="Dolor de estómago" />
        </div>
        <div class="form-group full-width">
          <label for="medicalHistory">Antecedentes médicos</label>
          <input type="text" id="medicalHistory" value="HTA, gastritis crónica" />
        </div>
        <div class="form-group full-width">
          <label for="socioCultural">Contexto socio-cultural</label>
          <input type="text" id="socioCultural" value="Trabajo nocturno, alimentación irregular" />
        </div>
      </div>
    </div>

    <!-- Clinical Text -->
    <div class="section">
      <div class="section-title">Anamnesis</div>
      <div class="form-group full-width">
        <textarea id="clinicalText" rows="4" placeholder="Descripción clínica...">Paciente indica dolor de estómago con vómitos reiterados de 3 días.</textarea>
      </div>
    </div>

    </div>



  <!-- =====================
       VORTEX MODE (Placeholder)
       ===================== -->
  <div id="vortexMode" class="lab-main hidden">
    <!-- Input Section -->
    <div class="section">
      <div class="section-title">Agente Activo (Vortex Core)</div>
      
      <!-- Role and Connection Status Bar -->
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
        <!-- Role Selector -->
        <div class="form-group" style="flex:1; max-width:200px; margin:0;">
          <select id="agentRole" style="padding:6px; font-size:12px; height:32px;">
             <option value="clinical" selected>🏥 Clínico (Experto)</option>
             <option value="administrative">📋 Administrativo</option>
             <option value="commercial">💼 Comercial</option>
             <option value="personal">🧘 Asistente Personal</option>
             <option value="support">🛠️ Soporte Técnico</option>
          </select>
        </div>
        
        <!-- Connection Status -->
        <div id="connectionStatus" style="font-size:11px; color:#94a3b8; display:flex; align-items:center; gap:6px;">
           <div class="spinner" style="width:10px; height:10px; border-width:1px;"></div>
           Conectando al motor...
        </div>
      </div>

      <div class="form-group full-width">
        <textarea id="agentInput" rows="4" placeholder="Escribe tu consulta al agente...">Paciente de 45 años con dolor abdominal...</textarea>
      </div>
      <div class="form-actions right">
        <button id="btnSendAgent" class="lab-btn primary" onclick="callAgent()">
          Enviar Consulta
        </button>
      </div>
    </div>

    <!-- Agent Output Section -->
    <div class="section">
      <div class="section-title">Respuesta Cognitiva</div>
      <div id="agentStatus" class="agent-status" style="display:none; color:#64748b; margin-bottom:10px;">
        <span class="spinner">●</span> Procesando...
      </div>
      <div id="agentOutput" class="agent-output" style="
          min-height: 100px;
          padding: 15px;
          background: #f8fafc;
          border: 1px solid #e2e8f0;
          border-radius: 6px;
          font-family: monospace;
          white-space: pre-wrap;
          color: #334155;">
        Esperando input...
      </div>
    </div>
  </div>

  <!-- =====================
       SIDEBAR: Observer Panel
       ===================== -->
  <aside class="lab-sidebar">
    <div class="observer-panel">
      <div class="observer-header">
        <div id="semaforo" class="semaforo"></div>
        <span class="observer-title">Contraste Clínico</span>
        <span id="phaseLabel" class="observer-phase">—</span>
      </div>
      <div id="observerStatus" class="observer-status">
        <span style="color:#64748b;">●</span> <span>Iniciando...</span>
      </div>
      <div id="observerContent" class="observer-content">
        <div class="observer-empty">
          Esperando contexto clínico...
        </div>
      </div>
    </div>
  </aside>

</div>

<script>
// =============================
// Mode Management
// =============================
let currentMode = 'sgmi';

function setMode(mode) {
  currentMode = mode;

  // Update buttons
  document.getElementById('btnModeSGMI').classList.toggle('active', mode === 'sgmi');
  document.getElementById('btnModeVortex').classList.toggle('active', mode === 'vortex');

  // Toggle views
  document.getElementById('sgmiMode').classList.toggle('hidden', mode !== 'sgmi');
  document.getElementById('vortexMode').classList.toggle('hidden', mode !== 'vortex');

  // In SGMI mode, trigger observer update
  if (mode === 'sgmi') {
    callObserver(true);
  }
}

// =============================
// Observer Agent
// =============================
let observerLoading = false;
let lastContextHash = '';
let lastAnalysis = null;
let lastUpdateTime = null;
let pollingInterval = null;
let currentController = null;  // AbortController activo
let countdownTimer = null;
let slowResponseTimer = null;
const POLL_INTERVAL_MS = 15000;  // 15 segundos (más espacio para análisis largo)

function getPatientContext() {
  return {
    patient_name: document.getElementById('patientName').value || '',
    patient_age: parseInt(document.getElementById('patientAge').value) || 0,
    patient_sex: document.getElementById('patientSex').value || '',
    medical_history: document.getElementById('medicalHistory').value || '',
    socio_cultural: document.getElementById('socioCultural').value || '',
    reason_for_visit: document.getElementById('reasonForVisit').value || '',
    clinical_text: document.getElementById('clinicalText').value || '',
    clinical_phase: document.getElementById('clinicalPhase').value || 'anamnesis'
  };
}

function hashContext(ctx) {
  // Simple hash del contexto para comparación
  const str = JSON.stringify(ctx);
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return hash.toString();
}

function hasContextChanged() {
  const currentHash = hashContext(getPatientContext());
  return currentHash !== lastContextHash;
}

function formatTime(date) {
  return date.toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function showStatus(status, message) {
  const statusEl = document.getElementById('observerStatus');
  if (!statusEl) return;

  const icons = {
    'analyzing': '<div class="spinner"></div>',
    'updating': '<div class="spinner"></div>',
    'updated': '<span style="color:#22c55e;">●</span>',
    'idle': '<span style="color:#64748b;">●</span>',
    'error': '<span style="color:#ef4444;">●</span>'
  };

  statusEl.innerHTML = `${icons[status] || ''} <span>${message}</span>`;
}

function showLoading(isUpdate = false) {
  const msg = 'Analizando razonamiento clínico profundo...';
  showStatus(isUpdate ? 'updating' : 'analyzing', msg);

  if (!lastAnalysis) {
    document.getElementById('observerContent').innerHTML = `
      <div class="observer-loading">
        <div class="spinner"></div>
        <span style="font-size:11px;">${msg}</span>
      </div>
    `;
  }
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function updateObserverUI(analysis) {
  const semaforo = document.getElementById('semaforo');
  const phaseLabel = document.getElementById('phaseLabel');
  const content = document.getElementById('observerContent');

  // Estados: ok, waiting, error
  const llmStatus = analysis.llm_status || 'ok';
  const isError = llmStatus === 'error';
  const isWaiting = llmStatus === 'waiting';
  const metrics = analysis.metrics || {};

  // Semaforo basado en visual_indicator
  semaforo.className = 'semaforo';
  if (isError || isWaiting || analysis.visual_indicator === 'gray') {
    semaforo.classList.add('gray');
  } else if (analysis.visual_indicator === 'yellow') {
    semaforo.classList.add('yellow');
  } else if (analysis.visual_indicator === 'red') {
    semaforo.classList.add('red');
  }

  // Badge según estado
  if (isError) {
    phaseLabel.textContent = 'ERROR';
    phaseLabel.style.background = '#7f1d1d';
    phaseLabel.style.color = '#fca5a5';
  } else if (isWaiting || analysis.insufficient) {
    phaseLabel.textContent = 'ESPERA';
    phaseLabel.style.background = '#334155';
    phaseLabel.style.color = '#94a3b8';
  } else if (analysis.no_additional) {
    phaseLabel.textContent = 'OK';
    phaseLabel.style.background = '#14532d';
    phaseLabel.style.color = '#86efac';
  } else {
    const ms = metrics.response_time_ms || 0;
    const model = metrics.model_name || 'ollama';
    const tokens = metrics.eval_count || 0;
    phaseLabel.textContent = `${model} · ${(ms/1000).toFixed(1)}s · ${tokens} tokens`;
    phaseLabel.style.background = ms > 4000 ? '#92400e' : '#14532d';
    phaseLabel.style.color = ms > 4000 ? '#fde047' : '#86efac';
  }

  let html = '';

  // Error del LLM (no de infraestructura)
  if (isError) {
    html = `<div class="observer-section"><div style="color:#fca5a5;font-size:12px;">${escapeHtml(analysis.llm_error || 'Error del modelo')}</div></div>`;
    content.innerHTML = html;
    return;
  }

  // Waiting - contexto insuficiente (no se llamó al LLM)
  if (isWaiting || analysis.insufficient) {
    const missing = Array.isArray(analysis.missing) ? analysis.missing : [];
    html = `
      <div class="observer-section insufficient-section">
        <div class="insufficient-msg">Completa anamnesis</div>
        ${missing.length > 0 ? `<div class="missing-list">Falta: ${missing.join(', ')}</div>` : ''}
      </div>
    `;
    content.innerHTML = html;
    return;
  }

  // Sin aporte adicional (LLM respondió pero no hay info nueva)
  if (analysis.no_additional) {
    html = `
      <div class="observer-section" style="text-align:center;padding:20px;">
        <div style="color:#94a3b8;font-size:12px;">Sin aporte clínico adicional en este momento</div>
      </div>
    `;
    content.innerHTML = html;
    return;
  }

  // 1. ALTO IMPACTO (rojo)
  const highImpact = Array.isArray(analysis.high_impact) ? analysis.high_impact : [];
  if (highImpact.length > 0) {
    html += `
      <div class="observer-section high-impact-section">
        <div class="observer-label label-red">A Descartar (Alto Impacto)</div>
        <ul class="item-list">
          ${highImpact.map(h => `<li><strong>${escapeHtml(h.scenario || h)}</strong>${h.rationale ? `<span class="rationale">${escapeHtml(h.rationale)}</span>` : ''}</li>`).join('')}
        </ul>
      </div>
    `;
  }

  // 2. ALTERNATIVAS
  const alts = Array.isArray(analysis.alternatives) ? analysis.alternatives : [];
  if (alts.length > 0) {
    html += `
      <div class="observer-section alternatives-section">
        <div class="observer-label label-yellow">Alternativas</div>
        <ul class="item-list">
          ${alts.map(a => `<li><strong>${escapeHtml(a.scenario || a)}</strong>${a.rationale ? `<span class="rationale">${escapeHtml(a.rationale)}</span>` : ''}</li>`).join('')}
        </ul>
      </div>
    `;
  }

  // 3. DISCRIMINADORES
  const discs = Array.isArray(analysis.discriminators) ? analysis.discriminators : [];
  if (discs.length > 0) {
    html += `
      <div class="observer-section discriminators-section">
        <div class="observer-label">Estudios Diferenciadores</div>
        <ul class="item-list disc-list">
          ${discs.map(d => `<li><strong>${escapeHtml(d.test || d)}</strong>${d.differentiates ? `<span class="diff-info">→ ${escapeHtml(d.differentiates)}</span>` : ''}</li>`).join('')}
        </ul>
      </div>
    `;
  }

  // 4. MANEJO
  const paths = Array.isArray(analysis.management_paths) ? analysis.management_paths : [];
  if (paths.length > 0) {
    html += `
      <div class="observer-section management-section">
        <div class="observer-label label-green">Escenarios de Manejo</div>
        <ul class="item-list">
          ${paths.map(p => `<li><strong>${escapeHtml(p.path || p)}</strong>${p.when ? `<span class="rationale">${escapeHtml(p.when)}</span>` : ''}</li>`).join('')}
        </ul>
      </div>
    `;
  }

  // 5. TRIGGERS
  const triggers = Array.isArray(analysis.pivot_triggers) ? analysis.pivot_triggers : [];
  if (triggers.length > 0) {
    html += `
      <div class="observer-section triggers-section">
        <div class="observer-label label-orange">Cambian Conducta</div>
        <ul class="trigger-list">
          ${triggers.map(t => `<li>${escapeHtml(String(t))}</li>`).join('')}
        </ul>
      </div>
    `;
  }

  // Métricas (footer)
  if (metrics.response_time_ms > 0) {
    html += `
      <div class="metrics-bar">
        <span>${metrics.response_time_ms}ms</span>
        <span>${metrics.eval_count || 0} tok</span>
      </div>
    `;
  }

  if (!html) {
    html = '<div class="observer-empty">Sin observaciones.</div>';
  }

  content.innerHTML = html;
}

async function pollObserverStatus(taskId) {
  try {
    const res = await fetch(`/lab/observer/${taskId}`);
    if (!res.ok) {
        showStatus('error', `Error polling task: ${res.status}`);
        observerLoading = false;
        return;
    }
    
    const data = await res.json();
    
    if (data.status === 'processing') {
      // Sigue procesando, volver a consultar en 1s
      setTimeout(() => pollObserverStatus(taskId), 1000);
      return;
    }
    
    // Task completada (status 'ok')
    if (data.status === 'ok') {
      observerLoading = false; // RELEASE LOCK
      
      // Verificar status interno del análisis
      if (data.task_status === 'error') {
         // Error de ejecución del LLM
         const err = data.analysis.llm_error || 'Error desconocido';
         showStatus('error', err);
         updateObserverUI(data.analysis); // Mostrar error visual
      } else {
         // Éxito
         lastAnalysis = data.analysis;
         lastContextHash = hashContext(getPatientContext()); // Update hash
         lastUpdateTime = new Date();
         updateObserverUI(data.analysis);

         const llmStatus = data.analysis.llm_status || 'connected';
         if (llmStatus === 'connected' || llmStatus === 'ok') {
            showStatus('updated', `Actualizado · ${formatTime(lastUpdateTime)}`);
         } else if (llmStatus === 'waiting') {
            showStatus('idle', 'Esperando contexto');
         } else {
            showStatus('idle', 'LLM no disponible');
         }
      }
    } else {
      showStatus('error', 'Respuesta inesperada del servidor');
      observerLoading = false; // RELEASE LOCK
    }
  } catch (err) {
    console.error(err);
    showStatus('error', 'Error de red en polling');
    observerLoading = false; // RELEASE LOCK
  }
}

async function callObserver(force = false) {
  if (currentMode !== 'sgmi') return;

  const ctx = getPatientContext();
  const currentHash = hashContext(ctx);

  // Si no hay cambios y no es forzado, no llamar al LLM
  if (!force && currentHash === lastContextHash && lastAnalysis) {
    showStatus('idle', `Actualizado · ${formatTime(lastUpdateTime)}`);
    return;
  }

  // Cancelar request anterior si existe (ahora es menos relevante pero mantenemos limpieza)
  if (currentController) {
    currentController.abort();
    currentController = null;
  }

  // Bloquear nuevas requests mientras esta está activa
  if (observerLoading) return;
  observerLoading = true;

  const isUpdate = lastAnalysis !== null;
  showLoading(isUpdate);

  // Limpiar timers previos de seguridad
  if (countdownTimer) clearTimeout(countdownTimer);
  if (slowResponseTimer) clearTimeout(slowResponseTimer);

  currentController = new AbortController();

  // Timer para mensaje lento (15s) - Feedback visual solamente
  slowResponseTimer = setTimeout(() => {
    // Solo si seguimos cargando
    if (observerLoading) {
        showStatus('analyzing', 'Modelo profundo, esperando respuesta...');
    }
  }, 15000);

  try {
    // 1. Iniciar Task (POST) - Respuesta inmediata
    const res = await fetch('/lab/observer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ patient_context: ctx, force: force }),
      signal: currentController.signal
    });

    if (!res.ok) {
      showStatus('error', `HTTP ${res.status}`);
      observerLoading = false;
      return;
    }

    const data = await res.json();
    const taskId = data.task_id;
    
    // 2. Iniciar Polling
    // Nota: observerLoading se mantiene true hasta que el polling termine
    pollObserverStatus(taskId);

  } catch (err) {
    if (err.name === 'AbortError') {
      showStatus('idle', 'Cancelado');
    } else {
      showStatus('error', 'Sin conexión al servidor');
    }
    observerLoading = false;
  } finally {
    currentController = null;
    // IMPORTANTE: NO poner observerLoading = false aquí, 
    // porque el polling sigue corriendo en background (promesa separada)
    // El polling se encarga de liberar el loading.
  }
}

// Helper para liberar loading dentro de pollObserverStatus
// Reescribimos para claridad


function pollObserver() {
  if (currentMode !== 'sgmi') return;
  if (observerLoading) return;  // No polling mientras hay análisis activo

  if (hasContextChanged()) {
    callObserver(false);
  }
  // No mostrar "sin cambios" constantemente - solo cuando hay cambios reales
}

function startPolling() {
  if (pollingInterval) clearInterval(pollingInterval);
  pollingInterval = setInterval(pollObserver, POLL_INTERVAL_MS);
}

function stopPolling() {
  if (pollingInterval) {
    clearInterval(pollingInterval);
    pollingInterval = null;
  }
}


// =============================
// AGENT ACTIVE LOGIC
// =============================

async function pollAgentStatus(taskId) {
  try {
    const res = await fetch(`/lab/agent/${taskId}`);
    if (!res.ok) {
        document.getElementById('agentStatus').innerText = "Error polling agent";
        return;
    }
    
    const data = await res.json();
    
    if (data.status === 'processing') {
       setTimeout(() => pollAgentStatus(taskId), 1000);
    } else {
       // Done
       document.getElementById('agentStatus').style.display = 'none';
       document.getElementById('btnSendAgent').disabled = false;
       
       if (data.status === 'ok') {
         if (data.task_status === 'error') {
            document.getElementById('agentOutput').innerText = JSON.stringify(data.result, null, 2); 
            document.getElementById('agentOutput').style.color = '#dc2626';
         } else {
            // Success
             const result = data.result || {};
             const answer = result.answer || JSON.stringify(result, null, 2);
             document.getElementById('agentOutput').innerText = answer;
             document.getElementById('agentOutput').style.color = '#334155';
         }
       } else {
         document.getElementById('agentOutput').innerText = "Respuesta inválida del servidor";
       }
    }
  } catch (err) {
     console.error("Polling agent error", err);
     document.getElementById('agentStatus').innerText = "Error de red";
     document.getElementById('btnSendAgent').disabled = false;
  }
}

async function callAgent() {
  const input = document.getElementById('agentInput').value;
  if (!input.trim()) return;

  // Visual Setup
  const btn = document.getElementById('btnSendAgent');
  const status = document.getElementById('agentStatus');
  const output = document.getElementById('agentOutput');

  btn.disabled = true;
  status.style.display = 'block';
  status.innerText = "Procesando...";
  output.innerText = "";
  
  try {
    // Get updated context
    const ctx = getPatientContext();
    const role = document.getElementById('agentRole').value;

    const res = await fetch('/lab/agent', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
          user_text: input, 
          role: role,
          patient_context: ctx,
          options: {} 
      })
    });

    if (!res.ok) {
      status.innerText = `Error HTTP ${res.status}`;
      btn.disabled = false;
      return;
    }

    const data = await res.json();
    const taskId = data.task_id;
    
    // Start Polling
    pollAgentStatus(taskId);

  } catch (err) {
    status.innerText = "Error de conexión";
    btn.disabled = false;
  }
}

function startCountdown() {
  if (observerLoading) return;
  
  if (countdownTimer) clearTimeout(countdownTimer);
  
  let count = 3;
  showStatus('idle', `Nuevo análisis en ${count}`);
  
  const tick = () => {
    count--;
    if (count > 0) {
      showStatus('idle', `Nuevo análisis en ${count}`);
      countdownTimer = setTimeout(tick, 1000);
    } else {
      callObserver(false);
    }
  };
  
  countdownTimer = setTimeout(tick, 1000);
}

// =============================
// Event Listeners
// =============================
const fields = [
  'patientName', 'patientAge', 'patientSex', 'clinicalPhase',
  'reasonForVisit', 'medicalHistory', 'socioCultural', 'clinicalText'
];

// Los cambios manuales disparan el countdown
fields.forEach(id => {
  const el = document.getElementById(id);
  if (el) {
    el.addEventListener('input', () => {
      startCountdown();
    });
  }
});

// Initial call + start polling + Warmup Check
document.addEventListener('DOMContentLoaded', () => {
  checkWarmupStatus();
  setTimeout(() => {
    callObserver(true);
    startPolling();
  }, 500);
});

// Warmup / Connection Status Logic
async function checkWarmupStatus() {
    const el = document.getElementById('connectionStatus');
    if (!el) return;

    try {
        const res = await fetch('/lab/status');
        const data = await res.json();
        
        if (data.warmup_done) {
            el.innerHTML = '<span style="color:#22c55e;">●</span> Motor cognitivo listo';
        } else {
            // Retry in 1s
            setTimeout(checkWarmupStatus, 1000);
        }
    } catch (e) {
        console.error("Status check failed", e);
        // Retry slower
        setTimeout(checkWarmupStatus, 3000);
    }
}
</script>

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

def run_agent_background(task_id: str, user_text: str, role: str, context: dict, options: dict):
    """Background worker para Agent Core"""
    try:
        # Llamada al núcleo cognitivo real con ROL y CONTEXTO
        result = run_llm(
            user_text=user_text,
            role=role,
            context=context,
            options=options
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
        request.options or {}
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

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
from backend.agents.observer_agent import get_observer

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
    """Contexto del paciente para el ObserverAgent."""
    patient_name: Optional[str] = None
    age: Optional[int] = None
    sex: Optional[str] = None
    medical_history: Optional[str] = None
    socio_cultural: Optional[str] = None
    reason_for_visit: Optional[str] = None
    clinical_text: Optional[str] = None
    clinical_phase: Optional[str] = "chief_complaint"  # FASE 1 por defecto


class ObserverRequest(BaseModel):
    """Request para el endpoint del observer."""
    patient_context: PatientContext
    force: bool = False  # Forzar análisis ignorando throttling


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

    /* Experimental Mode - Clinical Reasoning */
    .observer-reasoning {
      font-size: 13px;
      color: #cbd5e1;
      line-height: 1.6;
      padding: 12px;
      background: #1e293b;
      border-radius: 6px;
      border-left: 3px solid #8b5cf6;
    }

    /* Differential Diagnoses */
    .observer-diagnoses {
      list-style: none;
      padding: 0;
      margin: 0;
    }
    .observer-diagnoses li {
      padding: 10px 12px;
      background: #1e293b;
      border-radius: 6px;
      margin-bottom: 8px;
      font-size: 13px;
      color: #e5e7eb;
      border-left: 3px solid #64748b;
    }
    .observer-diagnoses li.prob-high {
      border-left-color: #22c55e;
    }
    .observer-diagnoses li.prob-med {
      border-left-color: #eab308;
    }
    .observer-diagnoses li.prob-low {
      border-left-color: #64748b;
    }
    .prob-badge {
      display: inline-block;
      font-size: 10px;
      padding: 2px 6px;
      border-radius: 4px;
      margin-left: 8px;
      text-transform: uppercase;
      background: #334155;
      color: #94a3b8;
    }
    .prob-high .prob-badge { background: #14532d; color: #86efac; }
    .prob-med .prob-badge { background: #713f12; color: #fde047; }
    .prob-low .prob-badge { background: #334155; color: #94a3b8; }
    .justification {
      font-size: 11px;
      color: #94a3b8;
      margin-top: 6px;
      font-style: italic;
    }

    /* Suggested Exams */
    .observer-exams {
      list-style: none;
      padding: 0;
      margin: 0;
    }
    .observer-exams li {
      padding: 10px 12px;
      background: #1e293b;
      border-radius: 6px;
      margin-bottom: 8px;
      font-size: 13px;
      color: #e5e7eb;
      border-left: 3px solid #3b82f6;
    }
    .exam-type {
      display: inline-block;
      font-size: 10px;
      padding: 2px 6px;
      border-radius: 4px;
      margin-left: 8px;
      text-transform: uppercase;
      background: #1e3a5f;
      color: #93c5fd;
    }

    /* Red Flags Section */
    .red-flags-section {
      background: rgba(239, 68, 68, 0.1);
      border: 1px solid rgba(239, 68, 68, 0.3);
      border-radius: 6px;
      padding: 12px;
    }
    .observer-red-flags {
      list-style: none;
      padding: 0;
      margin: 0;
    }
    .observer-red-flags li {
      padding: 8px 12px;
      background: rgba(239, 68, 68, 0.15);
      border-radius: 4px;
      margin-bottom: 6px;
      font-size: 12px;
      color: #fca5a5;
      border-left: 3px solid #ef4444;
    }

    /* Confidence Badge */
    .confidence-badge {
      display: inline-block;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
    }
    .confidence-badge.conf-high {
      background: #14532d;
      color: #86efac;
    }
    .confidence-badge.conf-med {
      background: #713f12;
      color: #fde047;
    }
    .confidence-badge.conf-low {
      background: #7f1d1d;
      color: #fca5a5;
    }

    /* Missing Info */
    .missing-info {
      font-size: 12px;
      color: #94a3b8;
      padding: 8px 12px;
      background: #1e293b;
      border-radius: 6px;
      line-height: 1.5;
    }

    /* Cognitive Section */
    .cognitive-section {
      background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
      border: 1px solid #4338ca;
      border-radius: 6px;
      padding: 12px;
    }
    .cognitive-metrics {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .metric {
      font-size: 11px;
      padding: 4px 10px;
      background: rgba(99, 102, 241, 0.2);
      color: #a5b4fc;
      border-radius: 4px;
      border: 1px solid rgba(99, 102, 241, 0.3);
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
      <div class="section-title">Texto Clínico</div>
      <div class="form-group full-width">
        <textarea id="clinicalText" rows="5" placeholder="Descripción clínica del paciente...">Paciente indica dolor de estómago con vómitos reiterados de 3 días.</textarea>
      </div>
    </div>

  </div>

  <!-- =====================
       VORTEX MODE (Placeholder)
       ===================== -->
  <div id="vortexMode" class="lab-main hidden">
    <div class="placeholder-mode">
      <div class="placeholder-icon">⚡</div>
      <div class="placeholder-text">
        <strong>Agentes Vortex</strong><br><br>
        Este modo permitirá interactuar con los agentes activos del sistema cognitivo.<br><br>
        <em>Próximamente disponible.</em>
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
        <span class="observer-title">Observer Agent</span>
        <span id="phaseLabel" class="observer-phase">Fase 1</span>
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
const POLL_INTERVAL_MS = 12000;  // 12 segundos

function getPatientContext() {
  return {
    patient_name: document.getElementById('patientName').value || null,
    age: parseInt(document.getElementById('patientAge').value) || null,
    sex: document.getElementById('patientSex').value || null,
    medical_history: document.getElementById('medicalHistory').value || null,
    socio_cultural: document.getElementById('socioCultural').value || null,
    reason_for_visit: document.getElementById('reasonForVisit').value || null,
    clinical_text: document.getElementById('clinicalText').value || null,
    clinical_phase: document.getElementById('clinicalPhase').value || 'chief_complaint'
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
  const msg = isUpdate ? 'Actualizando análisis...' : 'Analizando...';
  showStatus(isUpdate ? 'updating' : 'analyzing', msg);

  if (!lastAnalysis) {
    document.getElementById('observerContent').innerHTML = `
      <div class="observer-loading">
        <div class="spinner"></div>
        <span>${msg}</span>
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

  // Check LLM status
  const llmStatus = analysis.llm_status || 'connected';
  const isDisconnected = llmStatus !== 'connected';

  // Semaforo
  semaforo.className = 'semaforo';
  if (isDisconnected || analysis.visual_indicator === 'gray') {
    semaforo.classList.add('gray');
  } else if (analysis.visual_indicator === 'yellow') {
    semaforo.classList.add('yellow');
  } else if (analysis.visual_indicator === 'red') {
    semaforo.classList.add('red');
  }

  // Phase label - EXPERIMENTAL mode
  if (isDisconnected) {
    phaseLabel.textContent = 'LLM: OFF';
    phaseLabel.style.background = '#7f1d1d';
    phaseLabel.style.color = '#fca5a5';
  } else {
    phaseLabel.textContent = 'EXPERIMENTAL';
    phaseLabel.style.background = '#7c3aed';
    phaseLabel.style.color = '#e9d5ff';
  }

  let html = '';

  // Error state
  if (isDisconnected) {
    html = `
      <div class="observer-section">
        <div class="observer-label">Estado</div>
        <div class="observer-summary" style="color: #f87171;">
          ${escapeHtml(analysis.llm_error || 'LLM no disponible')}
        </div>
      </div>
      <div class="observer-empty" style="margin-top: 16px;">
        Verifica que Ollama esté corriendo:<br>
        <code style="background:#1e293b;padding:4px 8px;border-radius:4px;margin-top:8px;display:inline-block;">ollama serve</code>
      </div>
    `;
    content.innerHTML = html;
    return;
  }

  // Summary
  if (analysis.summary) {
    html += `
      <div class="observer-section">
        <div class="observer-label">Resumen</div>
        <div class="observer-summary">${escapeHtml(analysis.summary)}</div>
      </div>
    `;
  }

  // Clinical Reasoning
  if (analysis.clinical_reasoning) {
    html += `
      <div class="observer-section">
        <div class="observer-label">Razonamiento Clínico</div>
        <div class="observer-reasoning">${escapeHtml(analysis.clinical_reasoning)}</div>
      </div>
    `;
  }

  // Differential Diagnoses
  const diagnoses = Array.isArray(analysis.differential_diagnoses)
    ? analysis.differential_diagnoses : [];
  if (diagnoses.length > 0) {
    html += `
      <div class="observer-section">
        <div class="observer-label">Diagnósticos Diferenciales</div>
        <ul class="observer-diagnoses">
          ${diagnoses.map(d => {
            if (typeof d === 'object' && d.diagnosis) {
              const prob = d.probability || '';
              const probClass = prob.toLowerCase().includes('alta') ? 'prob-high' :
                               prob.toLowerCase().includes('media') ? 'prob-med' : 'prob-low';
              return `<li class="${probClass}">
                <strong>${escapeHtml(d.diagnosis)}</strong>
                ${prob ? `<span class="prob-badge">${escapeHtml(prob)}</span>` : ''}
                ${d.justification ? `<div class="justification">${escapeHtml(d.justification)}</div>` : ''}
              </li>`;
            }
            return `<li>${escapeHtml(String(d))}</li>`;
          }).join('')}
        </ul>
      </div>
    `;
  }

  // Suggested Exams
  const exams = Array.isArray(analysis.suggested_exams) ? analysis.suggested_exams : [];
  if (exams.length > 0) {
    html += `
      <div class="observer-section">
        <div class="observer-label">Estudios Sugeridos</div>
        <ul class="observer-exams">
          ${exams.map(e => {
            if (typeof e === 'object' && e.exam) {
              return `<li>
                <strong>${escapeHtml(e.exam)}</strong>
                ${e.type ? `<span class="exam-type">${escapeHtml(e.type)}</span>` : ''}
                ${e.justification ? `<div class="justification">${escapeHtml(e.justification)}</div>` : ''}
              </li>`;
            }
            return `<li>${escapeHtml(String(e))}</li>`;
          }).join('')}
        </ul>
      </div>
    `;
  }

  // Red Flags
  const redFlags = Array.isArray(analysis.red_flags) ? analysis.red_flags : [];
  if (redFlags.length > 0) {
    html += `
      <div class="observer-section red-flags-section">
        <div class="observer-label" style="color:#f87171;">Banderas Rojas</div>
        <ul class="observer-red-flags">
          ${redFlags.map(f => `<li>${escapeHtml(String(f))}</li>`).join('')}
        </ul>
      </div>
    `;
  }

  // Confidence & Missing Info
  if (analysis.confidence_level || (analysis.missing_info && analysis.missing_info.length > 0)) {
    html += `<div class="observer-section">`;
    if (analysis.confidence_level) {
      const confClass = analysis.confidence_level.toLowerCase().includes('alta') ? 'conf-high' :
                       analysis.confidence_level.toLowerCase().includes('media') ? 'conf-med' : 'conf-low';
      html += `
        <div class="observer-label">Nivel de Certeza</div>
        <div class="confidence-badge ${confClass}">${escapeHtml(analysis.confidence_level)}</div>
      `;
    }
    const missing = Array.isArray(analysis.missing_info) ? analysis.missing_info : [];
    if (missing.length > 0) {
      html += `
        <div class="observer-label" style="margin-top:12px;">Información Faltante</div>
        <div class="missing-info">${missing.map(m => escapeHtml(String(m))).join(' · ')}</div>
      `;
    }
    html += `</div>`;
  }

  // Cognitive Behavior (for research)
  if (analysis.cognitive_behavior && Object.keys(analysis.cognitive_behavior).length > 0) {
    const cb = analysis.cognitive_behavior;
    html += `
      <div class="observer-section cognitive-section">
        <div class="observer-label">Comportamiento Cognitivo</div>
        <div class="cognitive-metrics">
          <span class="metric">Incertidumbre: ${cb.uncertainty_markers?.length || 0}</span>
          <span class="metric">Fabricación: ${cb.fabrication_markers?.length || 0}</span>
          <span class="metric">${cb.confidence_assessment || 'N/A'}</span>
        </div>
      </div>
    `;
  }

  if (!html) {
    html = '<div class="observer-empty">Sin observaciones.</div>';
  }

  content.innerHTML = html;
}

async function callObserver(force = false) {
  if (observerLoading || currentMode !== 'sgmi') return;

  const ctx = getPatientContext();
  const currentHash = hashContext(ctx);

  // Si no hay cambios y no es forzado, no llamar al LLM
  if (!force && currentHash === lastContextHash && lastAnalysis) {
    showStatus('idle', `Sin cambios · ${formatTime(new Date())}`);
    return;
  }

  observerLoading = true;
  const isUpdate = lastAnalysis !== null;
  showLoading(isUpdate);

  try {
    const res = await fetch('/lab/observer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ patient_context: ctx, force: force })
    });

    if (res.ok) {
      const data = await res.json();
      if (data.status === 'ok' && data.analysis) {
        lastAnalysis = data.analysis;
        lastContextHash = currentHash;
        lastUpdateTime = new Date();
        updateObserverUI(data.analysis);
        showStatus('updated', `Actualizado · ${formatTime(lastUpdateTime)}`);
      }
    } else {
      showStatus('error', 'Error de conexión');
      if (!lastAnalysis) {
        updateObserverUI({
          summary: 'Error de conexión con el observador.',
          patterns: [],
          visual_indicator: 'gray',
          notes: []
        });
      }
    }
  } catch (err) {
    showStatus('error', 'Error: ' + err.message);
    if (!lastAnalysis) {
      updateObserverUI({
        summary: 'Error: ' + err.message,
        patterns: [],
        visual_indicator: 'gray',
        notes: []
      });
    }
  } finally {
    observerLoading = false;
  }
}

function pollObserver() {
  if (currentMode !== 'sgmi') return;

  if (hasContextChanged()) {
    callObserver(false);
  } else {
    showStatus('idle', `Sin cambios · ${formatTime(new Date())}`);
  }
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
// Event Listeners
// =============================
const fields = [
  'patientName', 'patientAge', 'patientSex', 'clinicalPhase',
  'reasonForVisit', 'medicalHistory', 'socioCultural', 'clinicalText'
];

// Los cambios manuales disparan análisis inmediato (con debounce)
let inputDebounce = null;
fields.forEach(id => {
  const el = document.getElementById(id);
  if (el) {
    el.addEventListener('input', () => {
      if (inputDebounce) clearTimeout(inputDebounce);
      inputDebounce = setTimeout(() => callObserver(false), 1500);
    });
  }
});

// Initial call + start polling
document.addEventListener('DOMContentLoaded', () => {
  setTimeout(() => {
    callObserver(true);
    startPolling();
  }, 500);
});
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
# Observer Agent Endpoint
# =========================

@router.post("/lab/observer")
async def observer_analyze(request: ObserverRequest):
    """
    Endpoint para el ObserverAgent.

    Analiza el contexto clínico de forma pasiva.
    NO conversa, NO recomienda, NO diagnostica.

    Incluye throttling para evitar llamadas excesivas:
    - Mínimo 2 segundos entre análisis
    - O cambio de > 20 caracteres en el contexto
    """
    try:
        observer = get_observer()
        result = observer.analyze(
            patient_context=request.patient_context.model_dump(),
            force=request.force
        )
        return JSONResponse(content={
            "status": "ok",
            "analysis": result
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": "OBSERVER_ERROR",
                "detail": str(e),
                "analysis": {
                    "summary": "Error en el análisis.",
                    "patterns": [],
                    "visual_indicator": "green",
                    "notes": [f"Error: {str(e)}"]
                }
            }
        )

from datetime import datetime
from sqlalchemy.orm import Session
from backend.app.db.session import SessionLocal, engine
from backend.app.models.agent_prompts import AgentPrompt, Base

# Asegurar que las tablas existan
Base.metadata.create_all(bind=engine)

def seed_prompts():
    """
    Inserta prompts iniciales obligatorios según definición del usuario.
    """
    db: Session = SessionLocal()
    
    # Prompts Definitions (Final Specs)
    prompts_data = [
        # --- CLINICAL ---
        {
            "role": "clinical", "mode": "work", "type": "baseline",
            "content": (
                "Eres un médico clínico senior con amplia experiencia hospitalaria y ambulatoria.\n"
                "Piensas en términos de fisiopatología, diagnóstico diferencial y evaluación de riesgo.\n"
                "Analizas la información de forma estructurada, prudente y técnica.\n"
                "No realizas diagnósticos definitivos ni prescribes tratamientos cerrados.\n"
                "Tu rol es apoyar el razonamiento clínico profesional, no reemplazar al médico tratante.\n"
                "Mantén siempre un tono profesional, objetivo y basado en evidencia."
            )
        },
        {
            "role": "clinical", "mode": "work", "type": "extended",
            "content": (
                "Organiza tu análisis en:\n"
                "- Resumen clínico relevante\n"
                "- Hipótesis diagnósticas principales\n"
                "- Diagnósticos diferenciales importantes\n"
                "- Señales de alerta o riesgo\n"
                "- Próximos pasos sugeridos (sin prescripción)\n"
                "Usa lenguaje técnico, preciso y profesional."
            )
        },
        # --- COMMERCIAL ---
        {
            "role": "commercial", "mode": "work", "type": "baseline",
            "content": (
                "Eres un consultor comercial senior B2B con experiencia en servicios tecnológicos y de salud.\n"
                "Explicas soluciones con claridad, foco en valor y sin promesas irreales."
            )
        },
        {
            "role": "commercial", "mode": "work", "type": "extended",
            "content": (
                "Estructura tu respuesta en:\n"
                "- Comprensión del problema\n"
                "- Propuesta de valor\n"
                "- Beneficios concretos\n"
                "- Próximos pasos\n"
                "Lenguaje profesional y persuasivo sin presión."
            )
        },
        # --- SUPPORT ---
        {
            "role": "support", "mode": "work", "type": "baseline",
            "content": (
                "Eres un ingeniero de soporte técnico senior.\n"
                "Ayudas a diagnosticar y resolver problemas de forma ordenada, segura y reproducible."
            )
        },
        {
            "role": "support", "mode": "work", "type": "extended",
            "content": (
                "Estructura tu respuesta en:\n"
                "- Diagnóstico probable\n"
                "- Pasos de verificación\n"
                "- Soluciones posibles\n"
                "- Recomendaciones preventivas"
            )
        },
         # --- PERSONAL ---
        {
            "role": "personal", "mode": "work", "type": "baseline",
            "content": (
                "Eres un asistente personal profesional, claro y confiable.\n"
                "Ayudas a organizar ideas, decisiones e información práctica."
            )
        },
        {
            "role": "personal", "mode": "work", "type": "extended",
            "content": (
                "Responde de forma concisa, estructurada y útil.\n"
                "Aclara ambigüedades antes de asumir."
            )
        },
    ]

    print("🌱 Seeding Agent Prompts (Upsert Mode)...")
    
    for p in prompts_data:
        # Upsert Logic
        existing = db.query(AgentPrompt).filter(
            AgentPrompt.agent_role == p["role"],
            AgentPrompt.mode == p["mode"],
            AgentPrompt.prompt_type == p["type"]
        ).first()
        
        if existing:
            existing.content = p["content"]
            existing.updated_at = datetime.utcnow()
            print(f"   [~] Updated: {p['role']} / {p['type']}")
        else:
            new_prompt = AgentPrompt(
                agent_role=p["role"],
                mode=p["mode"],
                prompt_type=p["type"],
                content=p["content"],
                active=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.add(new_prompt)
            print(f"   [+] Created: {p['role']} / {p['type']}")
            
    db.commit()
    db.close()
    print("✅ Seeding Complete.")

if __name__ == "__main__":
    seed_prompts()

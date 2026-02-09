from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from backend.app.db.base import Base

class AgentPrompt(Base):
    """
    Tabla de Prompts para LAB y Producción.
    Permite configurar el comportamiento de los agentes sin redeploy.
    """
    __tablename__ = "agent_prompts"

    id = Column(Integer, primary_key=True, index=True)
    agent_role = Column(String, nullable=False, index=True)  # clinical | commercial | support | personal
    mode = Column(String, nullable=False, default="work")    # work | life | audit
    prompt_type = Column(String, nullable=False)             # baseline | extended
    content = Column(String, nullable=False)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<AgentPrompt(role={self.agent_role}, type={self.prompt_type}, active={self.active})>"

from sqlalchemy.orm import Session
from backend.app.models.agent_prompts import AgentPrompt

class PromptsService:
    @staticmethod
    def get_prompt(db: Session, role: str, mode: str, prompt_type: str) -> str:
        """
        Obtiene el contenido del prompt activo desde la BD.
        Si no existe, retorna string vacío (o podría lanzar excepción según diseño).
        """
        prompt = (
            db.query(AgentPrompt)
            .filter(
                AgentPrompt.agent_role == role,
                AgentPrompt.mode == mode,
                AgentPrompt.prompt_type == prompt_type,
                AgentPrompt.active == True
            )
            .first()
        )
        return prompt.content if prompt else ""

    @staticmethod
    def get_system_prompts(db: Session, role: str, mode: str, raw: bool) -> list[str]:
        """
        Retorna la lista de prompts de sistema a inyectar.
        Regla RAW:
          - RAW=True: Solo Baseline
          - RAW=False: Baseline + Extended
        """
        prompts = []
        
        # 1. Baseline (Obligatorio)
        baseline = PromptsService.get_prompt(db, role, mode, "baseline")
        if not baseline:
             # Fallback si no existe en DB, para respetar "Baseline SIEMPRE"
             baseline = f"You are a helpful {role} assistant. (Fallback Mode)"
        
        prompts.append(baseline)
            
        # 2. Extended (Solo si no es RAW)
        if not raw:
            extended = PromptsService.get_prompt(db, role, mode, "extended")
            if extended:
                prompts.append(extended)
                
        return prompts

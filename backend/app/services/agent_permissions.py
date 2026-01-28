class AgentPermissionError(Exception):
    pass


def resolve_agent(
    requested_agent: str | None,
    allowed_agents: list[str],
    default_agent: str,
) -> str:
    """
    Decide el agente final de forma SEGURA.
    El LLM no participa aquí.
    """

    if requested_agent is None:
        return default_agent

    if requested_agent not in allowed_agents:
        raise AgentPermissionError(
            f"Acceso denegado al agente '{requested_agent}'"
        )

    return requested_agent

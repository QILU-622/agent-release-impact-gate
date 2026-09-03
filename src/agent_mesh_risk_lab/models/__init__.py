"""Provider-neutral and local real-LLM adapters."""

from .base import AgentDecision, AgentModel, AgentObservation
from .ollama_adapter import OllamaAgentModel

__all__ = ["AgentDecision", "AgentModel", "AgentObservation", "OllamaAgentModel"]

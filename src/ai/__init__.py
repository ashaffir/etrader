"""Azure AI Foundry / Azure OpenAI integration."""

from .azure_client import AzureFoundryClient, AzureUnavailable
from .prompts import build_decision_prompt, build_universe_rotation_prompt

__all__ = [
    "AzureFoundryClient",
    "AzureUnavailable",
    "build_decision_prompt",
    "build_universe_rotation_prompt",
]

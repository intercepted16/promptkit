"""PromptKit public interface."""

from src.py_promptkit.errors import (
    PromptConfigError,
    PromptKitError,
    PromptProviderError,
    PromptValidationError,
)
from src.py_promptkit.loader import PromptLoader
from src.py_promptkit.models.clients import (
    LLMClient,
    LLMResponse,
    ToolSpecification,
)
from src.py_promptkit.models.config import ModelConfig, PromptDefinition, ToolConfig
from src.py_promptkit.models.hooks import HookContext, HookManager, PromptHook
from src.py_promptkit.runner import PromptCacheProtocol, PromptRunner

__all__ = [
    # ClientFactory removed; clients are registered as instances
    "HookContext",
    "HookManager",
    "LLMClient",
    "LLMResponse",
    "ModelConfig",
    "PromptCacheProtocol",
    "PromptConfigError",
    "PromptDefinition",
    "PromptKitError",
    "PromptLoader",
    "PromptProviderError",
    "PromptRunner",
    "PromptValidationError",
    "PromptHook",
    "ToolConfig",
    "ToolSpecification",
]

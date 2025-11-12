"""PromptKit public interface."""

from src.promptkit.errors import (
    PromptConfigError,
    PromptKitError,
    PromptProviderError,
    PromptValidationError,
)
from src.promptkit.loader import PromptLoader
from src.promptkit.models.clients import (
    LLMClient,
    LLMResponse,
    ToolSpecification,
)
from src.promptkit.models.config import ModelConfig, PromptDefinition, ToolConfig
from src.promptkit.models.hooks import HookContext, HookManager, PromptHook
from src.promptkit.runner import PromptCacheProtocol, PromptRunner

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

"""Custom exceptions raised by the PromptKit library."""

from __future__ import annotations


class PromptKitError(Exception):
    """Base exception for all PromptKit-related errors."""


class PromptConfigError(PromptKitError):
    """Raised when a prompts configuration file is invalid."""


class PromptValidationError(PromptKitError):
    """Raised when provided prompt variables fail validation."""


class PromptProviderError(PromptKitError):
    """Raised when no suitable LLM provider client is available."""


class MCPError(PromptKitError):
    """Raised when an error occurs while interacting with MCP (Model Context Protocol)."""


class ModelRequestError(PromptKitError):
    """Raised when a model request fails."""

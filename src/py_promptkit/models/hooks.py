"""Hook system allowing extensions to observe prompt execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableSequence, Sequence

from src.py_promptkit.models.clients import LLMResponse, ToolSpecification
from src.py_promptkit.models.config import ModelConfig


@dataclass(frozen=True)
class HookContext:
    """Context shared with hooks before and after a prompt run."""

    prompt_name: str
    model: ModelConfig
    variables: Mapping[str, str]
    rendered_prompt: str
    tools: Sequence[ToolSpecification] | None


class PromptHook:
    """Base class for hooks that intercept prompt execution."""

    def before_run(self, context: HookContext) -> None:
        """Called before a prompt is executed."""

    def after_run(self, context: HookContext, response: LLMResponse) -> None:
        """Called after a prompt is executed successfully."""

    def on_error(self, context: HookContext, error: Exception) -> None:
        """Called when prompt execution raises an exception."""


class HookManager:
    """Lightweight dispatcher that fans out hook events."""

    def __init__(self, hooks: Iterable[PromptHook] | None = None) -> None:
        """Initialize the manager with an optional hook collection."""
        self._hooks: MutableSequence[PromptHook] = list(hooks or [])

    def register(self, hook: PromptHook) -> None:
        """Register a hook for future events."""
        self._hooks.append(hook)

    def before_run(self, context: HookContext) -> None:
        """Invoke `before_run` on every registered hook."""
        for hook in list(self._hooks):
            hook.before_run(context)

    def after_run(self, context: HookContext, response: LLMResponse) -> None:
        """Invoke `after_run` on every registered hook."""
        for hook in list(self._hooks):
            hook.after_run(context, response)

    def on_error(self, context: HookContext, error: Exception) -> None:
        """Invoke `on_error` on every registered hook."""
        for hook in list(self._hooks):
            hook.on_error(context, error)

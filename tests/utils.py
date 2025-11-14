"""Shared testing utilities."""

from __future__ import annotations
from typing import Iterator
from src.py_promptkit.models.clients import LLMClient, LLMResponse, ToolSpecification
from src.py_promptkit.models.hooks import HookContext, PromptHook
from pathlib import Path
from dotenv import find_dotenv, dotenv_values


CALL_COUNTER = {"generate_calls": 0}


class EchoClient(LLMClient):
    """Mock client that echoes back the prompt and exposes model used."""

    def __init__(self) -> None:
        """Initialize the echo client (no model/temperature at construction)."""
        self.supports_tools = False

    def generate(
        self,
        prompt: str,
        tools: list[ToolSpecification] | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Generate a response by echoing the prompt and incrementing a counter."""
        CALL_COUNTER["generate_calls"] += 1
        effective_model = model or "(unknown)"
        return {"reasoning": f"echo for model {effective_model}", "output": prompt}

    def stream_generate(
        self,
        prompt: str,
        tools: list[ToolSpecification] | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ):
        """Stream generate a response by echoing the prompt in one chunk."""
        yield prompt

    def close(self) -> None:
        """No resources to clean up for the echo client."""
        self.close_called = True


class FailingClient(LLMClient):
    """Client that always raises to test on_error hooks."""

    supports_tools = False

    def __init__(self) -> None:
        """Initialize the failing client (no model/temperature at construction)."""

    def generate(
        self,
        prompt: str,
        tools: list[ToolSpecification] | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:  # pragma: no cover - exercised by hook path
        """Always raise a runtime error to trigger error hooks."""
        raise RuntimeError("boom")

    def stream_generate(
        self,
        prompt: str,
        tools: list[ToolSpecification] | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        """Yield nothing; not used in error tests."""
        return iter(())


class RecordingHook(PromptHook):
    """Hook that records calls for assertions."""

    def __init__(self) -> None:
        """Create internal storage for recorded hook events."""
        self.before_calls: list[HookContext] = []
        self.after_calls: list[tuple[HookContext, LLMResponse]] = []
        self.error_calls: list[tuple[HookContext, Exception]] = []

    def before_run(self, context: HookContext) -> None:
        """Record the context before a run starts."""
        self.before_calls.append(context)

    def after_run(self, context: HookContext, response: LLMResponse) -> None:
        """Record the context and response after a successful run."""
        self.after_calls.append((context, response))

    def on_error(self, context: HookContext, error: Exception) -> None:
        """Record the context and error when a run fails."""
        self.error_calls.append((context, error))


# some useful globals
TEST_RESOURCES_DIR = Path(__file__).parent / "resources"

# dotenv
dotenv = find_dotenv(".env.test")

if not dotenv:
    raise RuntimeError("Could not find .env.test file")


# do this just so its easier to handle downstream.
secrets: dict[str, str] = {}
# load all non-None values from dotenv into secrets
for k, v in dotenv_values(dotenv).items():
    if v is None:
        continue
    secrets[k] = v


if not secrets:
    raise RuntimeError("Could not load .env.test values")

__all__ = [
    "EchoClient",
    "FailingClient",
    "RecordingHook",
    "TEST_RESOURCES_DIR",
]

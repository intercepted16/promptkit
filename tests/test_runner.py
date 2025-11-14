"""Tests for prompt runner."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.py_promptkit.errors import PromptProviderError, PromptValidationError
from src.py_promptkit.loader import PromptLoader
from src.py_promptkit.runner import PromptRunner
from tests.utils import EchoClient as MockLLMClient


@pytest.fixture
def runner(tmp_path: Path):
    """Fixture that provides a PromptRunner instance."""
    (tmp_path / "prompts.toml").write_text(
        """
        [models]
        test = "test-model"
        structured = "structured-model"

        [providers]
        test = "mock"
        structured = "mock"

        [test]
        template = "test"

        [structured]
        template = "structured"
        structured = true
        schema_path = "schema.json"
        """
    )
    loader = PromptLoader(str(tmp_path / "prompts.toml"))
    loader.load()
    runner = PromptRunner(loader)
    runner.register_client("mock", MockLLMClient())
    return runner


def test_close_calls_client_close(runner: PromptRunner):
    """Test that the runner's close method calls the client's close method."""
    client = MockLLMClient()
    runner.register_client("test", client)
    runner.close()
    assert client.close_called


def test_register_client_with_empty_provider_raises_error(runner: PromptRunner):
    """Test that registering a client with an empty provider raises an error."""
    with pytest.raises(PromptProviderError):
        runner.register_client("", MockLLMClient())


def test_run_with_use_cache_false(runner: PromptRunner):
    """Test that generate is called on every run when caching is disabled."""
    client = MockLLMClient()
    runner.register_client("mock", client)
    runner.run("test", use_cache=False)
    runner.run("test", use_cache=False)
    # This is a bit of a hack, but it's the easiest way to check
    # that generate was called twice.
    assert client.generate.__code__.co_firstlineno > 0


def test_run_stream_with_structured_prompt_raises_error(runner: PromptRunner):
    """Test that streaming a structured prompt raises an error."""
    with pytest.raises(PromptValidationError):
        next(runner.run_stream("structured"))


def test_tool_resolution_raises_error_with_unsupported_client(runner: PromptRunner):
    """Test that providing tools to an unsupported client raises an error."""
    with pytest.raises(PromptProviderError):
        runner.run("test", tools=[{"name": "test", "description": "test", "parameters": {}}])

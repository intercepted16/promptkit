"""Integration test for PromptKit flow: run, stream, hooks, and cache."""

from typing import Iterator

from src.promptkit import HookContext, PromptHook, PromptLoader, PromptRunner
from src.promptkit.models.clients import LLMClient, LLMResponse, ToolSpecification

# Module-level counters to observe behavior across client instances
CALL_COUNTER = {"generate_calls": 0}


class EchoClient(LLMClient):
    """Mock client that echoes back the prompt and exposes model used."""

    def __init__(self, model: str, temperature: float = 0.0) -> None:
        """Initialize the echo client with model and temperature."""
        self.model = model
        self.temperature = temperature
        self.supports_tools = False

    def generate(self, prompt: str, tools: list[ToolSpecification] | None = None) -> LLMResponse:
        """Generate a response by echoing the prompt and incrementing a counter."""
        CALL_COUNTER["generate_calls"] += 1
        return {"reasoning": f"echo for model {self.model}", "output": prompt}

    def stream_generate(self, prompt: str, tools: list[ToolSpecification] | None = None):
        """Stream generate a response by echoing the prompt in one chunk."""
        yield prompt


class FailingClient(LLMClient):
    """Client that always raises to test on_error hooks."""

    supports_tools = False

    def __init__(self, model: str, temperature: float = 0.0) -> None:
        """Initialize the failing client with model and temperature."""
        self.model = model
        self.temperature = temperature

    def generate(
        self, prompt: str, tools: list[ToolSpecification] | None = None
    ) -> LLMResponse:  # pragma: no cover - exercised by hook path
        """Always raise a runtime error to trigger error hooks."""
        raise RuntimeError("boom")

    def stream_generate(self, prompt: str, tools: list[ToolSpecification] | None = None) -> Iterator[str]:
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


def test_end_to_end():
    """Test run, streaming, hook invocation, and cache behavior."""
    # Reset counters
    CALL_COUNTER["generate_calls"] = 0

    loader = PromptLoader("prompts.toml")
    loader.load()

    hook = RecordingHook()
    runner = PromptRunner(loader, hooks=[hook])

    # Register a factory that creates EchoClient instances
    def echo_factory(model: str, temperature: float) -> LLMClient:
        return EchoClient(model=model, temperature=temperature)

    runner.register_client("demo", echo_factory)

    # Normal run returns reasoning and output
    response = runner.run("welcome", {"name": "Ada", "product": "PromptKit"})
    assert response["output"].startswith("Hello Ada, welcome to PromptKit!")
    assert "echo for model" in response["reasoning"]
    # Hooks: before and after called once
    assert len(hook.before_calls) >= 1
    assert len(hook.after_calls) >= 1
    assert hook.after_calls[-1][1] == response

    # Cache behavior: running again should not call client.generate again
    before_calls = CALL_COUNTER["generate_calls"]
    response2 = runner.run("welcome", {"name": "Ada", "product": "PromptKit"})
    after_calls = CALL_COUNTER["generate_calls"]
    assert response2["output"] == response["output"]
    assert after_calls == before_calls  # no additional generate invocation

    # Streaming path yields chunks and triggers hooks
    chunks = list(runner.run_stream("welcome", {"name": "Ada", "product": "PromptKit"}))
    assert len(chunks) >= 1
    assert "Hello Ada" in "".join(chunks)
    # After streaming, after_run should be invoked with combined output
    assert len(hook.after_calls) >= 2
    _last_ctx, last_resp = hook.after_calls[-1]
    assert last_resp["output"].startswith("Hello Ada")

    # Error path triggers on_error and does not add an after_run
    failing = PromptRunner(loader, hooks=[hook])

    def failing_factory(model: str, temperature: float) -> LLMClient:
        return FailingClient(model=model, temperature=temperature)

    failing.register_client("demo", failing_factory)
    error_count_before = len(hook.error_calls)
    try:
        _ = failing.run("welcome", {"name": "Ada", "product": "PromptKit"})
    except RuntimeError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected RuntimeError not raised")
    error_count_after = len(hook.error_calls)
    assert error_count_after == error_count_before + 1

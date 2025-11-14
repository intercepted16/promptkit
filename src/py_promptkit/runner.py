"""Prompt execution orchestration for PromptKit."""

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType
from typing import Dict, Iterator, List, Mapping, MutableMapping, Optional, Protocol, Sequence, runtime_checkable

from src.py_promptkit.errors import PromptProviderError, PromptValidationError
from src.py_promptkit.loader import PromptDefinition, PromptLoader
from src.py_promptkit.models.clients import (
    LLMClient,
    LLMResponse,
    ToolSpecification,
)
from src.py_promptkit.models.hooks import HookContext, HookManager, PromptHook


@dataclass(frozen=True)
class _ExecutionPlan:
    """Aggregated information required to execute a prompt."""

    cache_key: str
    context: HookContext
    client: LLMClient
    definition: PromptDefinition
    rendered_prompt: str
    variables: Dict[str, str]
    tools: Optional[Sequence[ToolSpecification]]


@runtime_checkable
class PromptCacheProtocol(Protocol):
    """Protocol that custom caches must implement.

    Implementations are free to provide any backing store (in-memory, redis,
    filesystem, etc.). The runner will only use the methods below when a cache
    instance is explicitly provided to `PromptRunner`.
    """

    def build_key(
        self,
        prompt: str,
        model_name: str,
        provider: str,
        temperature: float,
        variables: Mapping[str, str],
    ) -> str:
        """Return a deterministic cache key for the given arguments."""
        ...

    def get(self, key: str) -> Optional[str]:
        """Return cached value if present."""
        ...

    def set(self, key: str, value: str) -> None:
        """Store a cache entry."""
        ...


class PromptRunner:
    """High-level orchestrator that renders and executes prompts."""

    def __init__(
        self,
        loader: PromptLoader,
        *,
        hooks: Optional[Sequence[PromptHook]] = None,
        cache: Optional[PromptCacheProtocol] = None,
    ) -> None:
        """Create a PromptRunner bound to a particular loader."""
        self.loader = loader
        # No default cache is provided — users must explicitly supply a
        # cache implementation if they want caching behaviour.
        self.cache = cache
        self.hooks = HookManager(hooks)
        self._factories: MutableMapping[str, LLMClient] = {}

    def __enter__(self) -> PromptRunner:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager and clean up all registered clients."""
        self.close()

    def close(self) -> None:
        """Close all registered LLM clients that support cleanup."""
        for _provider, client in list(self._factories.items()):
            if hasattr(client, "close") and callable(client.close):
                try:
                    client.close()
                except Exception:
                    # Silently ignore cleanup errors
                    pass

    def register_client(self, provider: str, client: LLMClient) -> None:
        """Register a client factory for a provider key."""
        provider_key = provider.strip().lower()
        if not provider_key:
            raise PromptProviderError("Provider key must be a non-empty string.")
        self._factories[provider_key] = client

    def run(
        self,
        prompt_name: str,
        variables: Mapping[str, object] | None = None,
        *,
        tools: Optional[Sequence[ToolSpecification]] = None,
        use_cache: bool = True,
    ) -> LLMResponse:
        """Execute a prompt using the registered provider client."""
        plan = self._build_execution_plan(prompt_name, variables or {}, tools, streaming=False)

        self.hooks.before_run(plan.context)

        # Only attempt to read from the cache if the caller requested caching
        # and a cache implementation was explicitly provided to the runner.
        if use_cache and self.cache is not None:
            cached = self.cache.get(plan.cache_key)
            if cached is not None:
                return {"reasoning": "", "output": cached}

        try:
            response = plan.client.generate(
                plan.rendered_prompt,
                tools=self._tools_argument(plan.tools),
                model=plan.definition.model.name,
                temperature=plan.definition.model.temperature,
            )
        except Exception as exc:
            self.hooks.on_error(plan.context, exc)
            raise

        # Only persist to cache when a cache implementation was provided and
        # the caller requested caching.
        if use_cache and self.cache is not None:
            self.cache.set(plan.cache_key, response["output"])

        self.hooks.after_run(plan.context, response)
        return response

    def run_stream(
        self,
        prompt_name: str,
        variables: Mapping[str, object] | None = None,
        *,
        tools: Optional[Sequence[ToolSpecification]] = None,
    ) -> Iterator[str]:
        """Stream the prompt output if supported by the provider client."""
        plan = self._build_execution_plan(prompt_name, variables or {}, tools, streaming=True)

        self.hooks.before_run(plan.context)
        collected: List[str] = []
        try:
            iterator = plan.client.stream_generate(
                plan.rendered_prompt,
                tools=self._tools_argument(plan.tools),
                model=plan.definition.model.name,
                temperature=plan.definition.model.temperature,
            )
            for chunk in iterator:
                collected.append(chunk)
                yield chunk
        except Exception as exc:  # pragma: no cover - streaming errors bubble through
            self.hooks.on_error(plan.context, exc)
            raise
        else:
            response: LLMResponse = {"reasoning": "", "output": "".join(collected)}

            self.hooks.after_run(plan.context, response)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_execution_plan(
        self,
        prompt_name: str,
        variables: Mapping[str, object],
        tools: Optional[Sequence[ToolSpecification]],
        *,
        streaming: bool,
    ) -> _ExecutionPlan:
        definition = self.loader.get(prompt_name)
        if streaming and definition.model.structured:
            raise PromptValidationError("Streaming is not supported for structured prompts.")

        rendered_prompt, normalized_variables = definition.render_with(variables)
        client = self._resolve_client(definition)
        resolved_tools = self._resolve_tools(definition, client, tools)

        cache_key = ""
        # Build a cache key only when not streaming and when a cache
        # implementation was provided. If no cache was given, caching is
        # disabled and an empty cache_key is used.
        if not streaming and self.cache is not None:
            cache_key = self.cache.build_key(
                prompt=prompt_name,
                model_name=definition.model.name,
                provider=definition.model.provider,
                temperature=definition.model.temperature,
                variables=normalized_variables,
            )

        context = HookContext(
            prompt_name=prompt_name,
            model=definition.model,
            variables=normalized_variables,
            rendered_prompt=rendered_prompt,
            tools=resolved_tools,
        )

        return _ExecutionPlan(
            cache_key=cache_key,
            context=context,
            client=client,
            definition=definition,
            rendered_prompt=rendered_prompt,
            variables=normalized_variables,
            tools=resolved_tools,
        )

    def _resolve_client(self, definition: PromptDefinition) -> LLMClient:
        provider = definition.model.provider
        key = provider.strip().lower()
        if key not in self._factories:
            raise PromptProviderError(f"No LLM client factory registered for provider '{provider}'.")
        return self._factories[key]

    @staticmethod
    def _resolve_tools(
        definition: PromptDefinition,
        client: LLMClient,
        override_tools: Optional[Sequence[ToolSpecification]],
    ) -> Optional[Sequence[ToolSpecification]]:
        if override_tools is not None:
            if override_tools and not client.supports_tools:
                model_name = definition.model.name
                raise PromptProviderError(f"Client for model '{model_name}' does not support tool execution.")
            return override_tools
        configured = definition.build_tools()
        if configured and not client.supports_tools:
            model_name = definition.model.name
            raise PromptProviderError(f"Client for model '{model_name}' does not support configured tools.")
        return configured

    @staticmethod
    def _tools_argument(
        tools: Optional[Sequence[ToolSpecification]],
    ) -> Optional[List[ToolSpecification]]:
        if tools is None:
            return None
        return list(tools)

    @staticmethod
    def _extract_output(response: LLMResponse) -> str:
        return response["output"]

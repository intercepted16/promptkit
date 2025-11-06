"""Prompt execution orchestration for PromptKit."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Dict, Iterator, List, Mapping, MutableMapping, Optional, Sequence

from src.promptkit.errors import PromptProviderError, PromptValidationError
from src.promptkit.loader import PromptDefinition, PromptLoader
from src.promptkit.models.clients import (
    ClientFactory,
    LLMClient,
    LLMResponse,
    ToolSpecification,
)
from src.promptkit.models.hooks import HookContext, HookManager, PromptHook


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


class PromptCache:
    """In-memory cache keyed by prompt parameters."""

    def __init__(self) -> None:
        """Initialize an empty cache store."""
        self._store: MutableMapping[str, str] = {}

    def build_key(
        self,
        prompt: str,
        model_name: str,
        provider: str,
        temperature: float,
        variables: Mapping[str, str],
    ) -> str:
        """Return a deterministic cache key for the given arguments."""
        payload = {
            "prompt": prompt,
            "model": model_name,
            "provider": provider,
            "temperature": round(temperature, 3),
            "variables": dict(sorted(variables.items())),
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return sha256(encoded.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[str]:
        """Return cached value if present."""
        return self._store.get(key)

    def set(self, key: str, value: str) -> None:
        """Store a cache entry."""
        self._store[key] = value


class PromptRunner:
    """High-level orchestrator that renders and executes prompts."""

    def __init__(
        self,
        loader: PromptLoader,
        *,
        hooks: Optional[Sequence[PromptHook]] = None,
        cache: Optional[PromptCache] = None,
    ) -> None:
        """Create a PromptRunner bound to a particular loader."""
        self.loader = loader
        self.cache = cache or PromptCache()
        self.hooks = HookManager(hooks)
        self._factories: MutableMapping[str, ClientFactory] = {}

    def register_client(self, provider: str, factory: ClientFactory) -> None:
        """Register a client factory for a provider key."""
        provider_key = provider.strip().lower()
        if not provider_key:
            raise PromptProviderError("Provider key must be a non-empty string.")
        self._factories[provider_key] = factory

    def run(
        self,
        prompt_name: str,
        variables: Mapping[str, object] | None = None,
        *,
        tools: Optional[Sequence[ToolSpecification]] = None,
        use_cache: bool = True,
    ) -> LLMResponse:
        """Execute a prompt using the registered provider client."""
        plan = self._build_execution_plan(
            prompt_name, variables or {}, tools, streaming=False
        )

        self.hooks.before_run(plan.context)

        if use_cache:
            cached = self.cache.get(plan.cache_key)
            if cached is not None:
                return {"reasoning": "", "output": cached}

        try:
            response = plan.client.generate(
                plan.rendered_prompt,
                tools=self._tools_argument(plan.tools),
            )
        except Exception as exc:
            self.hooks.on_error(plan.context, exc)
            raise

        if use_cache:
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
        plan = self._build_execution_plan(
            prompt_name, variables or {}, tools, streaming=True
        )

        self.hooks.before_run(plan.context)
        collected: List[str] = []
        try:
            iterator = plan.client.stream_generate(
                plan.rendered_prompt,
                tools=self._tools_argument(plan.tools),
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
            raise PromptValidationError(
                "Streaming is not supported for structured prompts."
            )

        rendered_prompt, normalized_variables = definition.render_with(variables)
        client = self._resolve_client(definition)
        resolved_tools = self._resolve_tools(definition, client, tools)

        cache_key = ""
        if not streaming:
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
            raise PromptProviderError(
                f"No LLM client factory registered for provider '{provider}'."
            )
        factory = self._factories[key]
        return factory(definition.model.name, definition.model.temperature)

    @staticmethod
    def _resolve_tools(
        definition: PromptDefinition,
        client: LLMClient,
        override_tools: Optional[Sequence[ToolSpecification]],
    ) -> Optional[Sequence[ToolSpecification]]:
        if override_tools is not None:
            if override_tools and not client.supports_tools:
                model_name = definition.model.name
                raise PromptProviderError(
                    f"Client for model '{model_name}' does not support tool execution."
                )
            return override_tools
        configured = definition.build_tools()
        if configured and not client.supports_tools:
            model_name = definition.model.name
            raise PromptProviderError(
                f"Client for model '{model_name}' does not support configured tools."
            )
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

"""Core client-side abstractions used by PromptKit."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Protocol, TypedDict


class ToolSpecification(TypedDict, total=False):
    """Schema describing a callable tool exposed to an LLM provider."""

    name: str
    description: str
    parameters: Dict[str, Any]
    type: str
    url: str


class LLMResponse(TypedDict):
    """Normalized response returned from an LLM client."""

    reasoning: str
    output: str


class LLMClient(Protocol):
    """Protocol implemented by LLM provider adapters."""

    supports_tools: bool
    close_called: bool = False

    def generate(
        self,
        prompt: str,
        tools: Optional[List[ToolSpecification]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """Return a completed response for the given prompt."""
        ...

    def stream_generate(
        self,
        prompt: str,
        tools: Optional[List[ToolSpecification]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        """Yield tokens for the given prompt as they are produced."""
        ...

    def close(self) -> None:
        """Close client and clean up resources (optional)."""
        # you must set this.
        self.close_called = True
        ...

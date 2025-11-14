"""Define interface protocols for LLM models, with optional structured tool support."""

from typing import Any, Dict, Iterator, List, Literal, Optional, Protocol, TypedDict

from pydantic import BaseModel, Field


class ToolSpec(TypedDict, total=False):
    """Schema for a tool definition compatible with function-calling models."""

    name: str
    description: str
    parameters: Dict[str, Any]  # typically a JSON schema–style object
    type: str
    url: str


class LLMOutput(TypedDict):
    """Structured output from any LLM model."""

    reasoning: str
    output: str


class LLMModel(Protocol):
    """Protocol definition for all LLM client implementations."""

    supports_tools: bool  # indicates if this model can handle tool usage

    def __init__(self) -> None:
        """Initialize the model client (no model/temperature at construction)."""
        ...

    def generate(
        self,
        prompt: str,
        tools: Optional[List[ToolSpec]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> LLMOutput:
        """Generate a response from the model.

        Args:
            prompt (str): The prompt to send to the model.
            tools (Optional[List[ToolSpec]]): Optional list of structured tool definitions.
            model (Optional[str]): Optional model identifier to use for the request.
            temperature (Optional[float]): Sampling temperature to use.

        Raises:
            NotImplementedError: If tools are provided but not supported.

        Returns:
            LLMOutput: A dict containing 'reasoning' and 'output' strings.
        """
        ...

    def stream_generate(
        self,
        prompt: str,
        tools: Optional[List[ToolSpec]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        """Stream model tokens as they are generated.

        Args:
            prompt (str): The input prompt.
            tools (Optional[List[ToolSpec]]): Optional list of structured tool definitions.
            model (Optional[str]): Optional model identifier to use for streaming.
            temperature (Optional[float]): Sampling temperature to use.

        Raises:
            NotImplementedError: If tools are provided but not supported.

        Returns:
            Iterator[str]: Yields strings as model tokens are produced.
        """
        ...


class EmbeddingsModel(Protocol):
    """Protocol for embedding model clients."""

    def embed(self, text: str, model: str) -> list[float]:
        """Generate an embedding for the given text.

        Args:
            text (str): Input text to embed.
            model (str): The embedding model to use.

        Returns:
            list[float]: Vector of floats representing the embedding.
        """
        ...


class ToolCallFunction(BaseModel):
    """Details of a tool function call."""

    name: str
    arguments: str


class ToolCall(BaseModel):
    """Structured representation of a tool call from an LLM."""

    id: str
    type: str
    function: ToolCallFunction


# --- LiteLLM / provider response models ---


class StreamingToolCall(BaseModel):
    """Streaming fragment describing a tool invocation."""

    id: str | None = None
    type: str | None = None
    function: ToolCallFunction | None = None


class StreamingDelta(BaseModel):
    """Delta content object for streaming responses."""

    content: list[dict[str, Any]] | str | None = None
    tool_calls: list[StreamingToolCall] | None = None


class Message(BaseModel):
    """Normalized message shape inside a completion choice."""

    role: str
    content: str = ""
    tool_calls: list[dict[str, Any]] | None = None


class Choice(BaseModel):
    """Single choice in a non-streaming completion response."""

    finish_reason: str
    index: int
    message: Message


class Usage(BaseModel):
    """Token usage statistics returned by some providers."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    """Top-level shape for a non-streaming completion response."""

    choices: list[Choice]
    created: float
    model: str
    usage: Usage | None = None


class StreamChoice(BaseModel):
    """Single streamed choice chunk."""

    delta: StreamingDelta
    finish_reason: str | None = None
    index: int


class StreamChunk(BaseModel):
    """Container for streamed chunks returned by some providers."""

    choices: list[StreamChoice]


class MCPToolConfig(BaseModel):
    """Configuration model used when initializing MCP clients."""

    name: str
    url: str
    type: Literal["stdio", "sse"] = Field(default="stdio")

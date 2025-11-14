"""Optimized LiteLLM client with improved type safety and cleaner architecture."""

from __future__ import annotations

import json
import os
from typing import Any, ClassVar, Iterator, Literal, cast

import litellm
import requests
from litellm.utils import validate_environment as litellm_validate_environment  # type: ignore
from pydantic import ValidationError

from src.py_promptkit.errors import MCPError, ModelRequestError
from src.py_promptkit.litellm.mcp_client import MCPClient
from src.py_promptkit.litellm.sse_transport import SSETransport
from src.py_promptkit.litellm.stdio_transport import StdIOTransport
from src.py_promptkit.models.llms import (
    CompletionResponse,
    LLMModel,
    LLMOutput,
    MCPToolConfig,
    Message,
    StreamChunk,
    StreamingToolCall,
    ToolCall,
    ToolSpec,
)
from src.py_promptkit.utils._logger import logger

# Type aliases for clarity
ChatMessage = dict[str, Any]
OpenAITool = dict[str, Any]
TransportType = Literal["stdio", "sse"]

# Constants
MAX_TOOL_ITERATIONS = 10
TOOL_TIMEOUT_SECONDS = 15


class LiteLLMClient(LLMModel):
    """LiteLLM client with tool-calling support and MCP integration.

    This client supports both streaming and non-streaming generation,
    with automatic tool execution through MCP or HTTP endpoints.

    Use as context manager for proper resource cleanup:
        with LiteLLMClient(mcp_tools=tools) as client:
            response = client.generate(prompt, model)
    """

    _validated_models: ClassVar[set[str]] = set()
    _tool_format_cache: ClassVar[dict[str, list[OpenAITool]]] = {}

    def __init__(
        self,
        mcp_tools: list[dict[str, Any]] | None = None,
        secrets: dict[str, str | None] | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize LiteLLM client with optional MCP tools and secrets.

        Args:
            mcp_tools: MCP tool configurations
            secrets: Environment secrets to expose for LiteLLM SDK
            verbose: Enable verbose logging for debugging
        """
        self.verbose = verbose
        self._set_environment_secrets(secrets or {})
        self.mcp_tools = mcp_tools or []
        self._mcp_clients: dict[str, MCPClient] = {}
        self.close_called = False
        self.supports_tools = True
        self._initialize_mcp_tools()

    # --- Context Manager Support ---

    def __enter__(self) -> LiteLLMClient:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and clean up resources."""
        self.close()

    def close(self) -> None:
        """Explicitly close all MCP clients and clean up resources."""
        if self.close_called:
            return

        for name, client in self._mcp_clients.items():
            try:
                client.close()
                if self.verbose:
                    logger.info(f"Closed MCP client: {name}")
            except Exception as e:
                logger.error(f"Error closing MCP client {name}: {e}")

        self._mcp_clients.clear()
        self.close_called = True

    # --- Environment & Setup ---

    def _set_environment_secrets(self, secrets: dict[str, str | None]) -> None:  # noqa: C901
        """Set environment variables from secrets dict with validation."""
        for key, value in secrets.items():
            if value is None:
                logger.warning(f"Secret '{key}' is None, skipping")
                continue

            if key in os.environ and os.environ[key] != value:
                logger.warning(f"Secret '{key}' already set with different value, not overwriting")
                continue

            if key not in os.environ:
                os.environ[key] = value
                if self.verbose:
                    logger.info(f"Set environment secret: {key}")

    def _ensure_environment_ready(self, model: str) -> None:
        """Validate LiteLLM environment for model (cached)."""
        if model in self._validated_models:
            return

        result: dict[str, Any] = cast(dict[str, Any], litellm_validate_environment(model))
        if not result.get("keys_in_environment", False):
            missing = result.get("missing_keys", [])
            raise ModelRequestError(
                f"Environment validation failed for '{model}'. Missing keys: {missing if missing else 'unknown'}"
            )

        self._validated_models.add(model)

    # --- MCP Tools ---

    def _initialize_mcp_tools(self) -> None:
        """Initialize all configured MCP tools."""
        for raw_config in self.mcp_tools:
            try:
                config = MCPToolConfig.model_validate(raw_config)
                if config.name in self._mcp_clients:
                    logger.warning(f"MCP tool '{config.name}' already initialized, skipping")
                    continue

                # Build appropriate transport instance based on config.type
                if config.type == "stdio":
                    transport = StdIOTransport(config.url)
                else:
                    # default to SSETransport for other network-based transports
                    transport = SSETransport(config.url)

                client = MCPClient(transport=transport)
                client.initialize()
                self._mcp_clients[config.name] = client

                log_fn = logger.info if self.verbose else logger.debug
                log_fn(f"Initialized MCP tool: {config.name}")

            except (ValidationError, MCPError) as e:
                logger.error(f"Failed to initialize MCP tool '{raw_config.get('name', 'unknown')}': {e}")

    # --- Tool Conversion & Execution ---

    def _convert_tools_to_openai_format(self, tools: list[ToolSpec]) -> list[OpenAITool]:
        """Convert ToolSpec list to OpenAI tool format with caching."""
        # Create cache key from tool names
        cache_key = "|".join(sorted(str(t.get("name", "")) for t in tools))

        if cache_key in self._tool_format_cache:
            return self._tool_format_cache[cache_key]

        openai_tools: list[OpenAITool] = []

        for spec in tools:
            try:
                params = spec.get("parameters", {"type": "object", "properties": {}})
                if isinstance(params, str):
                    params = json.loads(params)

                openai_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": spec.get("name", ""),
                            "description": spec.get("description", ""),
                            "parameters": params,
                        },
                    }
                )
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                logger.error(f"Invalid tool spec '{spec.get('name', 'unknown')}': {e}")

        self._tool_format_cache[cache_key] = openai_tools
        return openai_tools

    def _execute_tool(  # noqa: C901
        self,
        name: str,
        arguments: dict[str, Any],
        tool_specs: list[ToolSpec],
    ) -> str:
        # noqa: C901
        """Execute tool by name, checking MCP clients first, then HTTP tools.

        Args:
            name: Tool name to execute
            arguments: Tool arguments
            tool_specs: List of available tool specifications

        Returns:
            Tool execution result as string
        """
        # Try MCP client
        if name in self._mcp_clients:
            try:
                if self.verbose:
                    logger.info(f"Calling MCP tool '{name}' with args: {arguments}")
                result = self._mcp_clients[name].call_tool(name, arguments)
                # Convert dict result to string for consistent return type
                return json.dumps(result)
            except MCPError as e:
                logger.error(f"MCP error calling '{name}': {e}")
                return f"[MCP Error] {e}"

        # Try HTTP tool
        for spec in tool_specs:
            if spec.get("name") == name:
                try:
                    url = spec.get("url", "")
                    if self.verbose:
                        logger.info(f"Executing HTTP tool '{name}' at {url}")
                    response = requests.post(url, json=arguments, timeout=TOOL_TIMEOUT_SECONDS)
                    response.raise_for_status()
                    return response.text
                except requests.RequestException as e:
                    logger.error(f"HTTP tool error for '{name}': {e}")
                    return f"[Tool Execution Error] {e}"

        logger.warning(f"Tool '{name}' not found in MCP clients or HTTP tools")
        return f"[Tool Error] Tool '{name}' not found"

    # --- API Calls ---

    def _complete(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float,
        tools: list[OpenAITool] | None = None,
    ) -> CompletionResponse:
        """Make non-streaming completion request."""
        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        try:
            response = litellm.completion(**params).model_dump()  # type: ignore
            return CompletionResponse.model_validate(response)
        except ValidationError as e:
            logger.error(f"Invalid API response structure: {e}")
            raise ModelRequestError(f"Invalid API response: {e}") from e
        except Exception as e:
            logger.error(f"LiteLLM completion failed: {e}")
            raise ModelRequestError(f"LiteLLM request failed: {e}") from e

    def _complete_stream(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float,
        tools: list[OpenAITool] | None = None,
    ) -> Iterator[StreamChunk]:
        """Make streaming completion request with typed response."""
        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        try:
            stream = litellm.completion(**params)  # type: ignore
            for chunk in stream:
                try:
                    yield StreamChunk.model_validate(chunk)
                except ValidationError as e:
                    logger.warning(f"Invalid stream chunk: {e}")
                    continue
        except Exception as e:
            logger.error(f"Streaming request failed: {e}")
            raise ModelRequestError(f"Streaming request failed: {e}") from e

    # --- Utilities ---

    @staticmethod
    def _to_dict(obj: Any) -> dict[str, Any]:
        """Convert object to dict using canonical Pydantic method."""
        if isinstance(obj, dict):
            return cast(dict[str, Any], obj)
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        raise TypeError(f"Cannot convert {type(obj)} to dict - not a dict or Pydantic model")

    @staticmethod
    def _extract_text_content(content: list[dict[str, Any]] | str) -> str:  # noqa: C901
        """Extract text from content (handles list or string format)."""
        if isinstance(content, str):
            return content
        parts: list[str] = []
        for item in content:
            t: str = item.get("type", "")
            if t == "text":
                txt: str = item.get("text", "")
                parts.append(txt)
        return "".join(parts)

    def _parse_tool_calls(self, message_dict: Message) -> list[ToolCall]:  # noqa: C901
        """Parse and validate tool calls from message dict.

        Raises warnings for malformed tool calls but continues processing.
        """
        tool_calls: list[dict[str, Any]] = message_dict.tool_calls or []

        validated_calls: list[ToolCall] = []
        for call in tool_calls:
            try:
                validated_calls.append(ToolCall.model_validate(call))
            except ValidationError as e:
                logger.warning(f"Skipping invalid tool call: {e}")

        return validated_calls

    def _parse_tool_arguments(self, arguments_str: str, tool_name: str) -> dict[str, Any]:
        """Parse tool arguments with error handling and logging."""
        try:
            return json.loads(arguments_str)
        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse arguments for tool '{tool_name}': {e}. Arguments: {arguments_str[:100]}..."
            )
            return {}

    # --- Public API ---

    def generate(  # noqa: C901
        self,
        prompt: str,
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> LLMOutput:  # noqa: C901
        """Generate non-streaming response with optional tool calling.

        Args:
            prompt: User prompt
            model: Model identifier
            tools: Optional list of tool specifications
            temperature: Sampling temperature (default: 0.0)

        Returns:
            LLMOutput with reasoning and output text

        Raises:
            ModelRequestError: If request fails
            NotImplementedError: If tools requested but model doesn't support them
        """
        if self.close_called:
            raise RuntimeError("Client has been closed")

        if model is None:
            raise ModelRequestError("Model must be provided to generate()")

        self._ensure_environment_ready(model)
        temp = temperature if temperature is not None else 0.0

        if tools and not self.supports_tools:
            raise NotImplementedError("This client does not support tool-calling.")

        openai_tools = self._convert_tools_to_openai_format(tools or [])
        messages: list[ChatMessage] = [{"role": "user", "content": prompt}]

        for _ in range(MAX_TOOL_ITERATIONS):
            response = self._complete(messages, model, temp, openai_tools)
            message = response.choices[0].message

            tool_calls = self._parse_tool_calls(message)
            if not tool_calls:
                # No tools to call, return final response
                content = self._extract_text_content(message.content)
                return {"reasoning": "", "output": content}

            # Add assistant message with tool calls
            assistant_content = message.content
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_content if assistant_content else "",
                    "tool_calls": [tc.model_dump() for tc in tool_calls],
                }
            )

            # Execute tools and add results
            for tc in tool_calls:
                args = self._parse_tool_arguments(tc.function.arguments, tc.function.name)
                result = self._execute_tool(tc.function.name, args, tools or [])
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )

        logger.error(
            f"Hit max tool iterations ({MAX_TOOL_ITERATIONS}) for model '{model}'. "
            f"This may indicate an infinite loop or very complex tool chain."
        )
        return {
            "reasoning": "",
            "output": f"[Error] Maximum tool iteration limit ({MAX_TOOL_ITERATIONS}) reached",
        }

    def stream_generate(  # noqa: C901
        self,
        prompt: str,
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:  # noqa: C901
        """Stream tokens from model with optional tool calling.

        Args:
            prompt: User prompt
            model: Model identifier
            tools: Optional list of tool specifications
            temperature: Sampling temperature (default: 0.0)

        Yields:
            Text tokens as they arrive from the model

        Raises:
            ModelRequestError: If streaming fails
            NotImplementedError: If tools requested but model doesn't support them
        """
        if self.close_called:
            raise RuntimeError("Client has been closed")

        if model is None:
            raise ModelRequestError("Model must be provided to stream_generate()")

        self._ensure_environment_ready(model)
        temp = temperature if temperature is not None else 0.0

        if tools and not self.supports_tools:
            raise NotImplementedError("This client does not support tool-calling.")

        openai_tools = self._convert_tools_to_openai_format(tools or [])
        messages: list[ChatMessage] = [{"role": "user", "content": prompt}]
        partial_tools: dict[str, dict[str, Any]] = {}

        stream = self._complete_stream(messages, model, temp, openai_tools)

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Handle tool calls in streaming
            if delta.tool_calls:
                self._accumulate_tool_calls(delta.tool_calls, partial_tools)
                continue

            # Yield text content
            if delta.content:
                yield self._extract_text_content(delta.content)

        # Execute accumulated tool calls if any
        if partial_tools:
            yield from self._execute_accumulated_tools(partial_tools, messages, openai_tools, model, temp, tools or [])

    @staticmethod
    def _accumulate_tool_calls(  # noqa: C901
        tool_calls: list[StreamingToolCall],
        accumulator: dict[str, dict[str, Any]],
    ) -> None:
        # noqa: C901
        """Accumulate streaming tool call fragments."""
        for tc in tool_calls:
            tid = tc.id or "temp"
            if tid not in accumulator:
                accumulator[tid] = {
                    "id": tid,
                    "function": {"name": "", "arguments": [], "args_complete": False},
                }

            if tc.function:
                if tc.function.name:
                    accumulator[tid]["function"]["name"] = tc.function.name
                if tc.function.arguments:
                    # Use list for efficient accumulation
                    accumulator[tid]["function"]["arguments"].append(tc.function.arguments)

    def _execute_accumulated_tools(
        self,
        tool_calls: dict[str, dict[str, Any]],
        messages: list[ChatMessage],
        openai_tools: list[OpenAITool],
        model: str,
        temperature: float,
        tool_specs: list[ToolSpec],
    ) -> Iterator[str]:
        """Execute accumulated tool calls and get final response."""
        # Prepare tool call results
        tool_results: list[tuple[str, dict[str, Any]]] = []

        for tid, tc in tool_calls.items():
            # Join accumulated argument fragments
            args_str = "".join(tc["function"]["arguments"])
            args = self._parse_tool_arguments(args_str, tc["function"]["name"])

            result = self._execute_tool(tc["function"]["name"], args, tool_specs)

            # Store for batch message update
            tool_call_message = {
                "id": tid,
                "type": "function",
                "function": {
                    "name": tc["function"]["name"],
                    "arguments": args_str,
                },
            }
            tool_results.append((tid, tool_call_message))

            # Add tool result message
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tid,
                    "content": result,
                }
            )

        # Add assistant message with all tool calls
        if tool_results:
            messages.insert(
                len(messages) - len(tool_results),
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [tc for _, tc in tool_results],
                },
            )

        # Get final response after tool execution
        try:
            final_response = self._complete(messages, model, temperature, openai_tools)
            final_message = final_response.choices[0].message
            content = self._extract_text_content(final_message.content)

            if content:
                yield content
        except Exception as e:
            logger.error(f"Failed to get final response after tool execution: {e}")
            yield f"[Error] Failed to complete response after tool execution: {e}"

    def __del__(self) -> None:
        """Cleanup fallback - prefer explicit close() or context manager."""
        if not self.close_called:
            logger.warning(
                "LiteLLMClient destroyed without explicit close(). Use context manager or call close() explicitly."
            )
            self.close()

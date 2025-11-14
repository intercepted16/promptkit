"""MCP tools manager: fetch, cache, and normalize tool results."""

from __future__ import annotations

from typing import Any, Dict, List, Union

from src.py_promptkit.models.transport import MCPTransport
from src.py_promptkit.utils._logger import logger


class MCPTools:
    """Handles fetching, caching, and normalizing tool results."""

    def __init__(self, transport: MCPTransport):
        """Initialize MCPTools with the given transport."""
        self.transport = transport
        self._tools_cache: Dict[str, Dict[str, Any]] = {}

    def fetch_tools(self) -> None:
        """Fetch and cache available tools."""
        result = self.transport.send_request("tools/list", {})
        tools = result.get("tools", [])
        for tool in tools:
            name = tool.get("name")
            if name:
                self._tools_cache[name] = tool
        logger.info("Cached %d MCP tools", len(self._tools_cache))

    @staticmethod
    def extract_content(result: Dict[str, Any]) -> str:
        """Normalize tool call result into a string."""
        content: Union[None, str, Dict[str, Any], List[Union[str, Dict[str, Any]]]] = result.get("content")

        if isinstance(content, list) and content:
            first_item: Union[str, Dict[str, Any]] = content[0]
            if isinstance(first_item, dict):
                return str(first_item.get("text", first_item))
            return str(first_item)

        if isinstance(content, dict):
            return str(content.get("text", content))

        if isinstance(content, str):
            return content

        return str(result)

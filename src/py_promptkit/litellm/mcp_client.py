"""Coordinator for MCP server using any transport and tool manager.

This module provides the MCPClient orchestration and re-exports transport
and tools helpers from the dedicated modules so other code can still import
from `src.py_promptkit.litellm.mcp_client`.
"""

from __future__ import annotations

import ast
import json
import traceback
from typing import Any, Dict

from src.py_promptkit.errors import MCPError
from src.py_promptkit.litellm.sse_transport import SSETransport
from src.py_promptkit.litellm.stdio_transport import StdIOTransport
from src.py_promptkit.litellm.tools import MCPTools
from src.py_promptkit.models.transport import MCPTransport
from src.py_promptkit.utils._logger import logger


class MCPClient:
    """Coordinator for MCP server using any transport and tool manager."""

    def __init__(self, transport: MCPTransport):
        """Initialize the MCP client with the given transport."""
        self.transport = transport
        self.tools = MCPTools(transport)
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the MCP client and fetch tools."""
        if self._initialized:
            return

        if isinstance(self.transport, StdIOTransport):
            self.transport.start()

        payload = {
            "protocolVersion": "1.0",
            "capabilities": {},
            "clientInfo": {"name": "mcp-client", "version": "1.0"},
        }
        self.transport.send_request("initialize", payload)
        self.tools.fetch_tools()
        self._initialized = True

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool by name with the given arguments."""
        if not self._initialized:
            self.initialize()
        result = self.transport.send_request("tools/call", {"name": name, "arguments": arguments})
        content = self.tools.extract_content(result)

        # Try to parse JSON first, then fall back to Python literal parsing
        try:
            return json.loads(content)
        except Exception:
            try:
                return ast.literal_eval(content)
            except Exception as e:
                logger.error("Failed to parse tool output: %s", traceback.format_exc())
                raise MCPError(f"Failed to parse tool output: {content}") from e

        return content

    def close(self) -> None:
        """Close the MCP client and clean up resources."""
        if isinstance(self.transport, StdIOTransport):
            self.transport.stop()
        self._initialized = False

    def __enter__(self):
        """Initialize the client on enter.

        (for ctx mgr usage)
        """
        self.initialize()
        return self

    def __exit__(self, **args: Any) -> bool:
        """Ensure proper cleanup on exit.

        (for ctx mgr usage)
        """
        self.close()
        return False


# Re-export common classes for backward compatibility
__all__ = ["MCPClient", "StdIOTransport", "SSETransport", "MCPTools", "MCPTransport", "MCPError"]

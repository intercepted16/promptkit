"""Transport protocol definitions for MCP transports."""

from __future__ import annotations

from typing import Any, Dict, Protocol


class MCPTransport(Protocol):
    """Protocol interface for MCP transport implementations.

    Implementations must provide a send_request(method, params) -> Dict[str, Any]
    method used by the MCP client and tools manager.
    """

    def send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON-RPC request and return the response."""
        ...

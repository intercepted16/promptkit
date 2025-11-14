"""SSE-based MCP transport implementation."""

from __future__ import annotations

import json
from typing import Any, Dict

import requests

from src.py_promptkit.errors import MCPError


class SSETransport:
    """MCP transport using SSE requests."""

    def __init__(self, url: str, timeout: float = 15.0):
        """Initialize SSETransport with the given URL and timeout."""
        self.url = url
        self.timeout = timeout
        self._request_id = 0

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON-RPC request over SSE."""
        request = {"jsonrpc": "2.0", "id": self._next_id(), "method": method, "params": params}
        try:
            response = requests.post(
                self.url,
                json=request,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
                stream=True,
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
                    try:
                        data = json.loads(data_str)
                        if "error" in data:
                            error = data["error"]
                            raise MCPError(f"MCP Error: {error.get('message', str(error))}")
                        return data.get("result", {})
                    except json.JSONDecodeError:
                        continue

            raise MCPError("No valid SSE response received")

        except requests.RequestException as e:
            raise MCPError(f"SSE request failed: {e}") from e
        except Exception as e:
            raise MCPError(f"SSE communication failed: {e}") from e

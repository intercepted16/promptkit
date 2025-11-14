"""Compatibility shim re-exporting the relocated transport types.

Historically this module contained the protocol and both transport
implementations. The implementation has been split into
`litellm.stdio_transport` and `litellm.sse_transport`, and the
transport protocol now lives in `py_promptkit.models.transport`.

Code importing the old module will continue to work via these
re-exports.
"""

from __future__ import annotations

from src.py_promptkit.litellm.sse_transport import SSETransport
from src.py_promptkit.litellm.stdio_transport import StdIOTransport
from src.py_promptkit.models.transport import MCPTransport

__all__ = ["MCPTransport", "StdIOTransport", "SSETransport"]

"""StdIO-based MCP transport implementation."""

from __future__ import annotations

import json
import select
import subprocess
import time
from typing import Any, Dict, Optional, TextIO, cast

from src.py_promptkit.errors import MCPError
from src.py_promptkit.utils._logger import logger


class StdIOTransport:
    """MCP transport using stdio subprocess."""

    def __init__(self, command: str, timeout: float = 15.0):
        """Initialize StdIOTransport with the given command and timeout."""
        self.command = command
        self.timeout = timeout
        self._request_id = 0
        self._process: Optional[subprocess.Popen[Any]] = None

    def start(self) -> None:
        """Start subprocess for MCP server."""
        cmd_parts = self.command.split()
        # Use line-buffering (bufsize=1) when in text mode so readline works as expected.
        self._process = subprocess.Popen(
            cmd_parts, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, text=True
        )
        if not self.stdin or not self.stdout:
            raise MCPError("Failed to open stdin or stdout for MCP subprocess")
        logger.info("Started MCP server process: %s", self.command)

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:  # noqa: C901
        """Send a JSON-RPC request over stdio."""
        if not self._process or self._process.poll() is not None:
            raise MCPError("MCP server process is not running")

        request = {"jsonrpc": "2.0", "id": self._next_id(), "method": method, "params": params}

        try:
            self.stdin.write((json.dumps(request) + "\n"))
            self.stdin.flush()

            start_time = time.time()
            response_line = ""
            # Wait for data to be available on stdout using select to avoid blocking indefinitely.
            while True:
                elapsed = time.time() - start_time
                remaining = max(0.0, self.timeout - elapsed)
                if remaining <= 0:
                    break

                try:
                    rlist, _, _ = select.select([self._process.stdout], [], [], remaining)
                except ValueError:
                    # In case fileno() is invalid or process closed
                    break

                if not rlist:
                    # Nothing ready; check if process exited
                    if self._process.poll() is not None:
                        # Attempt to read stderr for diagnostics
                        stderr = ""
                        try:
                            stderr = (self.stderr.read() or "").strip()
                        except Exception:
                            pass
                        raise MCPError(f"MCP subprocess terminated unexpectedly: {stderr}")

                if rlist:
                    line = self.stdout.readline()
                    # EOF
                    if line == "":
                        # Process closed stdout; check stderr for message
                        stderr = ""
                        try:
                            stderr = (self.stderr.read() or "").strip()
                        except Exception:
                            pass
                        raise MCPError(f"MCP subprocess closed stdout: {stderr}")
                    response_line = line.strip()
                    break

            if not response_line:
                raise MCPError(f"Timeout waiting for response after {self.timeout}s")

            response = json.loads(response_line)
            if "error" in response:
                error = response["error"]
                raise MCPError(f"MCP Error: {error.get('message', str(error))}")

            return response.get("result", {})

        except json.JSONDecodeError as e:
            raise MCPError(f"Invalid JSON response: {e}") from e
        except Exception as e:
            raise MCPError(f"stdio communication failed: {e}") from e

    def stop(self) -> None:
        """Stop the MCP subprocess."""
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception as e:
                logger.warning("Error terminating MCP process: %s", e)
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None

    @property
    def stdin(self) -> "TextIO":
        """Type safe wrapper for process.stdin."""
        assert self._process and self._process.stdin, "stdin not available"
        return cast("TextIO", self._process.stdin)

    @property
    def stdout(self) -> "TextIO":
        """Type safe wrapper for process.stdout."""
        assert self._process and self._process.stdout, "stdout not available"
        return cast("TextIO", self._process.stdout)

    @property
    def stderr(self) -> "TextIO":
        """Type safe wrapper for process.stderr."""
        assert self._process and self._process.stderr, "stderr not available"
        return cast("TextIO", self._process.stderr)

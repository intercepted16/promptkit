# pyright: reportUnusedFunction=false
"""Tests for SSE transport functionality."""

import json
import logging
import socket
import threading
import time
from typing import Any

import requests
from flask import Flask, Response, request

from src.py_promptkit.litellm.sse_transport import SSETransport


def _find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


app = Flask(__name__)


def sse_event(data: str):
    return f"data: {data}\n\n"


@app.route("/stream/<mode>", methods=["POST"])
def stream(mode: str):  # noqa: C901
    """Stream SSE events."""
    req: dict[str, Any] = (
        request.get_json(silent=True) or {}
    )  # make it return None on failure, then make it a blank dict if None.
    base: dict[str, Any] = {"jsonrpc": "2.0", "id": req.get("id")}

    def gen_multiple():
        for name in ("first", "second", "third"):
            payload = base | {"result": {"output": name}}
            yield sse_event(json.dumps(payload))
            time.sleep(0.1)

    def gen_partial():
        payload = base | {"result": {"output": "ok"}}
        full = sse_event(json.dumps(payload))
        # yield in two parts to simulate partial arrival
        mid = len(full) // 2
        yield full[:mid]
        time.sleep(0.05)
        yield full[mid:]

    def gen_invalid_then_valid():
        yield sse_event("not-json")
        time.sleep(0.05)
        payload = base | {"result": {"output": "ok"}}
        yield sse_event(json.dumps(payload))

    def gen_keep_open():
        for i in range(5):
            payload = base | {"result": {"output": f"n{i}"}}
            yield sse_event(json.dumps(payload))
            time.sleep(0.2)

    if mode == "multiple":
        return Response(gen_multiple(), mimetype="text/event-stream")
    if mode == "partial":
        return Response(gen_partial(), mimetype="text/event-stream")
    if mode == "invalid_then_valid":
        return Response(gen_invalid_then_valid(), mimetype="text/event-stream")
    if mode == "keep_open":
        return Response(gen_keep_open(), mimetype="text/event-stream")

    return ("", 404)


@app.route("/shutdown", methods=["POST"])
def shutdown():
    func = request.environ.get("werkzeug.server.shutdown")
    if func is None:
        return ("Not running with the Werkzeug Server", 500)
    func()
    return ("", 200)


def _create_app():
    return app


def _run_flask_app(app: Flask, port: int):
    # shut up the flask & its dependencies logging
    werkzeug_logger = logging.getLogger("werkzeug")
    flask_logger = logging.getLogger("flask.app")
    prev_werkzeug_level = werkzeug_logger.level
    prev_flask_level = flask_logger.level
    werkzeug_logger.setLevel(logging.ERROR)
    flask_logger.setLevel(logging.ERROR)
    try:
        app.run(host="127.0.0.1", port=port, threaded=True, use_reloader=False)
    finally:
        # restore previous levels to avoid affecting other tests
        werkzeug_logger.setLevel(prev_werkzeug_level)
        flask_logger.setLevel(prev_flask_level)


def _start_server():
    port = _find_free_port()
    # js run in bg so we can run tests
    thread = threading.Thread(target=_run_flask_app, args=(app, port), daemon=True)
    thread.start()
    # wait briefly for server to start
    time.sleep(0.2)
    return port


def _shutdown_server(port: int):
    try:
        requests.post(f"http://127.0.0.1:{port}/shutdown", timeout=1.0)
    except Exception:
        pass


def test_sse_multiple_events():
    """Test SSE transport with multiple events."""
    port = _start_server()
    url = f"http://127.0.0.1:{port}/stream/multiple"
    transport = SSETransport(url=url, timeout=5.0)
    try:
        result = transport.send_request("tools/call", {"name": "test", "arguments": {}})
        assert isinstance(result, dict)
        assert result.get("output") == "first"
    finally:
        _shutdown_server(port)


def test_sse_partial_lines():
    """Test SSE transport with partial lines."""
    port = _start_server()
    url = f"http://127.0.0.1:{port}/stream/partial"
    transport = SSETransport(url=url, timeout=5.0)
    try:
        result = transport.send_request("tools/call", {"name": "test", "arguments": {}})
        assert isinstance(result, dict)
        assert result.get("output") == "ok"
    finally:
        _shutdown_server(port)


def test_sse_invalid_then_valid():
    """Test SSE transport with invalid then valid events."""
    port = _start_server()
    url = f"http://127.0.0.1:{port}/stream/invalid_then_valid"
    transport = SSETransport(url=url, timeout=5.0)
    try:
        result = transport.send_request("tools/call", {"name": "test", "arguments": {}})
        assert isinstance(result, dict)
        assert result.get("output") == "ok"
    finally:
        _shutdown_server(port)


def test_sse_keep_connection_open_streams_multiple_events():
    """Test SSE transport with keep-alive connection streaming multiple events."""
    port = _start_server()
    url = f"http://127.0.0.1:{port}/stream/keep_open"
    transport = SSETransport(url=url, timeout=5.0)
    try:
        result = transport.send_request("tools/call", {"name": "test", "arguments": {}})
        assert isinstance(result, dict)
        assert result.get("output", "").startswith("n0")
    finally:
        _shutdown_server(port)


__all__ = [
    "stream",
]

"""Tests for MCP client with stdio transport."""

import os

import pytest

from src.py_promptkit.litellm.mcp_client import MCPClient, StdIOTransport


@pytest.fixture
def mcp_client():
    """Fixture that starts a fake MCP client and cleans up after test."""
    here = os.path.dirname(__file__)
    fake_path = os.path.join(here, "fake_stdio.py")

    transport = StdIOTransport(command=f"python -u {fake_path}")
    client = MCPClient(transport)

    client.initialize()
    yield client  # provide the client to the test
    client.close()


def test_call_tool_returns_expected_output(mcp_client: MCPClient):
    """Test that calling a tool returns the expected mocked result."""
    response = mcp_client.call_tool("test_tool", {"foo": "bar"})
    # MCP client should parse the tool output into a dict when possible.
    assert isinstance(response, dict)
    assert response.get("output") == "ok"

#!/usr/bin/env python3
"""Fake stdio server for testing MCP client."""

import json
import sys

for line in sys.stdin:
    req = json.loads(line)
    resp = {"jsonrpc": "2.0", "id": req["id"], "result": {"output": "ok"}}
    print(json.dumps(resp), flush=True)  # noqa: T201 - intentional for test server

#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import time

HANDSHAKE = [
    {"jsonrpc": "2.0", "id": 0, "method": "initialize",
     "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "claude-code", "version": "1"}}},
    {"jsonrpc": "2.0", "method": "notifications/initialized"},
]


def call(request: dict, timeout: float) -> dict:
    bridge = subprocess.Popen(["xcrun", "mcpbridge"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    for message in [*HANDSHAKE, request]:
        bridge.stdin.write(json.dumps(message) + "\n")
        bridge.stdin.flush()
    deadline = time.monotonic() + timeout
    try:
        for line in bridge.stdout:
            reply = json.loads(line) if line.strip().startswith("{") else {}
            if reply.get("id") == 1 or (reply.get("id") is None and "error" in reply):
                return reply
            if time.monotonic() > deadline:
                break
    finally:
        bridge.kill()
    return {"error": {"message": f"no reply within {timeout:.0f} s"}}


def text(reply: dict) -> str:
    result = reply.get("result", reply.get("error", {}))
    return "\n".join([part.get("text", "") for part in result.get("content", [])] + [result.get("message", "")]).strip()


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: xcode_mcp.py list | <Tool> '<json arguments>' [timeout seconds]", file=sys.stderr)
        return 2
    tool = sys.argv[1]
    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"} if tool == "list" else \
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
         "params": {"name": tool, "arguments": json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}}}
    timeout = float(sys.argv[3]) if len(sys.argv) > 3 else 120
    reply = call(request, timeout)
    if "isn't approved" in text(reply):
        call({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
              "params": {"name": "XcodeOpenWorkspace", "arguments": {"path": os.getcwd()}}}, 60)
        reply = call(request, timeout)
    if tool == "list":
        for item in reply.get("result", {}).get("tools", []):
            print(f"{item['name']}: {item.get('description', '').strip().splitlines()[0]}")
        return 0
    print(text(reply))
    return 1 if reply.get("result", {}).get("isError") or "error" in reply else 0


if __name__ == "__main__":
    sys.exit(main())

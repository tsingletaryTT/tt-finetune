"""Small host-side client for the persistent TT runtime service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket


def request(
    socket_path: str | Path,
    payload: dict,
    *,
    timeout_seconds: float | None = None,
) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        if timeout_seconds is not None:
            client.settimeout(timeout_seconds)
        client.connect(str(socket_path))
        client.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        response = bytearray()
        while not response.endswith(b"\n"):
            chunk = client.recv(65536)
            if not chunk:
                break
            response.extend(chunk)
    if not response:
        raise RuntimeError("persistent TT runtime closed without a response")
    return json.loads(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    arguments = parser.parse_args()
    payload = json.loads(arguments.request.read_text(encoding="utf-8"))
    response = request(arguments.socket, payload)
    arguments.response.parent.mkdir(parents=True, exist_ok=True)
    arguments.response.write_text(
        json.dumps(response, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(response))
    if response.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

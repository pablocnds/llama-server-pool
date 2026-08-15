#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--model", required=True)
parser.add_argument("--alias", required=True)
parser.add_argument("--host", required=True)
parser.add_argument("--port", required=True, type=int)
parser.add_argument("--api-key", required=True)
parser.add_argument("--startup-delay", type=float, default=0.0)
options, _ = parser.parse_known_args()
time.sleep(options.startup_delay)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        pass

    def _authorized(self) -> bool:
        return self.headers.get("Authorization") == f"Bearer {options.api_key}"

    def do_GET(self) -> None:
        if not self._authorized():
            self.send_error(401)
            return
        if self.path in {"/health", "/v1/health"}:
            body = b'{"status":"ok"}'
        elif self.path.startswith("/props?") or self.path.startswith("/slots?"):
            body = json.dumps({"path": self.path, "method": "GET"}).encode()
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if not self._authorized():
            self.send_error(401)
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length)
        if self.headers.get_content_type() == "application/json":
            request = json.loads(raw_body)
        else:
            request = {
                "content_type": self.headers.get("Content-Type"),
                "raw_body": raw_body.decode("latin-1"),
            }
        if request.get("stream") is True:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            for text in ("hello", " world"):
                data = json.dumps(
                    {
                        "id": "fake-stream",
                        "object": "chat.completion.chunk",
                        "model": options.alias,
                        "choices": [{"delta": {"content": text}, "index": 0}],
                    }
                )
                self.wfile.write(f"data: {data}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        body = json.dumps(
            {
                "id": "fake-completion",
                "object": "chat.completion",
                "model": options.alias,
                "path": self.path,
                "request": request,
                "authorization": self.headers.get("Authorization"),
                "x_api_key": self.headers.get("X-Api-Key"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


ThreadingHTTPServer((options.host, options.port), Handler).serve_forever()

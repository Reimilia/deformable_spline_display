"""Local API + built frontend. No framework dependency; bind to localhost."""
from __future__ import annotations
import argparse
import json
import mimetypes
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit
from engine import CheckpointEngine


def serve(port=8765, host="127.0.0.1", allow_origins=()):
    engine = CheckpointEngine()
    catalog = engine.catalog()
    root = (Path(__file__).resolve().parent.parent / "app" / "dist").resolve()
    run_lock = threading.Lock()
    allow_origins = set(allow_origins)

    class Handler(BaseHTTPRequestHandler):
        def allowed_origin(self):
            origin = self.headers.get("Origin")
            host = self.headers.get("Host", "")
            return not origin or origin in allow_origins or origin in {f"http://{host}", f"https://{host}"}

        def cors_headers(self):
            origin = self.headers.get("Origin")
            if origin and self.allowed_origin():
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")

        def respond(self, code, data):
            payload = json.dumps(data, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.cors_headers()
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            path = unquote(urlsplit(self.path).path)
            if path.startswith("/api/") and not self.allowed_origin():
                return self.respond(403, {"error": "This page origin is not allowed by the dynamics service"})
            if path == "/api/checkpoints":
                return self.respond(200, catalog)
            if path == "/api/health":
                return self.respond(200, {"ok": True, "checkpoints": catalog["count"]})
            file = (root / (path.lstrip("/") or "index.html")).resolve()
            if not file.is_relative_to(root) or not file.is_file():
                return self.respond(404, {"error": "File not found. Build app with npm run build if dist is missing."})
            payload = file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(file.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            if urlsplit(self.path).path != "/api/rollout":
                return self.respond(404, {"error": "Unknown API endpoint"})
            if not self.allowed_origin():
                return self.respond(403, {"error": "This page origin is not allowed by the dynamics service"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("Request must be a JSON object")
                with run_lock:
                    result = engine.run(data)
                self.respond(200, result)
            except (ValueError, KeyError, TypeError) as err:
                self.respond(400, {"error": str(err)})
            except Exception as err:
                traceback.print_exc()
                self.respond(500, {"error": f"Rollout failed: {err}"})

        def do_OPTIONS(self):
            if urlsplit(self.path).path != "/api/rollout":
                return self.respond(404, {"error": "Unknown API endpoint"})
            if not self.allowed_origin():
                return self.respond(403, {"error": "This page origin is not allowed by the dynamics service"})
            self.send_response(204)
            self.cors_headers()
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Max-Age", "600")
            self.end_headers()

    print(f"Loaded {catalog['count']} checkpoints. Open http://{host}:{port}", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--allow-origin", action="append", default=[], help="Allowed browser origin, e.g. https://YOUR-USERNAME.github.io; repeat for multiple origins")
    args = parser.parse_args()
    serve(args.port, args.host, args.allow_origin)

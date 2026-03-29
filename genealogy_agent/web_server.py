"""
Simple HTTP server for the web UI, runs alongside the WebSocket server.

Serves the static HTML file on the configured web port.
Injects theme and config into the HTML at serve time.
"""

import http.server
import logging
import os
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

WEB_DIR = os.path.join(os.path.dirname(__file__), "web")

# Module-level config, set before starting
_config: Dict[str, Any] = {}


def set_config(config: Dict[str, Any]) -> None:
    """Set the config for template injection."""
    global _config
    _config = config


class WebUIHandler(http.server.SimpleHTTPRequestHandler):
    """Serve files from the web/ directory with config injection."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_index()
        elif self.path == "/config.json":
            self._serve_config()
        else:
            super().do_GET()

    def _serve_index(self):
        index_path = os.path.join(WEB_DIR, "index.html")
        try:
            with open(index_path) as f:
                html = f.read()

            # Inject theme CSS variables
            theme = _config.get("theme", {})
            if theme:
                css_vars = "; ".join(
                    f"--{k.replace('_', '-')}: {v}" for k, v in theme.items()
                )
                html = html.replace(
                    "<head>",
                    f"<head><style>:root {{ {css_vars} }}</style>",
                )

            # Inject title
            title = _config.get("app", {}).get("title", "Chat Agent")
            html = html.replace(
                "<title>Genealogy Agent</title>",
                f"<title>{title}</title>",
            )
            html = html.replace(
                ">Genealogy Agent</h1>",
                f">{title}</h1>",
            )

            # Inject WebSocket URL
            ws_port = _config.get("server", {}).get("ws_port", 8765)
            html = html.replace(
                "const WS_URL = `ws://${location.hostname}:"
                "${location.port ? parseInt(location.port) - 1 : 8765}`;",
                f"const WS_URL = `ws://${{location.hostname}}:{ws_port}`;",
            )

            content = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            logger.error(f"Failed to serve index: {e}")
            self.send_error(500)

    def _serve_config(self):
        """Serve public config as JSON (for JS consumption)."""
        import json

        public = {
            "title": _config.get("app", {}).get("title", "Chat Agent"),
            "theme": _config.get("theme", {}),
            "ws_port": _config.get("server", {}).get("ws_port", 8765),
        }
        content = json.dumps(public).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format, *args):
        logger.debug(f"HTTP: {format % args}")


def start_web_server(
    host: str = "0.0.0.0",
    port: int = 8766,
    config: Optional[Dict[str, Any]] = None,
) -> threading.Thread:
    """Start the HTTP server for the web UI in a background thread."""
    if config:
        set_config(config)

    server = http.server.HTTPServer((host, port), WebUIHandler)
    thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
        name="web-ui",
    )
    thread.start()
    logger.info(f"Web UI serving at http://{host}:{port}")
    return thread

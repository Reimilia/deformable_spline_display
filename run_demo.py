"""Run the bundled frontend and checkpoint service with one command."""
from pathlib import Path
import runpy
import sys

backend = Path(__file__).resolve().parent / "racket_viz" / "multilink_backend"
sys.path.insert(0, str(backend))
runpy.run_path(str(backend / "server.py"), run_name="__main__")

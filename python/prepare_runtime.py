"""Stage the pinned, self-hosted WASM runtime. No pip dependencies required."""
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "web/node_modules/pyodide"
VERSION = "0.28.3"
DEST = ROOT / "web/public/wasm" / VERSION
CORE = ("pyodide.mjs", "pyodide.asm.js", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json")


def prepare():
    if json.loads((SOURCE/"package.json").read_text())["version"] != VERSION:
        raise RuntimeError("Pyodide package version differs from runtime pin")
    DEST.mkdir(parents=True,exist_ok=True)
    for name in CORE:
        shutil.copyfile(SOURCE/name,DEST/name)
    # Emscripten's .asm.js uses CommonJS when tested in Node. Browsers ignore
    # this package boundary; it prevents inheriting the frontend's ESM type.
    (DEST/"package.json").write_text('{"type":"commonjs"}\n')
    shutil.copyfile(ROOT/"licenses/PYODIDE-MPL-2.0.txt",DEST/"LICENSE.txt")
    lock = json.loads((SOURCE/"pyodide-lock.json").read_text())
    needed=set()
    def add(name):
        if name in needed: return
        needed.add(name)
        for dependency in lock["packages"][name]["depends"]: add(dependency)
    add("numpy"); add("autograd")
    for name in sorted(needed):
        package=lock["packages"][name]; target=DEST/package["file_name"]
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==package["sha256"]: continue
        url=f"https://cdn.jsdelivr.net/pyodide/v{VERSION}/full/{package['file_name']}"
        with urllib.request.urlopen(url,timeout=120) as response:
            data=response.read()
        if hashlib.sha256(data).hexdigest()!=package["sha256"]:
            raise RuntimeError(f"Checksum mismatch for {name}")
        target.write_bytes(data)
        print(f"Staged {name} ({len(data)/1e6:.1f} MB)",flush=True)
    print(f"Self-hosted Pyodide {VERSION}, NumPy and Autograd are ready",flush=True)


if __name__ == "__main__": prepare()

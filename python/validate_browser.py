"""Numerical regression against the original PyTorch checkpoint engine.

Generates small reference cases for the same tests inside actual WebAssembly.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from engine import CheckpointEngine

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"web/public/python"))
from browser_engine import BrowserEngine

KEYS=("body","world","poses","phase_q","phase_p","target_body","target_world","reference_body","reference_world")


def difference(left,right):
    worst=0.
    for a,b in zip(left["runs"],right["runs"]):
        for key in KEYS:
            x,y=np.array(a[key]),np.array(b[key])
            np.testing.assert_allclose(x,y,rtol=2e-8,atol=2e-9,err_msg=key)
            worst=max(worst,float(np.max(np.abs(x-y))))
        for key in a["metrics"]:
            x,y=np.array(a["metrics"][key]),np.array(b["metrics"][key])
            np.testing.assert_allclose(x,y,rtol=2e-8,atol=2e-9,err_msg=key)
            worst=max(worst,float(np.max(np.abs(x-y))))
    return worst


def validate(output):
    manifest=json.loads((ROOT/"web/public/models/manifest.json").read_text())
    native=CheckpointEngine(); browser=BrowserEngine(manifest,ROOT/"web/public/models")
    cases=[]; report=[]
    # Every stored snapshot, both dynamics families, every saved variant.
    for entry in manifest["checkpoints"]:
        config={"checkpoint":entry["id"],"horizon":1}
        if entry["stage"]=="Hard transfer": config["group"]="hard_test"
        expected=native.run(config); actual=browser.run(config)
        error=difference(expected,actual)
        report.append({"checkpoint":entry["id"],"max_absolute_error":error})
        cases.append({"config":config,"expected":expected})
        print(f"PASS {entry['family']} {entry['label']} ({error:.3g})",flush=True)
    # Changed parameters exercise each closure and conditioning path, with
    # settling beyond tau=1, pose offsets, different tasks and comparison.
    for entry in manifest["checkpoints"]:
        if entry["stage"]!="Final": continue
        config={"checkpoint":entry["id"],"task_index":1,"horizon":1.25,"steps":entry["defaults"]["steps"]+4,
                "stiffness":.8,"damping":1.4,"target_dx":.12,"target_dy":-.08,
                "target_angle":23,"initial_angle":-11,"tolerance":.12,"fim":False}
        expected=native.run(config); actual=browser.run(config)
        error=difference(expected,actual)
        report.append({"checkpoint":entry["id"],"case":"changed controls","max_absolute_error":error})
        cases.append({"config":config,"expected":expected})
        print(f"PASS changed controls {entry['label']} ({error:.3g})",flush=True)
    for family in ("multilink","spline"):
        entries=[e for e in manifest["checkpoints"] if e["family"]==family and e["stage"]=="Final"]
        config={"checkpoint":entries[0]["id"],"compare":entries[-1]["id"],"horizon":1.25,"damping":1.3,"tolerance":.11}
        expected=native.run(config); actual=browser.run(config)
        error=difference(expected,actual)
        report.append({"case":family+" comparison","max_absolute_error":error})
        cases.append({"config":config,"expected":expected})
    # Noise intentionally uses NumPy's PCG64 stream in both CPython/WASM,
    # rather than Torch's generator. Verify determinism and a physical effect.
    entry=next(e for e in manifest["checkpoints"] if e["family"]=="spline" and e["conditioning"]=="observation" and e["stage"]=="Final")
    config={"checkpoint":entry["id"],"horizon":1,"noise":.01,"seed":19}
    noisy=browser.run(config); difference(noisy,browser.run(config))
    clean=browser.run({**config,"noise":0})
    assert not np.allclose(noisy["runs"][0]["phase_q"],clean["runs"][0]["phase_q"])
    cases.append({"config":config,"expected":noisy})
    output.mkdir(parents=True,exist_ok=True)
    summary={"reference":"original PyTorch engine", "cases":len(report),"checkpoints":manifest["count"],
             "max_absolute_error":max(r["max_absolute_error"] for r in report),"results":report}
    (output/"parity-report.json").write_text(json.dumps(summary,indent=2)+"\n")
    # Cases aren't published in the frontend and do not supply its trajectories.
    import gzip
    with gzip.open(output/"wasm-cases.json.gz","wt") as f: json.dump(cases,f,separators=(",",":"))
    print(json.dumps({k:v for k,v in summary.items() if k!="results"}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=ROOT/"validation")
    validate(p.parse_args().output)

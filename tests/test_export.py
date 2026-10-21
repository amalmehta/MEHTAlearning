"""The browser demo must compute what PyTorch computes: run site/js/model.js in
Node on the parity fixture that export_demo.py writes and compare."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "site" / "data" / "demo.json"

pytestmark = pytest.mark.skipif(not DEMO.exists() or shutil.which("node") is None,
                                reason="needs site/data/demo.json (run mehta.export_demo) and node")


def test_js_matches_pytorch(tmp_path):
    (tmp_path / "model.mjs").write_text((ROOT / "site" / "js" / "model.js").read_text())
    script = tmp_path / "check.mjs"
    script.write_text(f"""
import fs from "fs";
import {{ act, Posterior }} from "./model.mjs";
const demo = JSON.parse(fs.readFileSync({json.dumps(str(DEMO))}, "utf8"));
const m = demo.models.mehta, p = demo.parity;
const actions = p.obs.map((o, i) => act(m.actor, o, p.z[i]));
const post = new Posterior(m.encoder, m.latent_dim);
p.ctx.forEach((row) => post.add(row));
console.log(JSON.stringify({{ actions, ...post.get() }}));
""")
    out = json.loads(subprocess.check_output(["node", str(script)], text=True))
    fx = json.loads(DEMO.read_text())["parity"]
    # weights are exported rounded to 5 decimals, so allow a small tolerance
    assert out["actions"] == pytest.approx(fx["action"], abs=2e-3)
    assert out["mu"] == pytest.approx(fx["mu"], abs=2e-3)
    assert out["var"] == pytest.approx(fx["var"], rel=1e-2, abs=1e-5)

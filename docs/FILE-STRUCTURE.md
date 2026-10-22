# File Structure

```
MEHTAlearning/
├── README.md                 What it is, figures, links
├── MEHTAlearning.md          Project spec: definitions, decisions, deferred items, changelog
├── pyproject.toml, uv.lock   Pinned Python environment (Python 3.12, torch 2.2.2 CPU)
├── mehta/                    The research code
│   ├── envs.py               Point-Goal and Point-Waypoints environments, task splits
│   ├── nets.py               Actor, Q-network, PEARL encoder (product of Gaussians)
│   ├── agent.py              SAC + latent; clone, proximal term, first-order meta-update
│   ├── buffers.py            Per-task replay buffers and context tensors
│   ├── rollout.py            Episode collection, z sampling
│   ├── train.py              Meta-training for pearl / mehta / mehta_short / maml
│   ├── adapt.py              Post-training protocol and the continual sequence
│   ├── randomize.py          Physics domain-randomization wrapper and held-out shifts
│   ├── shift_eval.py         Checkpoints → return vs. shift magnitude, figure, site/data/shift.json
│   ├── horizon_sweep.py      Inner-loop horizon ablation (K_inner 5–80) → figure, site/data/horizon.json
│   ├── experiments.py        One (env, method, seed) run → results JSON
│   ├── run_all.py            Every run, in parallel; then report + export (one command)
│   ├── report.py             Aggregates, verdicts, README figures, site/data/results.json
│   └── export_demo.py        Seed-0 weights and latent explorer → site/data/demo.json
├── tests/
│   ├── test_core.py          Envs, PEARL math, MEHTA pieces, metrics
│   ├── test_randomize.py     Wrapper determinism, physics effects, shifts, checkpoints
│   └── test_export.py        JS demo model vs PyTorch (needs Node)
├── results/                  Confirmatory runs (seeds 0–2) as JSON; models/ and logs/ are git-ignored
├── site/                     The website (GitHub Pages serves this folder)
│   ├── index.html, style.css, favicon.svg
│   ├── js/app.js             Demo, charts, rollouts, latent explorer, theme
│   ├── js/charts.js          Small SVG line and bar charts
│   ├── js/model.js           JS mirror of the env, MLP and encoder
│   └── data/                 results*.json, shift.json, horizon.json, demo.json (generated)
├── docs/
│   ├── INSTRUCTIONS.md       Setup, run, use, manual review checklist
│   ├── SYSTEM-DESIGN.md      Architecture, flows, decisions, testing, limits
│   ├── FILE-STRUCTURE.md     This file
│   ├── LEVEL-2-PLAN.md       Next experiments: hypotheses, matrix, budget, gates
│   └── img/                  README figures (generated)
├── .github/workflows/pages.yml  Deploys site/ to GitHub Pages
└── inputs/                   The original brief image
```

# Instructions

## Setup

Needs [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 and the pinned packages) and, optionally, Node for the JS parity test.

```bash
uv sync
```

PyTorch is pinned to 2.2.2, the last release with Intel-Mac (x86_64) wheels; everything runs on CPU.

## Reproduce every result

```bash
uv run python -m mehta.run_all
```

This trains and evaluates 4 methods × 2 environments × seeds 0, 1, 2 (24 runs as parallel single-thread processes), then writes `site/data/results.json`, `site/data/demo.json` and `docs/img/*.png`. Finished runs are skipped, so an interrupted run resumes; add `--force` to redo everything, `--jobs N` to change parallelism.

One run on its own:

```bash
uv run python -m mehta.experiments --env point_goal --method mehta --seed 0
```

Methods: `pearl`, `mehta`, `mehta_short`, `maml`. Environments: `point_goal`, `waypoints`.

Rebuild only the aggregates and figures from existing results:

```bash
uv run python -m mehta.report
```

## Physics shift curves (sim-to-real readiness)

```bash
uv run python -m mehta.run_all --version shift --seeds 9 10 11
```

This trains `pearl_phys`, `mehta_phys` and `maml_phys` with physics domain randomization on both environments, then runs `mehta.shift_eval`. It loads each full checkpoint and writes `results/shift/`, `site/data/shift.json` and `docs/img/shift_curves.png`. To evaluate checkpoints that already exist, or only rebuild the figure:

```bash
uv run python -m mehta.shift_eval --methods pearl_phys mehta_phys maml_phys --seeds 9 10 11
```

```bash
uv run python -m mehta.shift_eval --report-only
```

## Horizon ablation (inner-loop length)

```bash
uv run python -m mehta.horizon_sweep
```

This trains `mehta_v2` with K_inner ∈ {5, 10, 20, 40, 80} (1×–16× the short-horizon setting) on both environments for seeds 12–14, 30 runs, 10 at a time. Each run is `mehta.experiments --k-inner K --out results/horizon`, and nothing else changes. It writes `results/horizon/<env>/mehta_v2_k<K>_s<seed>.json` (returns, `wall_sec`, `peak_rss_mb`), `site/data/horizon.json` and `docs/img/horizon.png`. Existing runs are skipped, so an interrupted sweep resumes. Use `--jobs N` for fewer parallel processes, and `--report-only` to rebuild only the figure and data.

## Meta-World ML1 (between the 2-D tasks and level 2)

```bash
uv sync --group metaworld
uv run --group metaworld python -m mehta.experiments --env mw_push --method mehta --seed 0
```

The optional `metaworld` group adds Meta-World 3.1.1, Gymnasium 1.4.0 and MuJoCo 3.3.0; a plain `uv sync` removes it again. Families are `mw_reach`, `mw_push` and `mw_pick_place`, and methods `pearl`, `mehta`, `mehta_short` and `maml` (the v2 and physics methods perturb point-mass dynamics and are refused). Episodes are 200 steps with γ = 0.99, and training is 100 iterations, about 350k env steps. A run takes about an hour on one CPU core. Results go to `results/<env>/<method>_s<seed>.json`, with returns for the ML1 held-out goals at K ∈ {0, 50, 100, 200}. Details are in `mehta/mw_envs.py`.

## Tests

```bash
uv run pytest
```

The Meta-World tests are skipped unless the `metaworld` group is installed (`uv run --group metaworld pytest`).

## View the website locally

```bash
python3 -m http.server 8765 --directory site
```

Then open http://localhost:8765. Opening `index.html` straight from disk will not load the data (browsers block `fetch` on `file://`).

## Deploy

Pushing to `main` runs `.github/workflows/pages.yml`, which publishes `site/` to GitHub Pages.

## Using the site

- **Try it**: click the arena to pick a goal direction. MEHTA/PEARL switches the meta-trained model; "Shift dynamics" rotates the agent's actions by 60°.
- **Post-training on a new world**: pick a held-out goal and slide K to see the real adapted policy after K gradient steps.
- **Charts**: hover points for exact values; click legend items to hide a method.

## Manual review checklist

Things the tests cannot check. Go through these in both light and dark mode (the ◐ button) and at phone width.

**Feel and smoothness**
- [ ] Clicking the demo arena shows the goal immediately (well under 100 ms) and the rollout animates without stutter.
- [ ] Switching MEHTA/PEARL or toggling the dynamics shift restarts the animation instantly, with no flash of an empty canvas.
- [ ] Dragging the K slider updates both snapshot canvases at once, with no layout jump.
- [ ] Hovering chart points and latent paths highlights without lag; the tooltip never covers the cursor.
- [ ] Theme toggle: every canvas and chart changes color, with no flicker or flash on reload.
- [ ] Scrolling the whole page is fluid; nothing shifts as data loads.

**Look**
- [ ] Warm light palette; accents only on MEHTA and links; dark mode readable everywhere (tables, verdict colors, tooltips, feedback tab).
- [ ] Method colors read as families: MEHTA variants orange/amber, PEARL variants blue, MAML green, distinguishable in both themes.
- [ ] Phone width (375 px): no horizontal scroll, demo canvas fills the width, tables scroll inside their box.

**Honesty**
- [ ] Every number on the page comes from `results.json` (spot-check two against `results/`).
- [ ] The roadmap is clearly marked as a plan; the pilot badge is visible.
- [ ] Reference links open the right arXiv papers.

**Live services**
- [ ] GitHub Pages URL loads with the favicon; the Feedback tab opens a new GitHub issue.

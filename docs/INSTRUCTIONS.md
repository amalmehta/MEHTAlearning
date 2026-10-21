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

## Tests

```bash
uv run pytest
```

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

PROJECT NAME: MEHTAlearning

META-INSTRUCTIONS:

<Read it all before acting. Ask about anything unclear, contradictory or
 underspecified — before starting and mid-build. Ask in the question widget
 (AskUserQuestion): related questions batched, concrete options, your
 recommendation first. Plain text only if the widget isn't available.>

<Make answering easy. Before building, pick my brain about what it's really
 for and who for, what would make it great rather than fine, examples I love or
 hate, and what I've tried before — but mostly through the question widget: give
 each question 2–4 likely answers to pick from (several may apply), so I can
 click one or type my own in the box. Only when a question needs a fleshed-out
 answer that options can't capture (describe the person, paste an example, give
 detailed instructions) ask it in plain text, as one short question with a line
 on what a good answer covers, at most two per round. Never a long numbered list
 of open questions. Wait for my answers and follow up where they open something
 up.>

<Don't expand scope. Anything not listed here is a proposal, including changes
 to this file — propose it, don't do it.>

<Prefer doing over describing: run the code, write the files, test it.>

<As soon as a first working version (the MVP) exists, open it for me before
 carrying on: launch the app, open the page in the browser, or run it on a real
 sample, and say in a line or two what to try. Do the same after each big change.>

<Always in scope, no proposal needed: when it goes on GitHub, a README that is
 easy to read at a glance — a line on what it is, then clear visuals
 (screenshots, a diagram or a chart), then links. Everything else goes in
 linked files: docs/INSTRUCTIONS.md (setup, run, use),
 docs/SYSTEM-DESIGN.md (see below) and docs/FILE-STRUCTURE.md (what's where). If what you're
 building is an application rather than a script, also a small unobtrusive feedback tab, and a settings
 button or tab (⌘, on the Mac) that gathers its preferences in one place.>

<If what you're building is an application and the request names no platform,
 build it as a Mac app first; the website comes after, as its own step. If it
 names a platform or implies a different deliverable, follow that, and ask
 before applying this default.>

<Always create a relevant icon for applications: one that says what the app
 is at a glance, used everywhere it shows up (Dock or home screen, the website's
 favicon, notifications).>

<Never let anything made up pass as fact: check computed or generated facts
 against real sources and cite them, and label estimates, assumptions and
 invented parameters as such in the UI and the docs.>

<Fill in DEFINITIONS AND DEFERRED in this spec before building: give every
 vague term a threshold, formula or done criterion, and list what's left for
 later.>

<Keep the interface clean and restrained: a warm, light palette with sparing
 accents, and dark mode checked on every component. When the visuals are open-
 ended, offer two or three mockups to pick from. Finish with a manual review
 checklist for what tests can't check: feel, hardware, live services.>

<Make it buttery smooth; this is a priority, not polish. Every tap, click and
 keystroke answers at once (under 100 ms): show the change right away and do
 the slow work after. Never block the main thread: files, network, parsing and
 heavy work run in the background. Keep motion at the display's full frame
 rate, with short eased transitions (about 150–250 ms) for things that appear,
 move or change, and no flicker, jumps or layout shifts. Show the last known
 content while refreshing instead of an empty screen or a spinner, keep long
 lists lazy and scrolling fluid, and debounce work that follows typing.
 Profile the slowest interactions before calling it done, and add smoothness
 to the manual review checklist.>

<Work like a world-class founder: start from one real person and the problem
 they have now, and name the one result that would make this worth it. Build
 the smallest version that delivers it, put it in front of them early, and let
 what they do (not what's easy to build) decide what comes next. Cut what
 doesn't serve that result, and note cost, distribution and who pays.>

<Work like a world-class software engineer: choose the simplest design that
 will hold, and the boring proven tool over the clever one. Make it correct
 first, with tests for the behavior that matters and edge cases handled; keep
 changes small and reversible; measure before optimizing; secure by default
 (no secrets in the repo, least privilege); log what will help debug it; and
 record the key decisions and trade-offs in docs/SYSTEM-DESIGN.md.>

<For research, work like a world-class scientist: read what's already known
 before building, then state the question, hypothesis, baseline and metric
 before running anything. Change one thing at a time with controls and
 ablations; use several seeds and report variance; keep it reproducible
 (pinned versions and data, one command regenerates every figure); keep
 exploration separate from confirmation; and report negative results and
 limits as plainly as wins.>

<Before building, ask the design questions an expert software engineer would
 for this project: where data lives (database, schema, sync, backups), auth and
 security (who can see or change what, secrets and keys), caching and
 performance, scale and concurrency, offline behavior, privacy, error handling,
 deployment and cost. Include delivery in the same first batch: the
 deliverable's form, platforms and minimum OS versions, signing and
 distribution, and the repo (who creates it, public or private, push or only
 commit locally). For paid or external services, ask about test versus live
 mode and who pays, and offer a mock mode. Ask for real sample inputs and
 working credentials before building. Batch them in the question widget, each
 with your recommendation first; skip what clearly doesn't apply, and record
 the answers in the spec.>

<When an application takes files as input, offer a visible Upload button that
 opens the file picker, accept files dragged anywhere onto the window (not
 only onto a small drop zone), and accept pasting them with ⌘V.>

<Color-code tags, categories and types by similarity: related ones share a
 hue family, with shades far enough apart to tell them apart at a glance, in
 both light and dark mode.>

<Name things the way a person would say them — "Goal Tracker", not
 goal_tracker — for the app, its windows, titles, files people open, repo
 descriptions and README headings. When you create the GitHub repo, name it
 with no "_" or "-": one word or joined words, e.g. GoalTracker.>

<Always in scope: a system design doc in the codebase, docs/SYSTEM-DESIGN.md,
 kept current as the build changes. Cover the architecture (with a Mermaid
 diagram), each component's job, the main flows, where data lives, the key
 design decisions and their trade-offs, how it's tested, and known limits.>

<Finish by listing every deliverable: path, what it is, how to check it works.>

<Git rules (no Claude attribution, never commit .claude/) are in
 ~/.claude/CLAUDE.md and apply on their own — nothing to repeat here.>

<Keep the changelog at the bottom current.>

FROM SIMILAR PAST PROJECTS:

<The closest earlier projects by meaning, each with one concrete thing it settled. Examples to learn from, not choices to copy.>

<neuralink_product_dev (website, 89% alike): asked "Another Claude session ('Neuralink product development') is editing this repo right now: about 40 files across all four projects changed in …" and settled on "Wait, then commit (Recommended)">

<social_media_dev (web app, 86% alike): assumed: Text only for v1; single user on local machine; no git repo (declined).>

<photo_analyzer (api / server, 84% alike): assumed: Agent fills this in: anything it had to guess, decisions it made without>

LEARNED ACROSS PROJECTS:

<What held across several kinds of earlier projects: starting points, not rules.>

<Decide which data sources, external services and licenses are allowed, test versus live mode, key storage, who pays, spending limits, privacy, and offline behavior. (across 12 kinds)>

<Settle the deliverable's form, target platforms, minimum versions and distribution up front, including packaging, signing, hosting and who runs it. (across 11 kinds)>

<Define vague terms, scope boundaries and scoring rules with concrete thresholds, formulas and done criteria, recording what is deferred. (across 10 kinds)>

<Validate generated or computed content against real sources, cite them, and label estimates or invented parameters so nothing fabricated passes as fact. (across 8 kinds)>

<Settle version control policy early: who creates and hosts the repository, its visibility, and whether to push or only commit locally. (across 7 kinds)>

<Gather real sample inputs and working credentials early, then verify live APIs, model output and system tools, keeping a fallback ready. (across 7 kinds)>

<Define each output's specs, tone and visual style before building, offering a few concrete concepts or mockups to pick from when open-ended. (across 7 kinds)>

<Plan a manual or human review checklist for what automated tests can't verify, such as hardware, live services, feel or taste. (across 3 kinds)>

LEARNED FOR ML / DATA MODEL PROJECTS:

<Starting points from 13 earlier ml / data model projects, not rules: use what fits this one, and ask when it differs.>

<Settle what produces adaptive judgments or content (an LLM, pretrained or learned model, programmatic functions, bundled data) and which decisions stay in deterministic code. (seen in 8)>

<Settle the interaction loop: how the model gathers evidence (including cold start), what it outputs, and how user feedback updates it over time. (seen in 6)>

<Settle how vague goals become measurable: metrics, rating scales, splits, budgets, thresholds, scoring weights and their calibration, and local stand-ins for hidden tests. (seen in 6)>

<Settle the audience, accounts and login, which source material stays out, whether user data persists, where, and who may read it. (seen in 5)>

<Settle data and reference sources, licenses, keys, citation standards, refresh schedule, stored history, and an offline fallback snapshot before building. (seen in 4)>

<Settle how unmeasurable targets are estimated from proxies, and how assumptions, uncertainty ranges and limitations are disclosed to users. (seen in 2)>

<Settle the concrete task, observations or sensors, action space, and units, matching the conventions of the chosen model or framework. (seen in 2)>

<Settle safety boundaries for high-stakes advice: disclaimers, refusal and escalation rules, emergency handling, and applicable privacy or regulatory constraints. (tentative)>

CONTEXT:

Project type: ML / data model (training and evaluating a model on data, with reproducible runs).

use the image below. combine maml and pearl to do long-horizon post-training. figure out the best-fit environments to test it on at small scale, then test it in higher-fidelity environments later on. sim to real.

Reference images (look at these; they're part of the description):
- inputs/Dropped image.png

DELIVERABLES:

website detailing the novel combined method + experiments + demo

DEFINITIONS AND DEFERRED:

Answers recorded (2026-10-20): purpose = research proposal/preprint + personal
learning + portfolio. "Long-horizon" = all three meanings, as ONE method tested
on three axes. Envs = classic meta-RL suite, chosen after a lit check. Sim-to-real
and high fidelity = clearly labeled roadmap only (deferred). Site = static,
Distill-style article on GitHub Pages; demo = in-browser rollout + recorded
rollouts + latent-space explorer. Repo = public github.com/amalmehta/MEHTAlearning,
pushed. Compute = 2–4 h on this Mac (Intel i9, 16 threads, no GPU), 3 seeds.
Baselines = PEARL, MAML, PEARL + naive fine-tune. Method name = MEHTA.

- MEHTA (the method): "Meta-gradient + Embedded Hidden-Task Adaptation".
  PEARL (Rakelly et al. 2019, arXiv:1903.08254) context encoder q(z|c) + SAC
  actor/critics conditioned on z, PLUS a first-order MAML/Reptile
  (Finn et al. 2017, arXiv:1703.03400; Nichol et al. 2018, arXiv:1803.02999)
  meta-update of the actor/critic initialization taken through a LONG inner loop
  (K_inner = 50 SAC steps per task), with an iMAML-style proximal term
  (Rajeswaran et al. 2019, arXiv:1909.04630) that keeps post-trained weights near
  the meta-init.
- Post-training (test-time adaptation): on a new task, (1) collect context and
  infer z (PEARL, zero gradient steps), then (2) run K SAC gradient steps from the
  meta-init on the task's own online data, re-inferring z from all context, with
  proximal weight λ. K ∈ {0, 25, 50, 100, 200, 400}.
- Long horizon, axis (a), many-step adaptation: return after K post-training
  steps, K up to 400, on in-distribution (ID) and out-of-distribution (OOD) tasks.
- Long horizon, axis (b), long sparse episodes: a 3-waypoint sparse-reward
  episode (horizon 60, reward +1 per waypoint reached). Metric = mean waypoints
  reached (0–3) and success rate (all 3).
- Long horizon, axis (c), continual: 5 OOD tasks post-trained in sequence
  (K = 100 each, weights carried over). Metrics = final mean return over all 5,
  and forgetting = mean over tasks i<5 of (return right after task i) − (return
  at end).
- Small-scale envs (level 1): Point-Goal, modeled on PEARL's sparse point robot
  (origin start, actions in ±0.1, goals on the unit circle, H = 20) but with the
  DENSE reward −‖p − g‖. ID = goal angle in [0, π]; OOD-goal = angle in (π, 2π);
  OOD-dynamics = ID goals with actions rotated 60° (a sim-to-real-style dynamics
  shift). Point-Waypoints for axis (b).
- Higher fidelity (level 2): HalfCheetah-Vel (MuJoCo), a pilot run only if the
  measured throughput fits the budget; otherwise deferred.
- "Beats": MEHTA mean > baseline mean, AND the gap is larger than the sum of the
  two across-seed std devs, on the stated metric. With only 3 seeds, anything
  weaker is reported as "no clear difference".
- Exploration vs confirmation: hyperparameters tuned only on dev seed 100 and on
  ID validation tasks. Confirmatory seeds 0, 1, 2 are run once, and every result
  is reported, including negative ones.
- Done (MVP): `uv run python -m mehta.run_all` reproduces every result JSON and
  figure; the site shows the method, all results (mean ± std), the three demos,
  and the roadmap; tests pass.
- Level-2 plan (not yet run): docs/LEVEL-2-PLAN.md.
- DEFERRED: Meta-World, Isaac Gym / high-fidelity robot sims, domain-randomized
  sim-to-sim and sim-to-real on hardware, compute-matched baselines,
  >3 seeds, cloud GPU, second-order meta-gradients, LLM/foundation-model
  post-training.

RESULTS (2026-10-20, seeds 0–2, confirmatory):

- Latent methods ≫ first-order MAML without a latent (OOD goals −13.9 ± 1.4 vs −29.6 ± 1.6).
- MEHTA beats PEARL alone on ID, OOD-goal and OOD-dynamics after post-training.
- MEHTA vs PEARL + naive fine-tune: no clear difference anywhere. Long vs short
  inner loop: no clear difference. The novel meta-update is NOT supported at this
  scale.
- Continual: MEHTA forgets less than PEARL + fine-tune (−0.40 ± 0.67 vs
  1.12 ± 0.26), but no-proximal does as well, so the gain is not the proximal term.
- Point-Waypoints: every method fails (0% success).

RESULTS v2 (2026-10-20, diagnosed redesign, fresh seeds 3–5):

- Diagnosis of v1 (results/diagnostics): inner-loop gain on training tasks fell
  to +0.39 late in training; the MAML step was ~7% the size of PEARL's updates;
  tasks' inner-loop updates were nearly orthogonal (cos ≈ 0.03).
- MEHTA v2 (test-protocol inner loop on perturbed tasks): no clear difference vs
  compute-matched PEARL + fine-tune on any set; worse than the PEARL-DR control
  on shifted dynamics. The MAML step is still not supported.
- Potential-based Waypoints shaping: success 0% → 27–60% for all methods.

PRE-REGISTERED v3 (written 2026-10-21, before any v3 run):

- Goal: a held-out shift where fine-tuning from PEARL's init is clearly too slow.
- Candidates (all ID goal angles unless noted, never used in training): actions
  rotated 120°; inverted (180°); mirrored left-right; wind = constant drift 0.06
  per step (2× the training max) in a random direction per task; OOD goals +
  60° rotation.
- Oracle per task: a controller that knows the goal and the dynamics, moving
  straight at the goal (per-axis clipped); return over 20 steps.
- Gap closure = (R(K=400) − R(K=0)) / (oracle − R(K=0)).
- Selection uses ONLY compute-matched PEARL + fine-tune (pearl_cm_ft) on dev
  seed 100: pick the candidate with the lowest gap closure among those whose
  K=400 return is still ≥ 30% of the oracle gap short. MEHTA is not evaluated
  during selection.
- Confirmation: pearl_cm, pearl_dr_cm, mehta_v2 trained once on fresh seeds 6–8,
  evaluated on id, ood_goal, ood_dyn and the selected shift, same protocol.
- H (primary): MEHTA v2 beats BOTH pearl_cm_ft and pearl_dr_cm_ft on the
  selected shift at K=400 by the spec rule (gap > summed seed std).
  Secondary: gap closure for each method on that shift.
- Whatever the result, it is reported; seeds 6–8 are not reused for tuning.

RESULTS v3 (2026-10-21, pre-registered, fresh seeds 6–8):

- Selection (dev seed 100, PEARL only): mirrored actions, closure 21%, 79% short.
- H NOT supported: on mirror at K=400, MEHTA v2 −21.9 ± 1.2 beats PEARL-DR + FT
  (−24.8 ± 1.2) but not compute-matched PEARL + FT (−23.5 ± 2.6).
- MEHTA v2's edge is its K=0 start (−30.8 vs −34.6); gap closure 33% vs 36%.
- Rotation 60°: MEHTA v2 now worse than both fine-tuned baselines.
- Oracle reference (greedy, knows goal and dynamics): about −3.7 to −3.9.

OPEN QUESTIONS / ASSUMPTIONS:

- Assumed SAC v2 (twin Q + automatic entropy) in place of PEARL's original
  V-network SAC; MAML baseline is first-order and off-policy (SAC inner loop), not
  the original TRPO MAML. Both are labeled on the site.
- Hyperparameters are chosen or lightly tuned (dev seed only), not swept; labeled
  as such.
- Related prior work also combines latents with gradient adaptation (CAVIA
  arXiv:1810.03642; MAESN arXiv:1802.07245). The site states this plainly and
  positions MEHTA's difference: an amortized PEARL posterior + long first-order
  inner loop + proximal continual post-training.
- Dev-seed findings (exploration only, seed 100): SAC's learned temperature now
  starts at α = 0.1. With α = 1, a single-task run reached a return of −7.9 vs
  −4.0 after 3k steps. The machine was shared with other heavy workloads (load
  average about 50), so budgets were cut: hidden width 64, 50 updates per
  iteration, the MAML step on 2 tasks every 2nd iteration. Also: KL weight
  0.001 (PEARL's 0.1 assumes reward scale 100; dev PEARL scored −13.5 vs −15.2),
  and training lengthened to 300 iterations × 75 updates, because an oracle with
  the true goal learned fast (−6.3) and the bottleneck was task inference.
  Learning rate 1e-3 and a reward-decoder auxiliary loss gave no gain, so
  neither is used. λ = 0.5 and β = 0.3 were chosen, not tuned.
- No examples you loved or hated were given; the Distill-style direction was
  chosen from mockups.
- The website is not an app: it gets a favicon, a theme toggle and a small
  feedback link (GitHub issues), with no full settings pane.

CHANGELOG:

- 2026-10-20 — created
- 2026-09-15 — added meta-instruction: built-out applications include a small feedback tab
- 2026-09-15 — added meta-instruction: no "Claude" attribution in commits, PRs, or branches
- 2026-09-16 — added meta-instruction: always include a README when adding to GitHub
- 2026-09-16 — changed meta-instruction: ask clarifying questions in the question widget
- 2026-09-17 — added meta-instructions: Claude never a contributor; never commit .claude/
- 2026-09-26 — compressed the meta-instructions and every field prompt; git rules moved to the global instruction file
- 2026-09-27 — added meta-instruction: applications are built as a Mac app first, then a website
- 2026-09-28 — folded inputs, instructions, constraints, deliverables and done criteria into one free-form CONTEXT
- 2026-09-28 — changed meta-instruction: a README on GitHub always includes a visual
- 2026-09-28 — added meta-instruction: name things like a person would, never snake_case
- 2026-09-28 — changed meta-instruction: README leads with visuals; instructions live in a linked guide
- 2026-09-28 — changed meta-instruction: README is visuals and links; details in docs/INSTRUCTIONS.md and docs/FILE-STRUCTURE.md
- 2026-09-29 — changed meta-instruction: GitHub repo names have no "_" or "-"
- 2026-10-02 — added a DELIVERABLES field after CONTEXT
- 2026-10-02 — added meta-instruction: every project has a system design doc at docs/SYSTEM-DESIGN.md
- 2026-10-05 — added meta-instruction: applications include a settings button or tab
- 2026-10-08 — added meta-instruction: color-code tags by similarity (related share a hue family, each distinct)
- 2026-10-09 — added meta-instruction: apps that take files have an Upload button, drop anywhere, and paste
- 2026-10-17 — added meta-instruction: before building, ask the design questions an expert engineer would (data, security, caching, scale, deployment…)
- 2026-10-18 — added meta-instruction: open the MVP for me as soon as a first working version exists, and after each big change
- 2026-10-18 — added meta-instruction: always create a relevant icon for applications
- 2026-10-18 — added meta-instructions: work like a world-class founder, software engineer and (for research) scientist
- 2026-10-18 — added meta-instruction: pick my brain with open questions answered in my own words, as well as multiple choice
- 2026-10-18 — changed meta-instruction: Mac app first only when the request names no platform (from the meta report)
- 2026-10-18 — changed meta-instruction: the first batch of questions also covers delivery, the repo, test versus live mode, who pays, mocks, and real inputs and credentials (from the meta report)
- 2026-10-18 — added meta-instructions: nothing made up passes as fact; fill in DEFINITIONS AND DEFERRED; a clean, restrained interface with mockups and a manual review checklist (from the meta report)
- 2026-10-18 — added a DEFINITIONS AND DEFERRED section
- 2026-10-19 — changed meta-instruction: questions mostly as multiple choice (with a box for my own answer); plain-text questions only when a detailed answer is needed, at most two per round
- 2026-10-20 — added meta-instruction: make the UI/UX buttery smooth (instant response, nothing blocking the main thread, full-frame-rate motion, no flicker or jumps)
- 2026-10-20 — filled DEFINITIONS AND DEFERRED and OPEN QUESTIONS from the first question rounds
- 2026-10-20 — built MVP: method, 24 confirmatory runs, website with demo; results recorded (long-horizon MAML step: no measurable gain)
- 2026-10-20 — added the level-2 plan (HalfCheetah-Vel, Meta-World ML1/ML10; compute-matched; K_inner sweep; 5 seeds; ≈$125–285 on Modal)
- 2026-10-20 — v2: diagnosed v1, redesigned the MAML step, added compute-matched and DR controls, fixed Waypoints shaping; confirmed on fresh seeds 3–5 (MAML step: still no gain; Waypoints: now solvable)
- 2026-10-21 — v3: pre-registered hard-shift test (mirrored actions, chosen with PEARL only on dev seed), confirmed on seeds 6–8: not supported; corrected v2's ceiling claim using a measured oracle

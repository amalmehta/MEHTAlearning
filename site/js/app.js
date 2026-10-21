import { pearlEpisodes, rolloutFixed, PointGoal } from "./model.js";
import { barChart, cssVar, el, hideTip, lineChart, showTip } from "./charts.js";

const $ = (s) => document.querySelector(s);
const redraws = new Set();                 // canvas painters to re-run on theme / resize
const redrawAll = () => redraws.forEach((f) => f());

// ---------- theme ----------
$("#theme").addEventListener("click", () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("theme", root.dataset.theme); } catch (e) {}
  requestAnimationFrame(redrawAll);
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => requestAnimationFrame(redrawAll));

// ---------- canvas helpers ----------
function fitCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth || cv.width;
  if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(w * dpr); }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, size: w };
}
const ro = new ResizeObserver(() => requestAnimationFrame(redrawAll));

function world(size, extent = 1.45) {
  const s = size / (2 * extent);
  return { X: (x) => size / 2 + x * s, Y: (y) => size / 2 - y * s, s };
}

function drawArena(ctx, size, opts = {}) {
  const { X, Y, s } = world(size);
  ctx.clearRect(0, 0, size, size);
  ctx.strokeStyle = cssVar("--rule"); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(X(-1.4), Y(0)); ctx.lineTo(X(1.4), Y(0)); ctx.moveTo(X(0), Y(-1.4)); ctx.lineTo(X(0), Y(1.4)); ctx.stroke();
  // trained-on half circle (solid, thick) vs never-seen half (dashed)
  ctx.strokeStyle = cssVar("--train"); ctx.lineWidth = 7; ctx.lineCap = "round";
  ctx.beginPath(); ctx.arc(X(0), Y(0), s, Math.PI, 2 * Math.PI); ctx.stroke();
  ctx.lineWidth = 1.5; ctx.setLineDash([4, 5]);
  ctx.beginPath(); ctx.arc(X(0), Y(0), s, 0, Math.PI); ctx.stroke(); ctx.setLineDash([]);
  if (opts.labels) {
    ctx.fillStyle = cssVar("--muted"); ctx.font = "11px Inter, sans-serif"; ctx.textAlign = "center";
    ctx.fillText("trained goals", X(0), Y(1.12));
    ctx.fillText("never seen in training", X(0), Y(-1.2));
  }
  ctx.fillStyle = cssVar("--text");
  ctx.beginPath(); ctx.arc(X(0), Y(0), 3, 0, 7); ctx.fill();
}

function drawGoal(ctx, size, gx, gy, color) {
  const { X, Y, s } = world(size);
  ctx.fillStyle = color; ctx.globalAlpha = 0.15;
  ctx.beginPath(); ctx.arc(X(gx), Y(gy), 0.2 * s, 0, 7); ctx.fill();
  ctx.globalAlpha = 1; ctx.strokeStyle = color; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.arc(X(gx), Y(gy), 0.2 * s, 0, 7); ctx.stroke();
  ctx.beginPath(); ctx.arc(X(gx), Y(gy), 3.5, 0, 7); ctx.fillStyle = color; ctx.fill();
}

function drawPath(ctx, size, xy, color, { width = 2, alpha = 1, dash = null, upto = xy.length } = {}) {
  const { X, Y } = world(size);
  ctx.save();
  ctx.strokeStyle = color; ctx.globalAlpha = alpha; ctx.lineWidth = width; ctx.lineJoin = "round"; ctx.lineCap = "round";
  if (dash) ctx.setLineDash(dash);
  ctx.beginPath();
  for (let i = 0; i < Math.min(upto, xy.length); i++) (i ? ctx.lineTo : ctx.moveTo).call(ctx, X(xy[i][0]), Y(xy[i][1]));
  ctx.stroke();
  ctx.restore();
}

// ---------- data ----------
const getJSON = (u) => fetch(u).then((r) => (r.ok ? r.json() : null)).catch(() => null);
const demoP = getJSON("data/demo.json");
const resultsP = getJSON("data/results.json");

drawMethodDiagram();
demoP.then((demo) => (demo ? initDemo(demo) : emptyNote("#demo .demo", "Demo weights not generated yet: run uv run python -m mehta.export_demo")));
resultsP.then((res) => (res ? initResults(res) : emptyNote("#chart-a", "No results yet: run uv run python -m mehta.run_all")));
getJSON("data/results_v2.json").then((res) => (res ? initResults(res, V2_IDS) : emptyNote("#chart-a-v2", "Version 2 confirmatory runs are not finished yet.")));
getJSON("data/results_v3.json").then((res) => (res ? (initResults(res, V3_IDS), closureTable(res)) : emptyNote("#chart-a-v3", "Version 3 confirmatory runs are not finished yet.")));
getJSON("data/shift_selection.json").then((sel) => sel && selectionTable(sel));
getJSON("data/shift.json").then((sh) => (sh ? initShift(sh) : emptyNote("#chart-s2r", "Shift-curve runs are not finished yet: run uv run python -m mehta.run_all --version shift --seeds 9 10 11")));

function initShift(sh) {
  const pt = $("#s2r-params");
  const names = { mass: "Mass", friction: "Friction", gain: "Actuator gain", obs_noise: "Observation noise (std)", delay: "Action delay (steps)" };
  const at = (k, s) => { const [nom, lo, hi, kind] = sh.params[k]; return kind === "log" ? `×${((lo / nom) ** s).toFixed(2)} or ×${((hi / nom) ** s).toFixed(2)}` : kind === "lin" ? (nom + s * (hi - nom)).toFixed(2) : Math.round(nom + s * (hi - nom)); };
  pt.innerHTML = `<thead><tr><th>Parameter</th><th>Nominal</th><th>Training range</th><th>Test at s = 2</th><th>Test at s = 3</th></tr></thead><tbody>` +
    Object.entries(sh.params).map(([k, [nom, lo, hi]]) => `<tr><td>${names[k] || k}</td><td class="n">${nom}</td><td class="n">${lo} – ${hi}</td><td class="n">${at(k, 2)}</td><td class="n">${at(k, 3)}</td></tr>`).join("") + "</tbody>";
  const order = ["mehta_phys", "pearl_phys_ft", "pearl_phys", "maml_phys"];
  const kmax = String(Math.max(...sh.ckpt));
  const hidden = new Set(), allGroups = [];
  for (const [env, byV] of Object.entries(sh.curves)) {
    for (const K of ["0", kmax]) {
      const series = order.filter((v) => byV[v]).map((v) => ({ key: v, label: sh.labels[v],
        points: Object.keys(byV[v]).sort((a, b) => a - b).map((s) => [Number(s), ...byV[v][s][K]]) }));
      const { groups } = lineChart($("#chart-s2r"), series, { title: `${sh.env_labels[env]}, ${K === "0" ? "before post-training (K=0)" : `after K=${K} steps`}`,
        xName: "s", xLabel: "shift magnitude s", xTicks: sh.magnitudes, shadeTo: sh.train_edge, shadeLabel: "training range" });
      allGroups.push(groups);
    }
  }
  const leg = $("#legend-s2r");
  for (const v of order.filter((v) => Object.values(sh.curves).some((d) => d[v]))) {
    const b = document.createElement("button");
    b.innerHTML = `<span class="dot c-${v}"></span>${sh.labels[v]}`;
    b.onclick = () => {
      hidden.has(v) ? hidden.delete(v) : hidden.add(v);
      b.classList.toggle("off", hidden.has(v));
      allGroups.forEach((g) => g[v] && (g[v].style.display = hidden.has(v) ? "none" : ""));
    };
    leg.appendChild(b);
  }
}

function selectionTable(sel) {
  const t = $("#v3-selection");
  if (!t) return;
  const L = { shift_rot120: "Actions rotated 120°", shift_invert: "Actions inverted (180°)", shift_mirror: "Actions mirrored left-right",
              shift_wind: "Strong wind (drift 0.06/step)", shift_ood_rot60: "New goals + actions rotated 60°" };
  t.innerHTML = `<thead><tr><th>Candidate shift</th><th>PEARL+FT at K=0</th><th>at K=400</th><th>Oracle</th><th>Gap closed</th><th>Eligible</th></tr></thead><tbody>` +
    Object.entries(sel.rows).map(([k, r]) => `<tr${k === sel.chosen ? ' class="chosen"' : ""}><td>${L[k] || k}${k === sel.chosen ? " <b>← chosen</b>" : ""}</td>` +
      `<td class="n">${r.return["0"].toFixed(2)}</td><td class="n">${r.return["400"].toFixed(2)}</td><td class="n">${r.oracle.toFixed(2)}</td>` +
      `<td class="n">${(r.closure * 100).toFixed(0)}%</td><td>${r.eligible ? "yes" : "no"}</td></tr>`).join("") + "</tbody>";
}

function closureTable(res) {
  const t = $("#v3-closure");
  if (!t || !res.closure) return;
  const vs = ["mehta_v2", "pearl_cm_ft", "pearl_dr_cm_ft"];
  t.innerHTML = `<thead><tr><th>Tasks</th><th>Oracle</th>${vs.map((v) => `<th>${res.labels[v]}</th>`).join("")}</tr></thead><tbody>` +
    Object.entries(res.closure).map(([s, d]) => `<tr><td>${res.set_labels[s] || s}</td><td class="n">${res.oracle[s].toFixed(2)}</td>` +
      vs.map((v) => `<td class="n">${d[v] != null ? (d[v] * 100).toFixed(0) + "%" : "–"}</td>`).join("") + "</tr>").join("") + "</tbody>";
}

function emptyNote(sel, text) {
  const n = $(sel);
  if (n) n.insertAdjacentHTML("afterbegin", `<div class="empty">${text}</div>`);
}

// ---------- live demo ----------
function initDemo(demo) {
  const cv = $("#demo-canvas");
  const state = { model: "mehta", angle: null, eps: null, frame: 0, raf: 0, seed: 1 };
  const rot = () => ($("#demo-rot").checked ? Math.PI / 3 : 0);
  ro.observe(cv);

  const paint = () => {
    const { ctx, size } = fitCanvas(cv);
    drawArena(ctx, size, { labels: true });
    if (state.angle === null) return;
    const accent = cssVar(`--c-${state.model}`);
    drawGoal(ctx, size, Math.cos(state.angle), Math.sin(state.angle), cssVar("--text"));
    if (!state.eps) return;
    let f = state.frame;
    const styles = [{ width: 1.5, alpha: 0.35, dash: [3, 4] }, { width: 1.5, alpha: 0.55 }, { width: 3, alpha: 1 }];
    state.eps.forEach((ep, i) => {
      const n = ep.xy.length;
      if (f <= 0) return;
      drawPath(ctx, size, ep.xy, accent, { ...styles[i], upto: Math.min(n, f) });
      f -= n;
    });
  };
  redraws.add(paint);

  const latentSvg = $("#demo-latent");
  const drawLatent = (post) => {
    latentSvg.innerHTML = "";
    const d = demo.models[state.model].latent_dim, W = 260, H = 120, bw = W / d;
    const sy = (v) => H / 2 - v * 18;
    el("line", { x1: 0, x2: W, y1: H / 2, y2: H / 2, style: "stroke:var(--rule)" }, latentSvg);
    for (let i = 0; i < d; i++) {
      const mu = post ? post.mu[i] : 0, sd = post ? Math.sqrt(post.var[i]) : 1;
      const cx = i * bw + bw / 2;
      el("rect", { x: cx - bw * 0.28, y: Math.min(sy(mu + sd), sy(mu - sd)), width: bw * 0.56, height: Math.abs(sy(mu + sd) - sy(mu - sd)), rx: 3,
        style: `fill:var(--c-${state.model});opacity:.18` }, latentSvg);
      el("line", { x1: cx - bw * 0.3, x2: cx + bw * 0.3, y1: sy(mu), y2: sy(mu), style: `stroke:var(--c-${state.model});stroke-width:3;stroke-linecap:round` }, latentSvg);
      el("text", { x: cx, y: H - 4, "text-anchor": "middle", style: "font:10px Inter,sans-serif;fill:var(--muted)" }, latentSvg).textContent = `z${i + 1}`;
    }
  };
  drawLatent(null);

  const optimalReturn = (angle, rotation) => {
    const env = new PointGoal(angle, rotation);
    let o = env.reset(), ret = 0, done = false;
    const c = Math.cos(-rotation), s = Math.sin(-rotation);
    while (!done) {
      const dx = (env.goal[0] - o[0]) / 0.1, dy = (env.goal[1] - o[1]) / 0.1;
      const st = env.step([c * dx - s * dy, s * dx + c * dy]);
      ret += st.r; o = st.obs; done = st.done;
    }
    return ret;
  };

  const run = () => {
    cancelAnimationFrame(state.raf);
    state.seed += 1;
    state.eps = pearlEpisodes(demo.models[state.model], state.angle, rot(), state.seed);
    state.frame = 0;
    const lat = [...state.eps[0].latents, ...state.eps[1].latents];
    const total = state.eps.reduce((a, e) => a + e.xy.length, 0);
    $("#demo-ret").textContent = "…";
    $("#demo-opt").textContent = optimalReturn(state.angle, rot()).toFixed(2);
    let last = performance.now();
    const tick = (now) => {
      // ~90 steps/s, frame-rate independent
      state.frame = Math.min(total, state.frame + Math.max(1, Math.round((now - last) / 11)));
      last = now;
      const li = Math.min(lat.length - 1, state.frame - 2);
      if (li >= 0) drawLatent(lat[li]);
      paint();
      if (state.frame < total) state.raf = requestAnimationFrame(tick);
      else $("#demo-ret").textContent = state.eps[2].ret.toFixed(2);
    };
    state.raf = requestAnimationFrame(tick);
  };

  cv.addEventListener("pointerdown", (e) => {
    const r = cv.getBoundingClientRect();
    const { s } = world(r.width);
    const x = (e.clientX - r.left - r.width / 2) / s, y = -(e.clientY - r.top - r.height / 2) / s;
    if (Math.hypot(x, y) < 0.05) return;
    state.angle = Math.atan2(y, x);
    $("#demo-hint").classList.add("hide");
    run();
  });
  document.querySelectorAll("#demo .seg button[data-model]").forEach((b) =>
    b.addEventListener("click", () => {
      document.querySelectorAll("#demo .seg button[data-model]").forEach((x) => x.classList.toggle("on", x === b));
      state.model = b.dataset.model;
      if (state.angle !== null) run(); else drawLatent(null);
    }));
  $("#demo-rot").addEventListener("change", () => state.angle !== null && run());
  paint();

  initSnapshots(demo);
}

function initSnapshots(demo) {
  const variants = Object.keys(demo.adapted);
  const tasks = demo.adapted[variants[0]];
  const Ks = Object.keys(tasks[0].K).sort((a, b) => a - b);
  const slider = $("#snap-k");
  slider.max = Ks.length - 1;
  let ti = 0;
  const seg = $("#snap-task");
  tasks.forEach((t, i) => {
    const b = document.createElement("button");
    b.textContent = `Goal ${i + 1}`;
    b.className = i === 0 ? "on" : "";
    b.onclick = () => { ti = i; seg.querySelectorAll("button").forEach((x, j) => x.classList.toggle("on", j === i)); paintAll(); };
    seg.appendChild(b);
  });
  const painters = variants.map((v) => {
    const cv = $(`#snap-${v}`);
    if (!cv) return () => {};
    ro.observe(cv);
    return () => {
      const K = Ks[+slider.value];
      const t = demo.adapted[v][ti];
      const snap = t.K[K];
      const res = rolloutFixed(snap.actor, snap.z, t.task.angle, t.task.rotation);
      const { ctx, size } = fitCanvas(cv);
      drawArena(ctx, size);
      drawGoal(ctx, size, Math.cos(t.task.angle), Math.sin(t.task.angle), cssVar("--text"));
      drawPath(ctx, size, res.xy, cssVar(`--c-${v}`), { width: 3 });
      $(`#snap-${v}-ret`).textContent = res.ret.toFixed(2);
    };
  });
  const paintAll = () => { $("#snap-k-label").textContent = Ks[+slider.value]; painters.forEach((p) => p()); };
  slider.addEventListener("input", paintAll);
  painters.forEach((p) => redraws.add(p));
  paintAll();
}

// ---------- results ----------
const V1_IDS = { a: "#chart-a", legend: "#legend-a", b: "#chart-b", c: "#chart-c", verdicts: "#verdicts", traj: true, train: "#chart-train" };
const V2_IDS = { a: "#chart-a-v2", legend: "#legend-a-v2", b: "#chart-b-v2", c: "#chart-c-v2", verdicts: "#verdicts-v2" };
const V3_IDS = { a: "#chart-a-v3", legend: "#legend-a-v3", b: "#chart-b-v3", c: "#chart-c-v3", verdicts: "#verdicts-v3" };

function initResults(res, ids = V1_IDS) {
  const L = res.labels;
  const hero = L[res.hero || "mehta"];
  const order = ["mehta_v2", "mehta", "mehta_short", "pearl_dr_cm_ft", "pearl_cm_ft", "pearl_ft", "pearl_dr_cm", "pearl_cm", "pearl", "maml", "mehta_lam0", "mehta_v2_lam0"];
  const toSeries = (byVariant) =>
    order.filter((v) => byVariant[v]).map((v) => ({
      key: v, label: L[v],
      points: Object.keys(byVariant[v]).map(Number).sort((a, b) => a - b).map((k) => [k, ...byVariant[v][k]]),
    }));

  // (a)
  const hidden = new Set();
  const allGroups = [];
  const ca = $(ids.a);
  if (res.axis_a) {
    for (const s of Object.keys(res.set_labels)) {
      if (!res.axis_a[s]) continue;
      const { groups } = lineChart(ca, toSeries(res.axis_a[s]), { title: res.set_labels[s], xName: "K", xLabel: "post-training steps K", xTicks: [0, 100, 200, 400] });
      allGroups.push(groups);
    }
    const leg = $(ids.legend);
    for (const v of order.filter((v) => res.axis_a.id?.[v])) {
      const b = document.createElement("button");
      b.innerHTML = `<span class="dot c-${v}"></span>${L[v]}`;
      b.onclick = () => {
        hidden.has(v) ? hidden.delete(v) : hidden.add(v);
        b.classList.toggle("off", hidden.has(v));
        allGroups.forEach((g) => g[v] && (g[v].style.display = hidden.has(v) ? "none" : ""));
      };
      leg.appendChild(b);
    }
  }
  // (b)
  if (res.axis_b) {
    lineChart($(ids.b), toSeries(res.axis_b.waypoints), { title: "Waypoints reached (of 3)", xName: "K", xLabel: "post-training steps K", yMin: 0, yMax: 3, xTicks: [0, 100, 200, 400] });
    lineChart($(ids.b), toSeries(res.axis_b.success), { title: "Success rate (all 3 reached)", xName: "K", xLabel: "post-training steps K", yMin: 0, yMax: 1, xTicks: [0, 100, 200, 400] });
  } else if (ids === V1_IDS) emptyNote(ids.b, "No Point-Waypoints results yet.");
  // (c)
  if (res.axis_c && Object.keys(res.axis_c).length) {
    const vs = order.filter((v) => res.axis_c[v]);
    barChart($(ids.c), vs.map((v) => ({ key: v, label: L[v], mean: res.axis_c[v].final[0], std: res.axis_c[v].final[1] })), { title: "Final mean return over the 5 tasks (higher is better)" });
    barChart($(ids.c), vs.map((v) => ({ key: v, label: L[v], mean: res.axis_c[v].forgetting[0], std: res.axis_c[v].forgetting[1] })), { title: "Forgetting (lower is better)" });
  }
  // verdicts
  const tbl = $(ids.verdicts);
  const f = (x) => `${x[0].toFixed(2)} ± ${x[1].toFixed(2)}`;
  const axisName = { a: "(a) many-step", b: "(b) long sparse", c: "(c) continual" };
  tbl.innerHTML = `<thead><tr><th>Axis</th><th>Tasks</th><th>${hero}</th><th>vs.</th><th>Other</th><th>Verdict</th></tr></thead>` +
    `<tbody>${res.verdicts.map((v) => `<tr><td>${axisName[v.axis]}</td><td>${res.set_labels[v.set] || (v.set === "final" ? "final return" : v.set === "forgetting" ? "forgetting (lower is better)" : v.set === "waypoints_success" ? "Point-Waypoints, success rate" : "Point-Waypoints, waypoints reached")}${v.K ? `, K=${v.K}` : ""}</td>` +
      `<td class="n">${f(v.mehta)}</td><td>${L[v.vs]}</td><td class="n">${f(v.other)}</td>` +
      `<td class="${v.verdict === "better" ? "v-better" : v.verdict === "worse" ? "v-worse" : "v-none"}">${v.verdict === "better" ? `${hero} better` : v.verdict === "worse" ? `${hero} worse` : "no clear difference"}</td></tr>`).join("")}</tbody>`;
  if (!ids.traj) return;
  // trajectories
  initTraj(res);
  // training curves
  for (const [env, curves] of Object.entries(res.train_curves || {})) {
    if (!Object.keys(curves).length) continue;
    const by = {};
    for (const [m, d] of Object.entries(curves)) by[m] = d;
    lineChart($(ids.train), toSeries(by), { title: env === "point_goal" ? "Point-Goal" : "Point-Waypoints (waypoints reached)", xName: "iteration", xLabel: "meta-training iteration",
      xTicks: [0, 100, 200, 300].filter((t) => t <= Math.max(...Object.keys(Object.values(curves)[0]).map(Number)) + 1) });
  }
}

function initTraj(res) {
  if (!res.traj) return;
  const seg = $("#traj-set"), grid = $("#traj-grid");
  const options = [];
  for (const [env, sets] of Object.entries(res.traj))
    for (const s of Object.keys(sets)) options.push({ env, s, label: env === "waypoints" ? "Point-Waypoints" : res.set_labels[s] });
  if (!options.length) return;
  let cur = options.find((o) => o.s === "ood_goal") || options[0];
  const render = () => {
    grid.innerHTML = "";
    redraws.forEach((f) => f.traj && redraws.delete(f));
    const data = res.traj[cur.env][cur.s];
    for (const v of ["mehta", "pearl_ft", "pearl", "maml", "mehta_short"].filter((v) => data[v])) {
      const fig = document.createElement("figure");
      fig.innerHTML = `<canvas></canvas><figcaption><span class="dot c-${v}"></span>${res.labels[v]}</figcaption>`;
      grid.appendChild(fig);
      const cv = fig.querySelector("canvas");
      ro.observe(cv);
      const paint = () => {
        const { ctx, size } = fitCanvas(cv);
        drawArena(ctx, size);
        const col = cssVar(`--c-${v}`);
        for (const t of data[v]) {
          const a = t.task.angle;
          if (cur.env === "waypoints") for (let k = 0; k < 3; k++) drawGoal(ctx, size, Math.cos(a + k * Math.PI / 4), Math.sin(a + k * Math.PI / 4), cssVar("--muted"));
          else drawGoal(ctx, size, Math.cos(a), Math.sin(a), cssVar("--muted"));
          const ks = Object.keys(t.xy).map(Number).sort((x, y) => x - y);
          drawPath(ctx, size, t.xy[ks[0]], col, { width: 1.5, alpha: 0.6, dash: [3, 4] });
          drawPath(ctx, size, t.xy[ks[ks.length - 1]], col, { width: 2.5 });
        }
      };
      paint.traj = true;
      redraws.add(paint);
      paint();
    }
  };
  for (const o of options) {
    const b = document.createElement("button");
    b.textContent = o.label;
    b.className = o === cur ? "on" : "";
    b.onclick = () => { cur = o; seg.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b)); render(); };
    seg.appendChild(b);
  }
  render();
}

// ---------- latent explorer ----------
demoP.then((demo) => {
  if (!demo) return;
  const seg = $("#latent-model"), svg = $("#latent-svg");
  let cur = "mehta";
  const render = () => {
    const L = demo.latent[cur];
    svg.innerHTML = "";
    const pts = L.paths.flat();
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const W = 520, H = 380, m = 24;
    const sx = (x) => m + ((x - x0) / (x1 - x0 || 1)) * (W - 2 * m), sy = (y) => H - m - ((y - y0) / (y1 - y0 || 1)) * (H - 2 * m);
    const hue = (a) => `hsl(${(a * 180) / Math.PI} 62% 50%)`;
    L.paths.forEach((p, i) => {
      const a = L.angles[i], ood = a > Math.PI + 1e-6;
      const g = el("g", { style: "cursor:pointer" }, svg);
      el("path", { d: `M${p.map((q) => `${sx(q[0]).toFixed(1)},${sy(q[1]).toFixed(1)}`).join("L")}`, style: `fill:none;stroke:${hue(a)};stroke-width:1.2;opacity:.45` }, g);
      const end = p[p.length - 1];
      el("circle", { cx: sx(end[0]), cy: sy(end[1]), r: 4.5, style: ood ? `fill:var(--card);stroke:${hue(a)};stroke-width:2` : `fill:${hue(a)}` }, g);
      g.addEventListener("pointerenter", (e) => {
        svg.querySelectorAll("g").forEach((x) => (x.style.opacity = x === g ? 1 : 0.15));
        g.querySelector("path").style.strokeWidth = 2.5; g.querySelector("path").style.opacity = 1;
        showTip(e, `goal angle ${((a * 180) / Math.PI).toFixed(0)}° · ${ood ? "never seen in training" : "training range"}<br>final posterior variance ${end[2].toExponential(1)}`);
      });
      g.addEventListener("pointerleave", () => {
        svg.querySelectorAll("g").forEach((x) => (x.style.opacity = 1));
        g.querySelector("path").style.strokeWidth = 1.2; g.querySelector("path").style.opacity = 0.45; hideTip();
      });
    });
    $("#latent-explained").textContent = `Two PCA components explain ${(L.explained * 100).toFixed(0)}% of the variance of the posterior means (${demo.models[cur].latent_dim}-D latent).`;
  };
  for (const [k, label] of [["mehta", "MEHTA"], ["pearl", "PEARL"]]) {
    const b = document.createElement("button");
    b.textContent = label; b.className = k === cur ? "on" : "";
    b.onclick = () => { cur = k; seg.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b)); render(); };
    seg.appendChild(b);
  }
  render();
});

// ---------- method diagram ----------
function drawMethodDiagram() {
  const host = $("#method-diagram");
  host.innerHTML = `
<svg viewBox="0 0 720 250" role="img" aria-label="Context flows into the PEARL encoder which produces z; the actor and critics take z; the long-horizon MAML outer loop updates the initialization; at test time z is inferred then post-trained">
  <defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" style="fill:var(--muted)"/></marker></defs>
  <style>
    .bx{fill:var(--card);stroke:var(--rule);stroke-width:1.2} .ac{stroke:var(--c-mehta);stroke-width:1.6}
    .t{font:600 13px Inter,sans-serif;fill:var(--text)} .s{font:11.5px Inter,sans-serif;fill:var(--muted)}
    .ar{stroke:var(--muted);stroke-width:1.3;fill:none;marker-end:url(#ah)} .ar2{stroke:var(--c-mehta);stroke-width:1.6;fill:none;marker-end:url(#ah);stroke-dasharray:5 4}
  </style>
  <rect class="bx" x="10" y="30" width="140" height="62" rx="10"/><text class="t" x="80" y="56" text-anchor="middle">Context c</text><text class="s" x="80" y="75" text-anchor="middle">(s, a, r, s′) of this task</text>
  <rect class="bx" x="190" y="30" width="150" height="62" rx="10"/><text class="t" x="265" y="56" text-anchor="middle">Encoder q(z | c)</text><text class="s" x="265" y="75" text-anchor="middle">product of Gaussians</text>
  <rect class="bx" x="380" y="30" width="150" height="62" rx="10"/><text class="t" x="455" y="56" text-anchor="middle">Actor π(a | s, z)</text><text class="s" x="455" y="75" text-anchor="middle">+ critics Q(s, a, z)</text>
  <rect class="bx" x="570" y="30" width="140" height="62" rx="10"/><text class="t" x="640" y="56" text-anchor="middle">Environment</text><text class="s" x="640" y="75" text-anchor="middle">unknown task</text>
  <path class="ar" d="M150,61H186"/><path class="ar" d="M340,61H376"/><text class="s" x="358" y="52" text-anchor="middle">z</text>
  <path class="ar" d="M530,61H566"/><path class="ar" d="M640,92V112H80V96"/><text class="s" x="360" y="126" text-anchor="middle">new transitions refine z (PEARL, zero gradient steps)</text>
  <rect class="bx ac" x="300" y="160" width="310" height="70" rx="10"/><text class="t" x="455" y="186" text-anchor="middle">Long-horizon MAML (first-order)</text>
  <text class="s" x="455" y="206" text-anchor="middle">K_inner = 50 proximal SAC steps per task →</text><text class="s" x="455" y="221" text-anchor="middle">θ ← θ + β · mean(θᵢ − θ)</text>
  <path class="ar2" d="M455,160V96"/><text class="s" x="300" y="150" text-anchor="end">meta-trains the initialization θ</text>
  <text class="s" x="40" y="186">at test time:</text><text class="s" x="40" y="204">1. infer z (K = 0)</text><text class="s" x="40" y="222">2. post-train K steps from θ</text>
</svg>`;
}

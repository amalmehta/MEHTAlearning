// Tiny dependency-free SVG charts. Colors come from CSS variables, so the
// charts follow the light/dark theme with no re-render.

const NS = "http://www.w3.org/2000/svg";
export const el = (tag, attrs = {}, parent) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
};
export const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function niceTicks(lo, hi, n = 4) {
  const span = hi - lo || 1;
  const step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || mag * 10;
  const ticks = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) ticks.push(+t.toFixed(10));
  return ticks;
}
const fmt = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));

const tip = () => document.getElementById("tooltip");
export function showTip(evt, html) {
  const t = tip();
  t.innerHTML = html;
  t.classList.add("show");
  const x = Math.min(window.innerWidth - t.offsetWidth - 8, evt.clientX + 12);
  t.style.left = `${x}px`;
  t.style.top = `${evt.clientY + 14}px`;
}
export const hideTip = () => tip().classList.remove("show");

/**
 * series: [{key, label, points: [[x, mean, std], ...]}]
 * opts: {title, xLabel, yLabel, yMin, yMax, logX, shadeTo, shadeLabel, xFormat}
 * xFormat(x) labels x ticks and tooltips (e.g. plot log2 positions, show the raw value)
 */
export function lineChart(container, series, opts = {}) {
  const box = document.createElement("div");
  box.className = "chart";
  if (opts.title) box.innerHTML = `<h5>${opts.title}</h5>`;
  container.appendChild(box);
  const W = 300, H = 210, m = { l: 38, r: 10, t: 8, b: 34 };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.title || "chart" }, box);
  const all = series.flatMap((s) => s.points);
  const xs = all.map((p) => p[0]);
  const lo = opts.yMin ?? Math.min(...all.map((p) => p[1] - p[2]));
  const hi = opts.yMax ?? Math.max(...all.map((p) => p[1] + p[2]));
  const pad = (hi - lo) * 0.06 || 1;
  const y0 = opts.yMin ?? lo - pad, y1 = opts.yMax ?? hi + pad;
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const sx = (x) => m.l + ((x - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
  const sy = (y) => H - m.b - ((y - y0) / (y1 - y0 || 1)) * (H - m.t - m.b);
  if (opts.shadeTo != null) {
    el("rect", { x: sx(x0), y: m.t, width: sx(opts.shadeTo) - sx(x0), height: H - m.t - m.b, class: "shade" }, svg);
    if (opts.shadeLabel) el("text", { x: (sx(x0) + sx(opts.shadeTo)) / 2, y: m.t + 10, "text-anchor": "middle" }, svg).textContent = opts.shadeLabel;
  }
  for (const t of niceTicks(y0, y1)) {
    el("line", { x1: m.l, x2: W - m.r, y1: sy(t), y2: sy(t), class: "grid" }, svg);
    el("text", { x: m.l - 5, y: sy(t) + 3.5, "text-anchor": "end" }, svg).textContent = fmt(t);
  }
  const xt = opts.xTicks || [...new Set(xs)].sort((a, b) => a - b);
  const xf = opts.xFormat || ((x) => x);
  for (const t of xt) el("text", { x: sx(t), y: H - m.b + 14, "text-anchor": "middle" }, svg).textContent = xf(t);
  if (opts.xLabel) el("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle" }, svg).textContent = opts.xLabel;
  const groups = {};
  for (const s of series) {
    const g = el("g", { class: "series", "data-key": s.key }, svg);
    const col = `var(--c-${s.key})`;
    const up = s.points.map((p) => `${sx(p[0])},${sy(p[1] + p[2])}`);
    const dn = s.points.map((p) => `${sx(p[0])},${sy(p[1] - p[2])}`).reverse();
    el("path", { d: `M${up.join("L")}L${dn.join("L")}Z`, class: "band", style: `fill:${col}` }, g);
    el("path", { d: `M${s.points.map((p) => `${sx(p[0])},${sy(p[1])}`).join("L")}`, class: "line", style: `stroke:${col}` }, g);
    for (const p of s.points) {
      const c = el("circle", { cx: sx(p[0]), cy: sy(p[1]), r: 2.6, style: `fill:${col}` }, g);
      c.addEventListener("pointerenter", (e) => {
        c.setAttribute("r", 4.5);
        box.classList.add("dim"); g.classList.add("hl");
        showTip(e, `<b>${s.label}</b><br>${opts.xName || "x"} = ${xf(p[0])}: ${fmt(p[1])} ± ${fmt(p[2])}${p[3] ? ` (${p[3]} seeds)` : ""}`);
      });
      c.addEventListener("pointerleave", () => { c.setAttribute("r", 2.6); box.classList.remove("dim"); g.classList.remove("hl"); hideTip(); });
    }
    groups[s.key] = g;
  }
  return { box, groups };
}

/** rows: [{key, label, mean, std}] horizontal bars */
export function barChart(container, rows, opts = {}) {
  const box = document.createElement("div");
  box.className = "chart";
  if (opts.title) box.innerHTML = `<h5>${opts.title}</h5>`;
  container.appendChild(box);
  const W = 300, rowH = 26, m = { l: 118, r: 12, t: 4, b: 22 };
  const H = m.t + m.b + rows.length * rowH;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.title || "bars" }, box);
  const vals = rows.flatMap((r) => [r.mean - r.std, r.mean + r.std, 0]);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const sx = (v) => m.l + ((v - lo) / (hi - lo || 1)) * (W - m.l - m.r);
  for (const t of niceTicks(lo, hi, 4)) {
    el("line", { x1: sx(t), x2: sx(t), y1: m.t, y2: H - m.b, class: "grid" }, svg);
    el("text", { x: sx(t), y: H - 6, "text-anchor": "middle" }, svg).textContent = fmt(t);
  }
  rows.forEach((r, i) => {
    const y = m.t + i * rowH;
    const col = `var(--c-${r.key})`;
    const g = el("g", {}, svg);
    el("rect", { x: Math.min(sx(0), sx(r.mean)), y: y + 6, width: Math.abs(sx(r.mean) - sx(0)), height: rowH - 12, rx: 3, style: `fill:${col};opacity:.85` }, g);
    el("line", { x1: sx(r.mean - r.std), x2: sx(r.mean + r.std), y1: y + rowH / 2, y2: y + rowH / 2, style: "stroke:var(--text);stroke-width:1.2" }, g);
    const lab = el("text", { x: m.l - 6, y: y + rowH / 2 + 3.5, "text-anchor": "end" }, g);
    lab.textContent = r.label;
    lab.style.fill = "var(--text)";
    g.addEventListener("pointermove", (e) => showTip(e, `<b>${r.label}</b><br>${fmt(r.mean)} ± ${fmt(r.std)}`));
    g.addEventListener("pointerleave", hideTip);
  });
  return box;
}

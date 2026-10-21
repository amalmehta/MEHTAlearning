// Pure-JS mirror of mehta/envs.py (Point-Goal) and mehta/nets.py (actor, encoder).
// tests/test_export.py checks it against the PyTorch outputs in data/demo.json.

export function mlp(layers, x) {
  let h = x;
  for (let l = 0; l < layers.length; l++) {
    const { W, b } = layers[l];
    const out = new Array(b.length);
    for (let i = 0; i < b.length; i++) {
      let s = b[i];
      const row = W[i];
      for (let j = 0; j < h.length; j++) s += row[j] * h[j];
      out[i] = l < layers.length - 1 && s < 0 ? 0 : s;          // ReLU on hidden layers
    }
    h = out;
  }
  return h;
}

// Deterministic action = tanh(mu); stochastic adds N(0, std) before the tanh.
export function act(actor, obs, z, rand = null) {
  const h = mlp(actor, [...obs, ...z]);
  const n = h.length / 2;
  const a = [];
  for (let i = 0; i < n; i++) {
    let u = h[i];
    if (rand) u += Math.exp(Math.min(2, Math.max(-20, h[n + i]))) * rand();
    a.push(Math.tanh(u));
  }
  return a;
}

const softplus = (x) => (x > 20 ? x : Math.log1p(Math.exp(x)));

// Running product of Gaussians: O(1) per new transition.
export class Posterior {
  constructor(encoder, latentDim) {
    this.enc = encoder;
    this.d = latentDim;
    this.prec = new Array(latentDim).fill(0);
    this.wmu = new Array(latentDim).fill(0);
    this.n = 0;
  }
  add(row) {
    const h = mlp(this.enc, row);
    for (let i = 0; i < this.d; i++) {
      const v = Math.max(1e-7, softplus(h[this.d + i]));
      this.prec[i] += 1 / v;
      this.wmu[i] += h[i] / v;
    }
    this.n++;
  }
  get() {
    if (this.n === 0) return { mu: new Array(this.d).fill(0), var: new Array(this.d).fill(1) };
    const v = this.prec.map((p) => 1 / p);
    return { mu: this.wmu.map((w, i) => w * v[i]), var: v };
  }
  sample(rand) {
    const { mu, var: v } = this.get();
    return mu.map((m, i) => m + Math.sqrt(v[i]) * rand());
  }
}

export class PointGoal {
  constructor(angle, rotation = 0, step = 0.1, horizon = 20) {
    this.goal = [Math.cos(angle), Math.sin(angle)];
    this.rot = rotation;
    this.step_ = step;
    this.horizon = horizon;
    this.reset();
  }
  reset() {
    this.pos = [0, 0];
    this.t = 0;
    return [...this.pos];
  }
  step(action) {
    const ax = Math.max(-1, Math.min(1, action[0])) * this.step_;
    const ay = Math.max(-1, Math.min(1, action[1])) * this.step_;
    const c = Math.cos(this.rot), s = Math.sin(this.rot);
    this.pos = [this.pos[0] + c * ax - s * ay, this.pos[1] + s * ax + c * ay];
    this.t++;
    const r = -Math.hypot(this.pos[0] - this.goal[0], this.pos[1] - this.goal[1]);
    return { obs: [...this.pos], r, done: this.t >= this.horizon };
  }
}

// Seeded normal sampler (mulberry32 + Box-Muller) so a demo run is repeatable.
export function gaussian(seed = 1) {
  let a = seed >>> 0;
  const uni = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return () => Math.sqrt(-2 * Math.log(uni() + 1e-12)) * Math.cos(2 * Math.PI * uni());
}

// Two exploration episodes (prior sample, then posterior sample), then one
// deterministic episode with the posterior mean: the same protocol as Python.
export function pearlEpisodes(model, angle, rotation, seed) {
  const rand = gaussian(seed);
  const post = new Posterior(model.encoder, model.latent_dim);
  const env = new PointGoal(angle, rotation);
  const episodes = [];
  for (let e = 0; e < 3; e++) {
    const final = e === 2;
    const z = final ? post.get().mu : post.sample(rand);
    let o = env.reset();
    const xy = [[0, 0]], latents = [];
    let ret = 0, done = false;
    while (!done) {
      const a = act(model.actor, o, z, final ? null : rand);
      const st = env.step(a);
      ret += st.r;
      if (!final) {
        post.add([...o, ...a, st.r, ...st.obs]);
        latents.push(post.get());
      }
      o = st.obs;
      xy.push(o);
      done = st.done;
    }
    episodes.push({ xy, ret, latents, kind: final ? "exploit" : e === 0 ? "prior" : "posterior" });
  }
  return episodes;
}

export function rolloutFixed(actor, z, angle, rotation = 0) {
  const env = new PointGoal(angle, rotation);
  let o = env.reset(), done = false, ret = 0;
  const xy = [[0, 0]];
  while (!done) {
    const st = env.step(act(actor, o, z));
    ret += st.r;
    o = st.obs;
    xy.push(o);
    done = st.done;
  }
  return { xy, ret };
}

// Web Audio synthesis. Notes use the model's own inharmonic partials (additive synthesis);
// drums are the model's modal percussion. The same code drives live playback and WAV export.
import { timeline } from "./score.js";

export const DEFAULT_SOUND = {
  volume: 0.8,
  tempo: 1,
  transpose: 0, // cents
  warp: 1, // partial-ratio exponent: 1 = the model's timbre, <1 squeezes toward unison, >1 stretches
  tilt: 0, // brightness: per-partial gain slope
  sustain: 1,
  attack: 8, // ms
  release: 0.25, // s
  reverb: 0.25,
  width: 0.6,
  drumPitch: 1,
  drumDecay: 1,
  drumNoise: 1,
};

function rng(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function impulse(ctx, seconds = 2.6, decay = 2.8) {
  const n = Math.floor(seconds * ctx.sampleRate);
  const buf = ctx.createBuffer(2, n, ctx.sampleRate);
  const r = rng(7);
  for (let c = 0; c < 2; c++) {
    const x = buf.getChannelData(c);
    for (let i = 0; i < n; i++) x[i] = (r() * 2 - 1) * Math.pow(1 - i / n, decay);
  }
  return buf;
}

function buildChain(ctx, s) {
  const input = ctx.createGain();
  const dry = ctx.createGain();
  const wet = ctx.createGain();
  const conv = ctx.createConvolver();
  const comp = ctx.createDynamicsCompressor();
  const master = ctx.createGain();
  dry.gain.value = 1 - 0.5 * s.reverb;
  wet.gain.value = s.reverb * 0.6;
  conv.buffer = impulse(ctx);
  comp.threshold.value = -16;
  comp.ratio.value = 4;
  master.gain.value = s.volume;
  input.connect(dry).connect(comp);
  input.connect(conv).connect(wet).connect(comp);
  comp.connect(master).connect(ctx.destination);
  return { input, master };
}

function panners(ctx, dest, n, width) {
  return Array.from({ length: Math.max(n, 1) }, (_, v) => {
    const p = ctx.createStereoPanner();
    p.pan.value = n <= 1 ? 0 : ((v / (n - 1)) * 2 - 1) * width;
    p.connect(dest);
    return p;
  });
}

function scheduleNotes(ctx, dest, tl, timbre, s, t0) {
  const ratios = timbre.ratios.map((r) => Math.pow(r, s.warp));
  const amps = timbre.amps.map((a, k) => a * Math.exp(s.tilt * k));
  const mul = Math.pow(2, s.transpose / 1200);
  const nyq = ctx.sampleRate * 0.45;
  const att = Math.max(s.attack, 1) / 1000;
  const outs = panners(ctx, dest, tl.nVoices, s.width);
  for (const n of tl.notes) {
    const start = t0 + n.t;
    const end = Math.max(start + n.dur, start + att + 0.001);
    const out = outs[n.voice % outs.length];
    ratios.forEach((r, k) => {
      const f = n.f * mul * r;
      if (f > nyq) return;
      const osc = ctx.createOscillator();
      const g = ctx.createGain();
      osc.frequency.value = f;
      g.gain.setValueAtTime(0, start);
      g.gain.linearRampToValueAtTime(n.amp * amps[k] * 0.35, start + att);
      g.gain.setTargetAtTime(0, start + att, s.sustain / (timbre.decay * (1 + 0.4 * k)));
      g.gain.setTargetAtTime(0, end, s.release / 3);
      osc.connect(g).connect(out);
      osc.start(start);
      osc.stop(end + s.release * 5);
    });
  }
}

function drumBuffer(ctx, d, s, seed) {
  const sr = ctx.sampleRate;
  const n = Math.floor(0.8 * sr * Math.max(1, s.drumDecay));
  const buf = ctx.createBuffer(1, n, sr);
  const x = buf.getChannelData(0);
  const r = rng(seed);
  const f0 = d.f0 * s.drumPitch;
  for (let i = 0; i < n; i++) {
    const t = i / sr;
    let v = 0;
    for (let k = 0; k < d.ratios.length; k++) {
      const f = f0 * d.ratios[k];
      if (f < sr * 0.45) v += d.amps[k] * Math.sin(2 * Math.PI * f * t) * Math.exp((-t * d.decay * Math.sqrt(d.ratios[k])) / s.drumDecay);
    }
    v += d.noise * s.drumNoise * (r() * 2 - 1) * Math.exp((-t * d.noise_decay) / s.drumDecay);
    x[i] = v * Math.min(t / 0.002, 1) * 0.5;
  }
  return buf;
}

function scheduleHits(ctx, dest, tl, drums, s, t0) {
  const bufs = drums.map((d, i) => drumBuffer(ctx, d, s, i + 1));
  const outs = panners(ctx, dest, drums.length, s.width);
  for (const h of tl.hits) {
    const src = ctx.createBufferSource();
    const g = ctx.createGain();
    src.buffer = bufs[h.voice];
    g.gain.value = h.amp;
    src.connect(g).connect(outs[h.voice]);
    src.start(t0 + h.t);
  }
}

// items: [{kind, tokens|calls, world, transpose?}] all start together. Returns the longest length.
function schedule(ctx, dest, items, s, t0) {
  let len = 0;
  for (const item of items) {
    const sound = { ...s, transpose: s.transpose + (item.transpose || 0) };
    const tl = timeline(item, s.tempo);
    if (tl.notes.length) scheduleNotes(ctx, dest, tl, item.world.timbre, sound, t0);
    if (tl.hits.length) scheduleHits(ctx, dest, tl, item.world.drums, sound, t0);
    len = Math.max(len, tl.length);
  }
  return len;
}

export class Engine {
  constructor() {
    this.ctx = null;
    this.session = null;
  }

  ensure() {
    if (!this.ctx) this.ctx = new AudioContext();
    if (this.ctx.state === "suspended") this.ctx.resume();
    return this.ctx;
  }

  play(items, s, { loop = false, onEnd } = {}) {
    this.stop();
    const ctx = this.ensure();
    const chain = buildChain(ctx, s);
    const session = { chain, t0: ctx.currentTime + 0.08, len: 0, timer: null, onEnd };
    session.len = schedule(ctx, chain.input, items, s, session.t0);
    const tick = () => {
      if (this.session !== session) return;
      if (loop) {
        session.t0 += session.len;
        schedule(ctx, chain.input, items, s, session.t0);
        session.timer = setTimeout(tick, (session.t0 + session.len - ctx.currentTime - 0.3) * 1000);
      } else {
        this.session = null;
        setTimeout(() => chain.master.disconnect(), s.release * 6000 + 3000);
        onEnd?.();
      }
    };
    const wait = loop ? session.len - 0.3 : session.len + 0.1;
    session.timer = setTimeout(tick, Math.max(wait, 0.05) * 1000 + 80);
    this.session = session;
    return session;
  }

  stop() {
    const p = this.session;
    if (!p) return;
    clearTimeout(p.timer);
    const g = p.chain.master.gain;
    const now = this.ctx.currentTime;
    g.cancelScheduledValues(now);
    g.setTargetAtTime(0, now, 0.03);
    setTimeout(() => p.chain.master.disconnect(), 400);
    this.session = null;
    p.onEnd?.();
  }

  // Playback position in [0, 1) of the current pass, or null.
  progress() {
    const p = this.session;
    if (!p || !p.len) return null;
    return Math.min(Math.max((this.ctx.currentTime - p.t0) / p.len, 0), 1);
  }

  async render(items, s) {
    const sr = 44100;
    const len = Math.max(...items.map((it) => timeline(it, s.tempo).length)) + s.release * 5 + 2.8;
    const ctx = new OfflineAudioContext(2, Math.ceil(len * sr), sr);
    const chain = buildChain(ctx, s);
    schedule(ctx, chain.input, items, s, 0.05);
    return ctx.startRendering();
  }
}

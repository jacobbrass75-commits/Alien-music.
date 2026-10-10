import "./style.css";
import { Engine, DEFAULT_SOUND } from "./engine.js";
import { timeline, meaningLabel } from "./score.js";
import { drawTimeline, drawPlayhead, drawScale, drawPartials } from "./viz.js";
import { encodeWav, download } from "./wav.js";

const $ = (s, el = document) => el.querySelector(s);
const engine = new Engine();
const state = {
  models: [],
  modelId: store("xeno.model"),
  result: null,
  sound: { ...DEFAULT_SOUND, ...(safeJson(store("xeno.sound")) || {}) },
  gen: { n: 6, length: "", temperature: 1, seed: "", ...(safeJson(store("xeno.gen")) || {}) },
  mix: [],
  sentence: [],
  playing: null,
};

// ---------------------------------------------------------------- helpers

function store(key, value) {
  try {
    if (value === undefined) return localStorage.getItem(key);
    localStorage.setItem(key, value);
  } catch {
    return null;
  }
}

function safeJson(s) {
  try {
    return JSON.parse(s);
  } catch {
    return null;
  }
}

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "class") el.className = v;
    else if (v !== false && v != null) el.setAttribute(k, v === true ? "" : v);
  }
  el.append(...children.flat().filter((c) => c != null));
  return el;
}

function toast(msg, err = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "show" + (err ? " err" : "");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (t.className = ""), 4000);
}

async function api(path, body) {
  let r;
  try {
    r = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  } catch {
    throw new Error("Backend not reachable. Start it with: python server.py");
  }
  const j = await r.json().catch(() => ({ error: `HTTP ${r.status}. Is python server.py running?` }));
  if (!r.ok || j.error) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}

const fmt = (x, d = 2) => (typeof x === "number" ? x.toFixed(d) : x);

// ---------------------------------------------------------------- playback

function play(items, { canvas = null, button = null, loop = false } = {}) {
  if (button && state.playing?.button === button) return engine.stop();
  const session = { items, canvas, button, loop };
  engine.play(items, state.sound, {
    loop,
    onEnd: () => {
      if (button) button.classList.remove("on"), (button.textContent = button.dataset.idle || "▶");
      if (canvas) drawPlayhead(canvas, null);
      if (state.playing === session) state.playing = null;
    },
  });
  state.playing = session;
  if (button) button.classList.add("on"), (button.textContent = "■");
}

function restartPlaying() {
  const p = state.playing;
  if (!p) return;
  state.playing = null;
  play(p.items, { canvas: p.canvas, button: p.button, loop: p.loop });
}

(function frame() {
  const p = state.playing;
  if (p?.canvas) drawPlayhead(p.canvas, engine.progress());
  requestAnimationFrame(frame);
})();

async function exportWav(items, name, button) {
  const label = button.textContent;
  button.textContent = "…";
  button.disabled = true;
  try {
    download(encodeWav(await engine.render(items, state.sound)), name);
  } catch (e) {
    toast(`Export failed: ${e.message}`, true);
  } finally {
    button.textContent = label;
    button.disabled = false;
  }
}

// ---------------------------------------------------------------- models

async function loadModels(quiet = false) {
  try {
    state.models = await api("/api/models");
  } catch (e) {
    if (!quiet) toast(e.message, true);
    state.models = [];
  }
  const sel = $("#model");
  const keep = state.modelId;
  sel.replaceChildren();
  if (!state.models.length) sel.append(h("option", { value: "" }, "no trained models yet"));
  for (const env of ["composer", "rhythm", "signal"]) {
    const ms = state.models.filter((m) => m.env === env);
    if (!ms.length) continue;
    sel.append(h("optgroup", { label: env }, ms.map((m) => h("option", { value: m.id, title: m.run }, `${m.run.split("/").slice(-2).join("/")} · iter ${m.iter}`))));
  }
  if (state.models.some((m) => m.id === keep)) sel.value = keep;
  state.modelId = sel.value || null;
  renderGen();
  if (!state.result) renderStage();
}

const currentModel = () => state.models.find((m) => m.id === state.modelId);

async function generate() {
  const m = currentModel();
  if (!m) return toast("No model selected. Training may still be running.", true);
  const btn = $("#generate");
  btn.disabled = true;
  btn.textContent = "Generating…";
  try {
    const g = state.gen;
    const body = { model: m.id, n: +g.n, temperature: +g.temperature };
    if (g.length) body.length = +g.length;
    if (g.seed !== "") body.seed = +g.seed;
    engine.stop();
    state.result = { ...(await api("/api/generate", body)), model: m };
    state.sentence = [];
    renderStage();
    renderWorld();
  } catch (e) {
    toast(e.message, true);
  } finally {
    btn.disabled = false;
    btn.textContent = "Generate";
  }
}

// ---------------------------------------------------------------- side panels

function slider({ label, min, max, step, value, show, onInput }) {
  const out = h("output", {}, show(value));
  const input = h("input", { type: "range", min, max, step, value });
  input.addEventListener("input", () => {
    out.textContent = show(+input.value);
    onInput(+input.value);
  });
  return h("label", { class: "row" }, h("span", {}, label), input, out);
}

function renderGen() {
  const m = currentModel();
  const g = state.gen;
  const save = () => store("xeno.gen", JSON.stringify(g));
  const isSignal = m?.env === "signal";
  const lengthSel = h("select", { onchange: (e) => ((g.length = e.target.value), save()) },
    ["", "16", "32", "64", "128", "256"].map((v) => h("option", { value: v, selected: g.length === v }, v ? `${v} events` : "as trained")));
  const seedIn = h("input", { type: "text", inputmode: "numeric", placeholder: "random", value: g.seed, oninput: (e) => ((g.seed = e.target.value.replace(/\D/g, "")), save()) });
  $("#gen").replaceChildren(
    isSignal
      ? h("p", { class: "muted small" }, "Signal models generate their whole lexicon: one call per meaning.")
      : h("div", {},
          slider({ label: "Pieces", min: 1, max: 12, step: 1, value: g.n, show: (v) => v, onInput: (v) => ((g.n = v), save()) }),
          h("label", { class: "row" }, h("span", {}, "Length"), lengthSel)),
    slider({ label: "Temperature", min: 0.2, max: 1.8, step: 0.05, value: g.temperature, show: (v) => fmt(v), onInput: (v) => ((g.temperature = v), save()) }),
    h("label", { class: "row" }, h("span", {}, "Seed"), seedIn,
      h("button", { class: "icon", title: "Random seed", onclick: () => { g.seed = String(Math.floor(Math.random() * 1e6)); seedIn.value = g.seed; save(); } }, "⚄")),
  );
}

const LAB = [
  ["Time", [["tempo", "Tempo", 0.25, 3, 0.01, (v) => `${fmt(v)}×`]]],
  ["Tone", [
    ["transpose", "Transpose", -2400, 2400, 1, (v) => `${v > 0 ? "+" : ""}${v}¢`],
    ["warp", "Partial stretch", 0.6, 1.6, 0.01, (v) => fmt(v)],
    ["tilt", "Brightness", -0.8, 0.6, 0.01, (v) => fmt(v)],
    ["sustain", "Sustain", 0.2, 4, 0.01, (v) => `${fmt(v)}×`],
    ["attack", "Attack", 1, 400, 1, (v) => `${v}ms`],
    ["release", "Release", 0.02, 3, 0.01, (v) => `${fmt(v)}s`],
  ]],
  ["Drums", [
    ["drumPitch", "Tune", 0.25, 4, 0.01, (v) => `${fmt(v)}×`],
    ["drumDecay", "Ring", 0.3, 4, 0.01, (v) => `${fmt(v)}×`],
    ["drumNoise", "Noise", 0, 3, 0.01, (v) => `${fmt(v)}×`],
  ]],
  ["Space", [
    ["reverb", "Reverb", 0, 1, 0.01, (v) => `${Math.round(v * 100)}%`],
    ["width", "Width", 0, 1, 0.01, (v) => `${Math.round(v * 100)}%`],
    ["volume", "Volume", 0, 1, 0.01, (v) => `${Math.round(v * 100)}%`],
  ]],
];

let restartTimer;
function renderLab() {
  $("#lab").replaceChildren(...LAB.map(([group, rows]) => h("div", { class: "group" }, h("h3", {}, group),
    rows.map(([key, label, min, max, step, show]) => slider({ label, min, max, step, value: state.sound[key], show, onInput: (v) => {
      state.sound[key] = v;
      store("xeno.sound", JSON.stringify(state.sound));
      if (key === "warp") renderWorld();
      clearTimeout(restartTimer);
      restartTimer = setTimeout(restartPlaying, 250); // hear the change on whatever is playing
    } })))));
}

function mutateLab() {
  for (const [, rows] of LAB)
    for (const [key, , min, max, step] of rows) {
      if (key === "volume") continue;
      const v = state.sound[key] + (Math.random() - 0.5) * (max - min) * 0.3;
      state.sound[key] = Math.min(max, Math.max(min, Math.round(v / step) * step));
    }
  store("xeno.sound", JSON.stringify(state.sound));
  renderLab();
  renderWorld();
  restartPlaying();
}

function renderWorld() {
  const r = state.result;
  if (!r) return;
  const w = r.world;
  const el = $("#world");
  el.className = "";
  const blocks = [];
  if (w.scale_cents) {
    const scale = h("canvas", { class: "viz" });
    const partials = h("canvas", { class: "viz short" });
    blocks.push(h("h3", {}, "Scale (cents) vs. the human 12-note grid"), scale, h("h3", {}, "Timbre partials vs. harmonic series"), partials);
    requestAnimationFrame(() => {
      drawScale(scale, w);
      drawPartials(partials, w.timbre.ratios, w.timbre.amps, state.sound.warp);
    });
  }
  if (w.drums) {
    blocks.push(h("h3", {}, "Drums: modes vs. harmonic series"));
    w.drums.forEach((d, i) => {
      const c = h("canvas", { class: "viz tiny" });
      blocks.push(h("div", { class: "drum" }, h("span", {}, `v${i} ${Math.round(d.f0 * state.sound.drumPitch)}Hz`), c));
      requestAnimationFrame(() => drawPartials(c, d.ratios, d.amps, 1, i));
    });
  }
  blocks.push(h("details", {}, h("summary", {}, "Raw description"), h("pre", {}, w.describe)));
  el.replaceChildren(...blocks);
}

async function pollTraining() {
  let runs = [];
  try {
    runs = await api("/api/status");
  } catch {
    $("#training").replaceChildren(h("span", { class: "muted" }, "Backend offline."));
    return;
  }
  const key = { composer: "tension", rhythm: "density", signal: "listener_acc" };
  const recent = runs.filter((r) => Date.now() / 1000 - r.updated < 600 && r.iter < r.iters);
  const shown = [...recent, ...runs.filter((r) => !recent.includes(r))].slice(0, 6);
  $("#training").replaceChildren(
    ...(shown.length ? shown : [null]).map((r) => r
      ? h("div", { class: "run" + (recent.includes(r) ? " live" : "") },
          h("div", { class: "run-top" }, h("span", {}, `${r.env} · ${r.run.split("/").pop()}`), h("span", { class: "mono" }, `${r.iter}/${r.iters}`)),
          h("div", { class: "bar" }, h("i", { style: `width:${(100 * r.iter) / r.iters}%` })),
          h("div", { class: "mono small muted" }, `entropy ${fmt(r.last.entropy)} · ${key[r.env]} ${fmt(r.last[key[r.env]])}${r.last.surprise != null ? ` · surprise ${fmt(r.last.surprise)}` : ""}`))
      : h("span", { class: "muted" }, "No runs yet. The first experiment starts when the PC is idle.")),
  );
  // Checkpoints save every 50 iterations while training; pick new ones up quietly.
  const sig = runs.map((r) => `${r.run}:${Math.floor(r.iter / 50)}`).join();
  if (sig !== pollTraining.last) loadModels(true);
  pollTraining.last = sig;
}

// ---------------------------------------------------------------- stage

function renderStage() {
  const r = state.result;
  const head = $("#stage-head");
  const box = $("#pieces");
  if (!r) {
    head.replaceChildren();
    box.className = "empty";
    box.replaceChildren(h("div", { class: "hero" },
      h("div", { class: "glyph big" }, "◈"),
      h("h1", {}, state.models.length ? "Pick a model, hit Generate." : "No trained models yet."),
      h("p", { class: "muted" }, state.models.length
        ? "Everything you hear is synthesised here from the model's events, so the Sound lab reshapes it live."
        : "The first experiment trains a rhythm, a signal-game and a composer model. They show up here as checkpoints save.")));
    return;
  }
  const m = r.model;
  const metrics = Object.entries(r.metrics).map(([k, v]) => `${k} ${fmt(v)}`).join(" · ");
  head.replaceChildren(h("div", { class: "stage-title" },
    h("span", { class: `tag ${r.env}` }, r.env),
    h("span", {}, `${m.run} · iter ${m.iter}`),
    h("button", { class: "ghost mono", title: "Reuse this seed", onclick: () => { state.gen.seed = String(r.seed); store("xeno.gen", JSON.stringify(state.gen)); renderGen(); toast(`Seed ${r.seed} locked in.`); } }, `seed ${r.seed}`),
    h("span", { class: "mono small muted" }, metrics)));
  box.className = r.env === "signal" ? "lexicon-wrap" : "grid";
  if (r.env === "signal") return renderLexicon(box, r);
  box.replaceChildren(...r.pieces.map((tokens, i) => pieceCard({ kind: r.env, tokens, world: r.world, label: `${r.env} #${i + 1}` }, i, r)));
}

function pieceCard(item, i, r) {
  const tl = timeline(item, 1);
  const canvas = h("canvas", { class: "roll" });
  const playBtn = h("button", { class: "play", "data-idle": "▶" }, "▶");
  playBtn.addEventListener("click", () => play([item], { canvas, button: playBtn }));
  const card = h("article", { class: "card" },
    h("div", { class: "card-head" }, h("span", { class: "idx" }, `#${i + 1}`),
      h("span", { class: "mono small muted" }, `${item.tokens.length} events · ${fmt(tl.length / state.sound.tempo, 1)}s`)),
    canvas,
    h("div", { class: "card-actions" }, playBtn,
      h("button", { onclick: () => addToMix(item) }, "+ mix"),
      h("button", { onclick: (e) => exportWav([item], `xeno-${r.env}-s${r.seed}-${i + 1}.wav`, e.currentTarget) }, "⬇ wav")));
  canvas.addEventListener("click", () => play([item], { canvas, button: playBtn }));
  requestAnimationFrame(() => drawTimeline(canvas, tl));
  return card;
}

function renderLexicon(box, r) {
  const w = r.world;
  const callItem = (calls) => ({ kind: "calls", calls, world: w });
  const sentenceBox = h("div", { class: "sentence" });
  const renderSentence = () => sentenceBox.replaceChildren(
    state.sentence.length
      ? state.sentence.map((p, i) => h("button", { class: "chip", title: "Remove", onclick: () => { state.sentence.splice(i, 1); renderSentence(); } }, meaningLabel(p.meaning)))
      : h("span", { class: "muted small" }, "Shift-click calls to build a sentence."));
  const babble = () => {
    const second = -(w.scale_cents[2] ?? 500); // the other speaker sits an alien interval lower
    const calls = Array.from({ length: 10 }, (_, i) => {
      const p = r.pieces[Math.floor(Math.random() * r.pieces.length)];
      return { frames: p.frames, speaker: i % 2, transpose: i % 2 ? second : 0 };
    });
    play([callItem(calls)]);
  };
  const sentenceItem = () => callItem(state.sentence.map((p) => ({ frames: p.frames, speaker: 0 })));
  const toolbar = h("div", { class: "toolbar" },
    h("button", { onclick: babble }, "☍ babble"),
    h("button", { onclick: () => play([callItem(r.pieces.map((p) => ({ frames: p.frames, speaker: 0 })))]) }, "▶ whole lexicon"),
    h("span", { class: "sep" }),
    sentenceBox,
    h("button", { onclick: () => state.sentence.length && play([sentenceItem()]) }, "▶ sentence"),
    h("button", { onclick: () => state.sentence.length && addToMix({ ...sentenceItem(), label: `sentence ${state.sentence.map((p) => meaningLabel(p.meaning)).join(" ")}` }) }, "+ mix"),
    h("button", { class: "ghost", onclick: () => ((state.sentence = []), renderSentence()) }, "clear"));
  renderSentence();

  const nv = w.n_values;
  const grid = h("div", { class: "lexicon", style: w.n_attrs === 2 ? `grid-template-columns: 2.2rem repeat(${nv}, 1fr)` : "" });
  if (w.n_attrs === 2) grid.append(h("span"), ...Array.from({ length: nv }, (_, j) => h("span", { class: "axis" }, `B${j + 1}`)));
  r.pieces.forEach((p, i) => {
    if (w.n_attrs === 2 && i % nv === 0) grid.append(h("span", { class: "axis" }, `A${i / nv + 1}`));
    const canvas = h("canvas", { class: "roll small" });
    const item = callItem([{ frames: p.frames, speaker: 0 }]);
    const silent = p.frames.every(([, l]) => !l);
    const cell = h("button", { class: "cell" + (silent ? " silent" : ""), title: "Click: play · Shift-click: add to sentence" },
      h("span", { class: "mono small" }, meaningLabel(p.meaning)), canvas);
    cell.addEventListener("click", (e) => {
      if (e.shiftKey) {
        state.sentence.push(p);
        return renderSentence();
      }
      play([item], { canvas });
    });
    grid.append(cell);
    requestAnimationFrame(() => drawTimeline(canvas, timeline(item, 1)));
  });
  box.replaceChildren(toolbar, grid,
    h("p", { class: "muted small" }, `Listener decodes ${Math.round(100 * (r.metrics.listener_acc ?? 0))}% of these. Dashed or empty cells are calls that are (almost) silent: the energy cost at work.`));
}

// ---------------------------------------------------------------- mix

function addToMix(item) {
  state.mix.push({ ...item, label: item.label || item.kind });
  renderMix();
  toast(`Added ${item.label || item.kind} to the mix.`);
}

function renderMix() {
  $("#mix-items").replaceChildren(...(state.mix.length
    ? state.mix.map((it, i) => h("button", { class: `chip ${it.kind}`, title: "Remove", onclick: () => { state.mix.splice(i, 1); renderMix(); } }, `${it.label} ×`))
    : [h("span", { class: "muted" }, "Add pieces with “+ mix” to layer them.")]));
}

// ---------------------------------------------------------------- wiring

$("#model").addEventListener("change", (e) => {
  state.modelId = e.target.value;
  store("xeno.model", state.modelId);
  renderGen();
});
$("#refresh").addEventListener("click", () => loadModels());
$("#generate").addEventListener("click", generate);
$("#lab-reset").addEventListener("click", () => {
  state.sound = { ...DEFAULT_SOUND };
  store("xeno.sound", JSON.stringify(state.sound));
  renderLab();
  renderWorld();
  restartPlaying();
});
$("#lab-mutate").addEventListener("click", mutateLab);
const mixBtn = $("#mix-play");
mixBtn.dataset.idle = "▶ play";
mixBtn.addEventListener("click", () => state.mix.length && play(state.mix, { button: mixBtn, loop: $("#mix-loop").checked }));
$("#mix-wav").addEventListener("click", (e) => state.mix.length && exportWav(state.mix, "xeno-mix.wav", e.currentTarget));
$("#mix-clear").addEventListener("click", () => ((state.mix = []), renderMix()));
document.addEventListener("keydown", (e) => {
  if (e.target.matches("input, select, textarea")) return;
  if (e.code === "Escape" || (e.code === "Space" && state.playing)) {
    e.preventDefault();
    engine.stop();
  } else if (e.code === "Enter") generate();
});
window.addEventListener("resize", () => {
  clearTimeout(window.__rz);
  window.__rz = setTimeout(() => (renderStage(), renderWorld()), 200);
});

renderLab();
renderMix();
loadModels();
pollTraining();
setInterval(pollTraining, 10000);

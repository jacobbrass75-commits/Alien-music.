// Canvas drawing: piece timelines (with a live playhead), the alien scale, and timbre spectra.
export const VOICE_COLORS = ["#5ef1c2", "#b38cff", "#ff8fb1", "#ffd36e", "#6ec9ff", "#c2ff6e"];
const color = (v) => VOICE_COLORS[v % VOICE_COLORS.length];

function sized(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 300;
  const h = canvas.clientHeight || 80;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { g, w, h };
}

function pill(g, x, y, w, h) {
  const r = Math.min(h / 2, w / 2);
  g.beginPath();
  g.roundRect(x, y, w, h, r);
  g.fill();
}

export function drawTimeline(canvas, tl) {
  const { w, h } = sized(canvas);
  const dpr = window.devicePixelRatio || 1;
  const base = document.createElement("canvas");
  base.width = canvas.width;
  base.height = canvas.height;
  const g = base.getContext("2d");
  g.scale(dpr, dpr);
  const len = tl.length || 1;
  const pad = 6;
  if (tl.notes.length) {
    const lf = tl.notes.map((n) => Math.log2(n.f));
    let lo = Math.min(...lf), hi = Math.max(...lf);
    if (hi - lo < 0.6) (lo -= 0.3), (hi += 0.3);
    for (const n of tl.notes) {
      const x = (n.t / len) * w;
      const y = h - pad - ((Math.log2(n.f) - lo) / (hi - lo)) * (h - 2 * pad);
      g.fillStyle = color(n.voice);
      g.globalAlpha = 0.45 + 0.55 * Math.min(1, n.amp / 0.25);
      pill(g, x, y - 2.5, Math.max(3, (n.dur / len) * w - 1), 5);
    }
  }
  if (tl.hits.length) {
    const rh = (h - 2 * pad) / tl.nVoices;
    g.globalAlpha = 0.9;
    for (const hit of tl.hits) {
      g.fillStyle = color(hit.voice);
      g.fillRect((hit.t / len) * w, pad + hit.voice * rh + rh * 0.18, 2.2, rh * 0.64);
    }
  }
  canvas.__base = base;
  drawPlayhead(canvas, null);
}

export function drawPlayhead(canvas, frac) {
  const g = canvas.getContext("2d");
  g.save();
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.clearRect(0, 0, canvas.width, canvas.height);
  if (canvas.__base) g.drawImage(canvas.__base, 0, 0);
  if (frac != null) {
    const x = frac * canvas.width;
    const grad = g.createLinearGradient(x - 40, 0, x, 0);
    grad.addColorStop(0, "rgba(94,241,194,0)");
    grad.addColorStop(1, "rgba(94,241,194,0.18)");
    g.fillStyle = grad;
    g.fillRect(x - 40, 0, 40, canvas.height);
    g.fillStyle = "rgba(255,255,255,0.9)";
    g.fillRect(x, 0, Math.max(1, window.devicePixelRatio || 1), canvas.height);
  }
  g.restore();
}

// The model's scale across one period, against the human 12-note grid for contrast.
export function drawScale(canvas, world) {
  const { g, w, h } = sized(canvas);
  const P = world.period_cents;
  const x = (c) => 10 + (c / P) * (w - 20);
  const mid = h * 0.58;
  g.strokeStyle = "rgba(255,255,255,0.12)";
  g.fillStyle = "rgba(255,255,255,0.28)";
  g.font = "9px 'JetBrains Mono', monospace";
  g.textAlign = "center";
  for (let c = 0; c <= P; c += 100) {
    const big = c % 1200 === 0;
    g.beginPath();
    g.moveTo(x(c), mid - (big ? 12 : 5));
    g.lineTo(x(c), mid + (big ? 12 : 5));
    g.stroke();
    if (big && c) g.fillText("2:1", x(c), mid + 24);
  }
  g.strokeStyle = "rgba(255,255,255,0.35)";
  g.beginPath();
  g.moveTo(x(0), mid);
  g.lineTo(x(P), mid);
  g.stroke();
  world.scale_cents.forEach((c, i) => {
    g.fillStyle = color(1);
    g.beginPath();
    g.arc(x(c), mid, 4.5, 0, Math.PI * 2);
    g.fill();
    g.fillStyle = "rgba(230,236,255,0.85)";
    g.fillText(Math.round(c), x(c), i % 2 ? mid + 20 : mid - 12);
  });
  g.fillStyle = color(0);
  g.beginPath();
  g.arc(x(P), mid, 5.5, 0, Math.PI * 2);
  g.fill();
  g.fillStyle = color(0);
  g.textAlign = "left";
  g.fillText(`period ${Math.round(P)}¢ (its "octave")`, 4, 9);
}

// Partials (bars) against the harmonic series (faint lines) on a log-frequency axis.
export function drawPartials(canvas, ratios, amps, warp = 1, tint = 0) {
  const { g, w, h } = sized(canvas);
  const rs = ratios.map((r) => Math.pow(r, warp));
  const top = Math.max(...rs, ratios.length) * 1.08;
  const x = (r) => 8 + (Math.log(r) / Math.log(top)) * (w - 16);
  g.strokeStyle = "rgba(255,255,255,0.13)";
  g.setLineDash([2, 3]);
  for (let k = 1; k <= top; k++) {
    g.beginPath();
    g.moveTo(x(k), 4);
    g.lineTo(x(k), h - 4);
    g.stroke();
  }
  g.setLineDash([]);
  const amax = Math.max(...amps);
  rs.forEach((r, k) => {
    const bh = (amps[k] / amax) * (h - 10);
    g.fillStyle = color(tint);
    g.globalAlpha = 0.85;
    g.fillRect(x(r) - 1.5, h - 4 - bh, 3, bh);
  });
  g.globalAlpha = 1;
}

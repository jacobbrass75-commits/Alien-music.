// Turns model output (tokens) + the model's world into timed notes and hits.
// Mirrors the Python renderers in alien/envs so the studio sounds like the training samples.

export function composerTimeline(tokens, world, tempo = 1) {
  const nv = world.n_voices;
  const notes = [];
  const held = new Array(nv).fill(-1);
  let t = 0;
  for (const tok of tokens) {
    const dur = (world.base_dur * world.dur_ratios[tok[nv]]) / tempo;
    for (let v = 0; v < nv; v++) {
      const p = tok[v];
      if (p && held[v] >= 0 && notes[held[v]].p === p) notes[held[v]].dur += dur; // same pitch again: sustain
      else if (p) {
        notes.push({ t, dur, f: world.pitches[p - 1], p, amp: 0.25, voice: v });
        held[v] = notes.length - 1;
      } else held[v] = -1;
    }
    t += dur;
  }
  return { notes, hits: [], length: t, nVoices: nv };
}

export function rhythmTimeline(tokens, world, tempo = 1) {
  const hits = [];
  let t = 0;
  for (const [pattern, ioi] of tokens) {
    for (let k = 0; k < world.n_voices; k++) if ((pattern >> k) & 1) hits.push({ t, voice: k, amp: 0.6 });
    t += (world.base_ioi * world.ioi_ratios[ioi]) / tempo;
  }
  return { notes: [], hits, length: t, nVoices: world.n_voices };
}

// calls: [{frames, speaker, transpose?}] played one after another (a "sentence").
export function callsTimeline(calls, world, tempo = 1, gap = 0.22) {
  const fs = world.frame_s / tempo;
  const notes = [];
  let start = 0;
  for (const call of calls) {
    const mul = Math.pow(2, (call.transpose || 0) / 1200);
    let prev = null;
    call.frames.forEach(([p, l], i) => {
      if (!l) return void (prev = null);
      const t = start + i * fs;
      const amp = (0.3 * l) / (world.n_loud - 1);
      if (prev && prev.p === p && Math.abs(prev.t + prev.dur - t) < 1e-6) prev.dur += fs;
      else {
        prev = { t, dur: fs, f: world.pitches[p] * mul, p, amp, voice: call.speaker || 0 };
        notes.push(prev);
      }
    });
    start += call.frames.length * fs + gap / tempo;
  }
  return { notes, hits: [], length: Math.max(start - gap / tempo, 0.01), nVoices: 2 };
}

export function timeline(item, tempo) {
  if (item.kind === "composer") return composerTimeline(item.tokens, item.world, tempo);
  if (item.kind === "rhythm") return rhythmTimeline(item.tokens, item.world, tempo);
  return callsTimeline(item.calls, item.world, tempo);
}

export function meaningLabel(m) {
  return m.map((v, i) => String.fromCharCode(65 + i) + (v + 1)).join("·");
}

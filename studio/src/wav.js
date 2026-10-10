// AudioBuffer -> 16-bit stereo WAV Blob, peak-normalised to -1 dBFS.
export function encodeWav(buffer) {
  const chans = [...Array(buffer.numberOfChannels)].map((_, c) => buffer.getChannelData(c));
  let peak = 1e-9;
  for (const x of chans) for (let i = 0; i < x.length; i++) peak = Math.max(peak, Math.abs(x[i]));
  const gain = 0.89 / peak;
  const n = buffer.length, nc = chans.length, sr = buffer.sampleRate;
  const view = new DataView(new ArrayBuffer(44 + n * nc * 2));
  const str = (o, s) => [...s].forEach((ch, i) => view.setUint8(o + i, ch.charCodeAt(0)));
  str(0, "RIFF");
  view.setUint32(4, 36 + n * nc * 2, true);
  str(8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, nc, true);
  view.setUint32(24, sr, true);
  view.setUint32(28, sr * nc * 2, true);
  view.setUint16(32, nc * 2, true);
  view.setUint16(34, 16, true);
  str(36, "data");
  view.setUint32(40, n * nc * 2, true);
  let o = 44;
  for (let i = 0; i < n; i++)
    for (let c = 0; c < nc; c++, o += 2) view.setInt16(o, Math.max(-1, Math.min(1, chans[c][i] * gain)) * 32767, true);
  return new Blob([view], { type: "audio/wav" });
}

export function download(blob, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

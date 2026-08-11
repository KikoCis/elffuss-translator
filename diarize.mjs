// diarize.mjs — diarización por turnos 100% en el navegador (ONNX).
// pyannote-segmentation-3.0 (fronteras habla/silencio) → wespeaker resnet34 (embeddings) →
// clustering greedy por coseno con filtrado de singletons. Devuelve turnos etiquetados por hablante.
//
// LÍMITE HONESTO: separar voces SIMULTÁNEAS en un único canal mono NO es fiable (el hablante más
// fuerte domina el embedding). Eso se resuelve en la UX con una captura por interlocutor. Aquí
// resolvemos el 95% real: quién habló en cada turno.
//
//   const D = await loadDiarizer(TF, {device:'webgpu'});
//   const turns = await diarize(D, audioFloat32_16k, {joinTau:0.55, maxSpeakers:2});
//   // → [{start,end,speaker:'S1'|'S2', sim}]

export async function loadDiarizer(TF, { device = 'wasm' } = {}) {
  TF.env.allowLocalModels = false;
  const seg  = await TF.AutoModelForAudioFrameClassification.from_pretrained('onnx-community/pyannote-segmentation-3.0', { device });
  const segP = await TF.AutoProcessor.from_pretrained('onnx-community/pyannote-segmentation-3.0');
  const emb  = await TF.AutoModel.from_pretrained('onnx-community/wespeaker-voxceleb-resnet34-LM', { device });
  const embP = await TF.AutoProcessor.from_pretrained('onnx-community/wespeaker-voxceleb-resnet34-LM');
  return { seg, segP, emb, embP };
}

const cos = (a, b) => { let dt = 0, na = 0, nb = 0; for (let i = 0; i < a.length; i++) { dt += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; } return dt / (Math.sqrt(na)*Math.sqrt(nb) || 1); };
const rms = (a) => { let s = 0; for (const v of a) s += v*v; return Math.sqrt(s/a.length); };

async function embed(D, slice) {
  if (slice.length < 8000) { const q = new Float32Array(8000); q.set(slice); slice = q; }
  const o = await D.emb(await D.embP(slice));
  const t = o.embeddings || o.logits || Object.values(o)[0];
  return Array.from(t.data);
}

// Embedding de hablante de un clip (Float32 16 kHz) — para diarización ONLINE (etiquetar turno a turno).
export async function speakerVec(D, audio) { return embed(D, audio); }
export function cosine(a, b) { return cos(a, b); }

// Clasificador online de hablante: acumula centroides y asigna cada intervención a S1/S2/…
export function makeSpeakerBook({ tau = 0.5, maxSpeakers = 4 } = {}) {
  const cents = []; // {v, n, label}
  return {
    assign(vec) {
      let bi = -1, bv = -2;
      for (let i = 0; i < cents.length; i++) { const s = cos(vec, cents[i].v); if (s > bv) { bv = s; bi = i; } }
      if (bi >= 0 && bv >= tau) { const c = cents[bi]; for (let k = 0; k < c.v.length; k++) c.v[k] = (c.v[k]*c.n + vec[k])/(c.n+1); c.n++; return { label: c.label, sim: +bv.toFixed(2), isNew: false }; }
      if (cents.length >= maxSpeakers) { const c = cents[bi]; return { label: c.label, sim: +bv.toFixed(2), isNew: false }; }
      const label = 'S' + (cents.length + 1); cents.push({ v: vec.slice(), n: 1, label }); return { label, sim: bi >= 0 ? +bv.toFixed(2) : 0, isNew: true };
    },
    count() { return cents.length; },
  };
}

// audio: Float32Array mono 16 kHz
export async function diarize(D, audio, { joinTau = 0.55, mergeTau = 0.5, minSpeechRms = 0.015, win = 1.0, step = 0.5, maxSpeakers = 4 } = {}) {
  // 1) regiones de habla (buenas fronteras cuando hay pausa entre turnos)
  const out = await D.seg(await D.segP(audio));
  const rp = D.segP.post_process_speaker_diarization(out.logits, audio.length)[0] || [];
  let regions = rp.filter(s => (s.end - s.start) >= 0.3).map(s => [+s.start, +s.end]).sort((a, b) => a[0]-b[0]);
  if (!regions.length) regions = [[0, audio.length/16000]];

  // 2) átomos: sub-ventanas 1.0s/0.5s con energía. Se descartan ventanas cortas (colas) que
  //    degradan el embedding y generan hablantes fantasma.
  const atoms = [];
  for (const [rs, re] of regions) {
    for (let t = rs; t < re - 0.15; t += step) {
      const s = t, e = Math.min(t + win, re);
      if (e - s < 0.6) continue;
      const sl = audio.slice(Math.floor(s*16000), Math.ceil(e*16000));
      if (rms(sl) < minSpeechRms) continue;
      atoms.push({ s, e, emb: await embed(D, sl) });
    }
  }
  if (!atoms.length) return [];

  // 3) clustering greedy por coseno (con duración acumulada por cluster)
  const C = [], N = [], DUR = [];
  for (const a of atoms) {
    let bi = -1, bv = -2;
    for (let i = 0; i < C.length; i++) { const s = cos(a.emb, C[i]); if (s > bv) { bv = s; bi = i; } }
    if (bi >= 0 && bv >= joinTau) { for (let k = 0; k < C[bi].length; k++) C[bi][k] = (C[bi][k]*N[bi] + a.emb[k])/(N[bi]+1); N[bi]++; DUR[bi] += (a.e - a.s); a.c = bi; }
    else { C.push(a.emb.slice()); N.push(1); DUR.push(a.e - a.s); a.c = C.length - 1; }
  }
  // 4) clusters reales = ≥2 ventanas Y ≥1s de habla (los singleton/colas caen)
  let real = C.map((c, i) => ({ i, c, n: N[i], dur: DUR[i] })).filter(r => r.n >= 2 && r.dur >= 1.0);
  if (!real.length) real = C.map((c, i) => ({ i, c, n: N[i], dur: DUR[i] })).sort((a, b) => b.dur - a.dur).slice(0, 1);
  // fusiona centroides parecidos
  for (let i = 0; i < real.length; i++) for (let j = i+1; j < real.length; j++) {
    if (real[j] && cos(real[i].c, real[j].c) >= mergeTau) { real[i].n += real[j].n; real[j] = null; }
  }
  real = real.filter(Boolean).sort((a, b) => b.n - a.n).slice(0, maxSpeakers);

  // 5) etiqueta cada átomo al centroide real más cercano; orden de aparición → S1,S2,…
  const order = []; const labelOf = new Map();
  const nearest = (e) => { let bi = 0, bv = -2; for (let k = 0; k < real.length; k++) { const s = cos(e, real[k].c); if (s > bv) { bv = s; bi = k; } } return { k: bi, sim: bv }; };
  for (const a of atoms) { const { k } = nearest(a.emb); if (!labelOf.has(k)) { order.push(k); labelOf.set(k, 'S' + order.length); } }
  // 6) construye turnos fusionando átomos contiguos del mismo hablante
  const turns = [];
  for (const a of atoms) { const { k, sim } = nearest(a.emb); const spk = labelOf.get(k);
    const last = turns[turns.length - 1];
    if (last && last.speaker === spk && a.s - last.end <= step + 0.3) { last.end = a.e; last.sim = Math.max(last.sim, +sim.toFixed(2)); }
    else turns.push({ start: +a.s.toFixed(2), end: +a.e.toFixed(2), speaker: spk, sim: +sim.toFixed(2) });
  }
  return turns;
}

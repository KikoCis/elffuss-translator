<div align="center">

# Elffuss Translator

**Real-time conversation translation, 100% in your browser.**

It listens, tells speakers apart, translates and speaks back.
No server, no account, no paid API — nothing you say ever leaves your machine.

[**▶ Open the app**](https://translator.elffuss.com) · [Elffuss](https://elffuss.com) · Apache-2.0

</div>

---

## Live skills: a meeting facilitator that costs nothing

The translator does more than translate. It can hand the conversation to
[Elffuss Claw](https://github.com/KikoCis/elffuss-claw), which runs a **live audio skill**
and works while you talk. Pick one when you hit **Skills en vivo**:

| | Skill | What it does |
|---|---|---|
| 🎯 | **Sales copilot** | Feeds you the exact line to say and tracks the call's objectives. |
| 📇 | **Customer card** | Fills a customer record by itself from what the client says. Exports to Excel. |
| 🧑‍🏫 | **Meeting facilitator** | Topics, decisions and action items, live. Exports. |
| 📝 | **Summary & ideas** | A running summary plus the ideas that come up. |

Each skill paints its own **surface**: a live view that renders *inside the translator*,
next to the subtitles — not just a notification. It updates as the conversation goes on
and, where it makes sense, exports what it captured as a spreadsheet.

![Demo: sales call with live translation and the copilot surface](docs/demo.gif)

<div align="center"><sub>

**Staged recreation** of the flow (scripted dialogue and timings), not a screen recording of a live run ·
[full video with audio](docs/demo.mp4)

</sub></div>

Skills are defined as plain `SKILL.md` files and are **self-editing**: ask in Claw's chat for
"add an email field" or "keep the advice shorter" and the skill rewrites itself.

## How the surface gets there

The two apps live on different origins, so the pipe is `postMessage` with origin checks on
both ends. The skill's view is **projected** from Claw into the translator and rendered in a
sandboxed iframe (`allow-scripts` only — never `allow-same-origin`, so it can't reach back
into the page that hosts it):

```
translator ──[hablante + texto]──▶  Claw  ──▶ model ──▶ CONSEJO / DATOS
    ▲                                 │
    └───[surface html + state]────────┘        (rendered sandboxed, right-hand panel)
```

This is written up as a small spec — see **[Skill Surface Protocol](https://github.com/KikoCis/elffuss-claw)**.

## What it does

- **Conversation mode**: one microphone, hands free. It finds the turns by energy, tells
  **who is speaking** from their voice, and routes each turn into the other language.
- **Subtitles** with the original and the translation underneath, growing downward like a real call.
- **Voice**: reads the translation out loud — or mute it and keep translating and subtitling.
- **Half duplex**: it doesn't listen to itself while speaking. Without this it loops, translating its own voice.
- **Battery aware**: pauses in the background and recovers on its own (and never pauses while a live skill is running).
- **Actually offline**: download the models once, they stay cached, and it works with no network.

## The stack, all ONNX in the browser

| Piece | Model |
|---|---|
| **ASR** | `whisper-base` (WebGPU, falls back to WASM) |
| **Diarization** | `pyannote-segmentation-3.0` (speech/silence) + `wespeaker-voxceleb-resnet34-LM` (speaker embeddings) |
| **Translation** | The browser's on-device Translator API, with `opus-mt` (transformers.js) as fallback |
| **Voice** | The system's `speechSynthesis` |

Runs on [transformers.js](https://github.com/huggingface/transformers.js) and ONNX Runtime Web.
No framework and no build step: HTML and ES modules, served as static files.

## Using the diarizer on its own

`diarize.mjs` is standalone:

```js
import { loadDiarizer, diarize, makeSpeakerBook, speakerVec } from './diarize.mjs';

const D = await loadDiarizer(transformers, { device: 'webgpu' });

// (a) split a recorded clip into turns
const turns = await diarize(D, audio16k);        // [{ start, end, speaker }]

// (b) live: assign each chunk to a stable speaker
const book = makeSpeakerBook({ tau: 0.5, maxSpeakers: 2 });
const { label } = book.assign(await speakerVec(D, chunk16k));   // 'S1' | 'S2'
```

Segmentation → per-window embeddings → cosine clustering. Two chunks from the same speaker
land around 0.8 similarity; different speakers, about 0.15.

## Honest edges

- **The first load is a download.** You ship the model to the user once; after that it lives in cache.
- **Separating simultaneous voices on a mono mic is not reliable** — the louder speaker dominates the
  embedding. Turn-by-turn diarization does work; real overlap needs one capture per speaker.
- **Whisper hallucinates on silence.** There's an energy gate and a filter for the usual filler
  ("Subtitles by…"), but some still slips through.
- **Chrome/Edge** for WebGPU. Without it, it falls back to CPU and gets slower.
- Voice quality is whatever the operating system provides, which varies a lot by language.

## Files

| | |
|---|---|
| `index.html` | The whole app: UI, capture, pipeline, live-skill picker and the pipe to Claw. |
| `diarize.mjs` | Diarization engine (segmentation + embeddings + clustering). |

## Hosting it yourself

Static files; any server works. One catch: **`.mjs` must be served as JavaScript**, or the
browser refuses the module. In nginx:

```nginx
location ~ \.mjs$ { default_type text/javascript; }
```

Serve it over HTTPS — the microphone and WebGPU require it outside `localhost`.

## License

Apache-2.0.

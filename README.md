# Elffuss Translator

Traducción de conversaciones en tiempo real, **100% en tu navegador**. Escucha, traduce y responde por voz — sin servidor, sin cuenta, sin API de pago. Nada de lo que digas sale de tu máquina.

Es la versión navegador de la idea de un "traductor de conversaciones" local: el modelo corre en tu propia GPU vía **WebGPU**, y cae a CPU (WASM) cuando no hay GPU disponible.

**En vivo:** [translator.elffuss.utopiaia.com](https://translator.elffuss.utopiaia.com) · Parte de [Elffuss](https://elffuss.utopiaia.com).

## Qué hace

- **Pulsa y habla** por cada lado (dos idiomas). Cada interlocutor con su propia captura.
- **Modo conversación**: un solo micrófono, manos libres. Separa **quién habla** por su voz (diarización) y enruta cada turno al idioma del otro.
- Escucha → transcribe → traduce → responde por voz. Todo local.

## El stack, todo ONNX en el navegador

| Pieza | Modelo |
|---|---|
| **ASR** | Whisper — `whisper-large-v3-turbo` en GPU, `whisper-base` en CPU |
| **Diarización** | `pyannote-segmentation-3.0` (habla/silencio) + `wespeaker` (embeddings de hablante) |
| **Traducción** | Translator API on-device del navegador, con `opus-mt` (transformers.js) de reserva |
| **TTS** | `Kokoro-82M` para voz neural en inglés; voz del sistema para el resto |

Se ejecuta con [transformers.js](https://github.com/huggingface/transformers.js) sobre ONNX Runtime Web. Sin framework, sin build: HTML y ES modules. Los pesos se descargan una vez y quedan cacheados en el navegador (funciona offline después).

## Bordes honestos

- **La primera carga son megas (o gigas en GPU):** le mandas el modelo al usuario una vez; luego, en caché.
- **Separar voces simultáneas en un micro mono no es fiable** — el hablante más fuerte domina el embedding. La separación limpia se resuelve con una captura por interlocutor; la diarización por turnos sí funciona.
- **Kokoro en navegador es solo inglés** (los demás idiomas necesitan un phonemizador que no viene empaquetado); el resto usa la voz del sistema.
- **Chrome/Edge** para WebGPU. Sin él, cae a CPU y va más lento.

## Ficheros

- `index.html` — la app completa (UI + pipeline).
- `diarize.mjs` — motor de diarización por turnos (segmentación + embeddings + clustering).

## Licencia

Apache-2.0.

# tarjama-transcription

GPU transcription worker for the **Tarjama** Arabic ASR platform. Runs the fine-tuned Whisper Large V3 model (converted to CTranslate2, served with faster-whisper) on a dedicated GPU instance. It handles both batch transcription (Kafka tasks) and the live path (reading audio buffers from Redis), then returns results to the pipeline.

## Architecture

Layered like the other workers, with inference isolated behind its own service:

- **Consumer / entrypoint** — subscribes to the transcribe topic (and services live sessions from Redis).
- **Inference service** — loads the CTranslate2 model once and runs transcription on audio buffers.
- **Services** — coordinates fetching audio, running inference, and publishing results.
- **Repositories** — wrap object storage, Redis, and the Kafka producer.
- **Config** — model path, device/compute type, and infrastructure wiring.

The model is fine-tuned and converted separately; this repo is the serving worker.

Part of a multi-service system — see the [platform overview](https://github.com/maleksabbah/tarjama-docker) for the full architecture, pipeline flow, and the other services.

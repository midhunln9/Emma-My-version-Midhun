# Practice Voice Agent

Simple LiveKit voice agent for local console testing.

## Required environment variables

- `LIVEKIT_URL`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`
- `OPENAI_API_KEY`

## Setup

```bash
uv sync
uv run python agent.py download-files
```

## Run in Console

```bash
uv run python agent.py console
```

## Run in Playground

```bash
uv run python agent.py dev
```

The agent name is `practice-voice-agent`.

This version keeps the code intentionally simple:

- STT: OpenAI `gpt-4o-mini-transcribe`
- LLM: OpenAI `gpt-4.1`
- TTS: OpenAI `gpt-4o-mini-tts`
- VAD: Silero

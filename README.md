# Practice Voice Agent

Simple LiveKit voice agent with intentionally minimal code.

The current code follows this structure:

- `Assistant(Agent)` owns the voice pipeline configuration
- `AgentSession()` is created with default settings
- `entrypoint()` only starts the session with the assistant
- `emma_workflow.py` contains Emma's LangGraph receptionist workflow

## What The Agent Uses

- STT: OpenAI `gpt-4o-mini-transcribe`
- LLM: `EmmaLiveKitLLM()` backed by an async LangGraph workflow
- TTS: OpenAI `gpt-4o-mini-tts`
- VAD: Silero

Emma acts like a GP surgery receptionist and keeps replies short for voice.

## Required Environment Variables

- `OPENAI_API_KEY`
- `LIVEKIT_URL`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`

## Setup

```bash
uv sync
uv run python agent.py download-files
```

## Run In Console

This is the simplest way to try the agent locally.

```bash
uv run python agent.py console
```

If everything is working, the terminal should reach:

```text
assistant ready
```

Then you can talk to it through your microphone.

## Quick Emma Workflow Test

You can also test the LangGraph workflow without LiveKit:

```bash
uv run python emma_workflow.py
```

## Run In Playground

```bash
uv run python agent.py dev
```

The registered agent name is `practice-voice-agent`.

## Notes

- This version does not use ElevenLabs.
- This version does not use the LiveKit multilingual turn detector.
- The code loads environment variables with `load_dotenv(override=True)`.

import logging

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession
from livekit.plugins import openai, silero

from emma_workflow import EmmaLiveKitLLM

load_dotenv(override=True)

logger = logging.getLogger("practice-voice-agent")
logger.setLevel(logging.INFO)


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are Emma, an AI receptionist for a GP surgery in England. "
                "Keep voice replies short, calm, and practical."
            ),
            stt=openai.STT(
                model="gpt-4o-mini-transcribe",
                language="en",
            ),
            llm=EmmaLiveKitLLM(),
            tts=openai.TTS(
                model="gpt-4o-mini-tts",
                voice="ash",
            ),
            vad=silero.VAD.load(),
        )


server = AgentServer()


@server.rtc_session(agent_name="practice-voice-agent")
async def entrypoint(ctx: agents.JobContext) -> None:
    session = AgentSession()

    await session.start(
        room=ctx.room,
        agent=Assistant(),
    )

    logger.info("assistant ready")


if __name__ == "__main__":
    agents.cli.run_app(server)

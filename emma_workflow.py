from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable
from copy import deepcopy
from typing import Annotated, Any, Literal
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from livekit.agents import llm as livekit_llm
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from typing_extensions import TypedDict

load_dotenv(override=True)


class PatientInfo(TypedDict, total=False):
    name: str
    date_of_birth: str
    nhs_number: str


class TriageInfo(TypedDict, total=False):
    chief_complaint: str
    duration: str
    severity: str
    red_flags: list[str]


class EMMAState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    patient_info: PatientInfo
    triage_info: TriageInfo
    next_action: Literal[
        "collect_patient_info",
        "collect_symptoms",
        "check_red_flags",
        "offer_gp_appointment",
        "offer_self_care",
        "escalate_999",
        "end_call",
    ] | None
    is_emergency: bool
    turn_count: int


ResponseCallback = Callable[[str], Awaitable[None] | None]


@tool
async def book_gp_appointment(
    patient_name: str,
    preferred_time: str = "next available",
) -> str:
    """Book a GP appointment after the patient confirms they want one."""
    return f"Appointment booked for {patient_name} at {preferred_time}. Confirmation SMS sent."


@tool
async def check_symptom_urgency(symptoms: list[str]) -> dict[str, str]:
    """Check whether reported symptoms indicate an emergency, urgent, or routine case."""
    emergency_keywords = [
        "chest pain",
        "difficulty breathing",
        "stroke",
        "unconscious",
        "severe bleeding",
        "allergic reaction",
        "can't breathe",
        "hard to breathe",
    ]
    urgent_keywords = [
        "high fever",
        "vomiting blood",
        "sudden severe headache",
        "confusion",
        "severe abdominal pain",
    ]

    symptoms_lower = [symptom.lower() for symptom in symptoms]

    for symptom in symptoms_lower:
        for keyword in emergency_keywords:
            if keyword in symptom:
                return {"urgency": "emergency", "reason": f"Possible emergency: {keyword}"}

    for symptom in symptoms_lower:
        for keyword in urgent_keywords:
            if keyword in symptom:
                return {"urgency": "urgent", "reason": f"Urgent symptom: {keyword}"}

    return {"urgency": "routine", "reason": "Symptoms appear routine"}


@tool
async def get_self_care_advice(condition: str) -> str:
    """Retrieve basic self-care advice for common minor conditions."""
    advice_map = {
        "cold": "Rest, stay hydrated, and use paracetamol for fever if needed.",
        "sore throat": "Drink warm fluids, gargle salt water, and rest your voice.",
        "minor cut": "Clean it with running water, apply pressure, and use a plaster.",
    }
    for key, advice in advice_map.items():
        if key in condition.lower():
            return advice

    return (
        f"For {condition}, monitor the symptoms and contact the surgery if they worsen or do not improve."
    )


EMMA_SYSTEM_PROMPT = """You are EMMA, an AI receptionist for a GP surgery in England.

Your job is to:
1. Greet the patient warmly and collect their name and date of birth
2. Understand their reason for calling
3. Assess urgency and always check for red flags
4. Either book a GP appointment, give self-care advice, or direct the patient to 999

CRITICAL RULES:
- If the patient mentions chest pain, difficulty breathing, stroke symptoms, or severe bleeding, direct them to call 999 immediately.
- Keep responses short because this is a live voice call. Use 1-2 short sentences.
- Speak in plain English and never diagnose.
- Ask only the next best question. Do not ask multiple stacked questions unless needed.
- Use the urgency tool before deciding on an urgent or emergency care pathway.

Current patient state:
{state_summary}
"""


EXTRACTOR_PROMPT = """Extract structured information from this GP receptionist conversation.
Return ONLY valid JSON with these fields:
{{
  "name": "patient name or null",
  "date_of_birth": "DOB if mentioned or null",
  "chief_complaint": "main symptom or reason for calling or null",
  "severity": "mild, moderate, severe, or null",
  "red_flags": ["list", "of", "red", "flags"]
}}

Do not infer facts that were not explicitly said.

Conversation:
{conversation}
"""


def empty_emma_state() -> EMMAState:
    return {
        "messages": [],
        "patient_info": {},
        "triage_info": {},
        "next_action": None,
        "is_emergency": False,
        "turn_count": 0,
    }


class EmmaWorkflow:
    def __init__(
        self,
        *,
        triage_model: str = "gpt-4o-mini",
        extractor_model: str = "gpt-4o-mini",
    ) -> None:
        self._tools: list[BaseTool] = [
            book_gp_appointment,
            check_symptom_urgency,
            get_self_care_advice,
        ]
        self._triage_llm = ChatOpenAI(model=triage_model, temperature=0)
        self._triage_llm_with_tools = self._triage_llm.bind_tools(self._tools)
        self._extractor_llm = ChatOpenAI(model=extractor_model, temperature=0)
        self._graph = self._build_graph()
        self._state: EMMAState = empty_emma_state()
        self._lock = asyncio.Lock()

    @property
    def state(self) -> EMMAState:
        return deepcopy(self._state)

    def _build_graph(self):
        graph = StateGraph(EMMAState)
        graph.add_node("triage", self._triage_node)
        graph.add_node("extract_state", self._state_extractor_node)
        graph.set_entry_point("triage")
        graph.add_edge("triage", "extract_state")
        graph.add_edge("extract_state", END)
        return graph.compile()

    def _format_state_summary(self, state: EMMAState) -> str:
        parts: list[str] = []
        patient_info = state.get("patient_info", {})
        triage_info = state.get("triage_info", {})

        if patient_info.get("name"):
            parts.append(f"Patient name: {patient_info['name']}")
        else:
            parts.append("Patient name: NOT YET COLLECTED")

        if patient_info.get("date_of_birth"):
            parts.append(f"DOB: {patient_info['date_of_birth']}")

        if triage_info.get("chief_complaint"):
            parts.append(f"Reason for call: {triage_info['chief_complaint']}")
        else:
            parts.append("Reason for call: NOT YET COLLECTED")

        if triage_info.get("severity"):
            parts.append(f"Severity: {triage_info['severity']}")

        if triage_info.get("red_flags"):
            parts.append(f"Red flags: {', '.join(triage_info['red_flags'])}")

        if state.get("is_emergency"):
            parts.append("Emergency already identified: YES")

        return "\n".join(parts)

    def _message_text(self, message: BaseMessage) -> str:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
            return " ".join(part.strip() for part in text_parts if part).strip()
        return ""

    def _latest_assistant_text(self, messages: list[BaseMessage]) -> str | None:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                text = self._message_text(message)
                if text:
                    return text
        return None

    def _assistant_message_count(self, messages: list[BaseMessage]) -> int:
        return sum(1 for message in messages if isinstance(message, AIMessage) and self._message_text(message))

    async def _triage_node(self, state: EMMAState) -> dict[str, Any]:
        system_prompt = EMMA_SYSTEM_PROMPT.format(
            state_summary=self._format_state_summary(state),
        )
        messages_for_llm = [SystemMessage(content=system_prompt), *state["messages"]]

        planner_response = await self._triage_llm_with_tools.ainvoke(messages_for_llm)
        new_messages: list[BaseMessage] = []
        next_action = state.get("next_action")
        is_emergency = state.get("is_emergency", False)

        if planner_response.tool_calls:
            tool_results: list[tuple[str, Any]] = []
            tool_map = {tool.name: tool for tool in self._tools}

            for tool_call in planner_response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool = tool_map.get(tool_name)
                if tool is None:
                    continue

                result = await tool.ainvoke(tool_args)
                tool_results.append((tool_name, result))

                if tool_name == "check_symptom_urgency":
                    urgency_data = result if isinstance(result, dict) else json.loads(result)
                    urgency = urgency_data.get("urgency")
                    if urgency == "emergency":
                        is_emergency = True
                        next_action = "escalate_999"
                    elif urgency == "urgent":
                        next_action = "offer_gp_appointment"
                    else:
                        next_action = "offer_self_care"
                elif tool_name == "book_gp_appointment":
                    next_action = "offer_gp_appointment"
                elif tool_name == "get_self_care_advice":
                    next_action = "offer_self_care"

            tool_observation = "\n".join(
                f"[{tool_name}]: {result}" for tool_name, result in tool_results
            )
            internal_observation = SystemMessage(
                content=(
                    "[TOOL RESULTS]\n"
                    f"{tool_observation}\n\n"
                    "Use these internal results to respond to the patient in 1-2 short sentences."
                )
            )
            final_response = await self._triage_llm.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    *state["messages"],
                    internal_observation,
                ]
            )
            new_messages.extend([internal_observation, final_response])
        else:
            direct_text = self._message_text(planner_response)
            if direct_text:
                new_messages.append(planner_response)
            else:
                new_messages.append(AIMessage(content="I'm sorry, could you repeat that?"))

        return {
            "messages": new_messages,
            "turn_count": state.get("turn_count", 0) + 1,
            "next_action": next_action,
            "is_emergency": is_emergency,
        }

    async def _state_extractor_node(self, state: EMMAState) -> dict[str, Any]:
        conversation_lines: list[str] = []
        for message in state["messages"]:
            text = self._message_text(message)
            if not text:
                continue

            if isinstance(message, HumanMessage):
                conversation_lines.append(f"Patient: {text}")
            elif isinstance(message, AIMessage):
                conversation_lines.append(f"EMMA: {text}")

        if not conversation_lines:
            return {}

        try:
            response = await self._extractor_llm.ainvoke(
                [
                    HumanMessage(
                        content=EXTRACTOR_PROMPT.format(
                            conversation="\n".join(conversation_lines)
                        )
                    )
                ]
            )
            raw = self._message_text(response).replace("```json", "").replace("```", "").strip()
            extracted = json.loads(raw)
        except Exception:
            return {}

        patient_info = {**state.get("patient_info", {})}
        triage_info = {**state.get("triage_info", {})}

        if extracted.get("name"):
            patient_info["name"] = extracted["name"]
        if extracted.get("date_of_birth"):
            patient_info["date_of_birth"] = extracted["date_of_birth"]
        if extracted.get("chief_complaint"):
            triage_info["chief_complaint"] = extracted["chief_complaint"]
        if extracted.get("severity"):
            triage_info["severity"] = extracted["severity"]
        if extracted.get("red_flags"):
            triage_info["red_flags"] = extracted["red_flags"]

        return {
            "patient_info": patient_info,
            "triage_info": triage_info,
        }

    async def process_turn(
        self,
        patient_text: str,
        *,
        on_response: ResponseCallback | None = None,
    ) -> str:
        if not patient_text.strip():
            return "I'm sorry, I didn't catch that. Could you say that again?"

        async with self._lock:
            working_state = deepcopy(self._state)
            starting_ai_count = self._assistant_message_count(working_state["messages"])
            working_state["messages"] = [*working_state["messages"], HumanMessage(content=patient_text)]

            response_text: str | None = None

            async for snapshot in self._graph.astream(working_state, stream_mode="values"):
                working_state = snapshot
                current_ai_count = self._assistant_message_count(working_state["messages"])
                if response_text is None and current_ai_count > starting_ai_count:
                    response_text = self._latest_assistant_text(working_state["messages"])
                    if response_text and on_response is not None:
                        callback_result = on_response(response_text)
                        if inspect.isawaitable(callback_result):
                            await callback_result

            self._state = working_state
            return response_text or "I'm sorry, could you repeat that?"

    def should_end_call(self) -> bool:
        if self._state.get("is_emergency"):
            return True

        last_response = self._latest_assistant_text(self._state["messages"])
        if not last_response:
            return False

        content = last_response.lower()
        return any(
            phrase in content
            for phrase in (
                "appointment booked",
                "take care",
                "goodbye",
                "thank you for calling",
                "call 999",
                "call an ambulance",
            )
        )


class EmmaLiveKitLLM(livekit_llm.LLM):
    def __init__(self, *, workflow: EmmaWorkflow | None = None) -> None:
        super().__init__()
        self._workflow = workflow or EmmaWorkflow()

    @property
    def model(self) -> str:
        return "emma-langgraph"

    @property
    def provider(self) -> str:
        return "local"

    def chat(
        self,
        *,
        chat_ctx: livekit_llm.ChatContext,
        tools: list[livekit_llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[livekit_llm.ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> "EmmaLiveKitLLMStream":
        del parallel_tool_calls, tool_choice, extra_kwargs
        return EmmaLiveKitLLMStream(
            self,
            workflow=self._workflow,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
        )

    async def aclose(self) -> None:
        return None


class EmmaLiveKitLLMStream(livekit_llm.LLMStream):
    def __init__(
        self,
        llm: EmmaLiveKitLLM,
        *,
        workflow: EmmaWorkflow,
        chat_ctx: livekit_llm.ChatContext,
        tools: list[livekit_llm.Tool],
        conn_options: APIConnectOptions,
    ) -> None:
        self._workflow = workflow
        self._chunk_id = f"emma_{uuid4().hex}"
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)

    async def _run(self) -> None:
        user_text = self._latest_user_text()
        response_sent = False

        async def emit_response(text: str) -> None:
            nonlocal response_sent
            if response_sent:
                return

            self._event_ch.send_nowait(
                livekit_llm.ChatChunk(
                    id=self._chunk_id,
                    delta=livekit_llm.ChoiceDelta(role="assistant", content=text),
                )
            )
            response_sent = True

        if not user_text:
            await emit_response("Hello, you've reached the GP surgery. This is Emma. How can I help today?")
            return

        response_text = await self._workflow.process_turn(user_text, on_response=emit_response)
        if not response_sent:
            await emit_response(response_text)

    def _latest_user_text(self) -> str:
        for message in reversed(self._chat_ctx.messages()):
            if message.role != "user":
                continue
            if message.text_content:
                return message.text_content.strip()
        return ""


async def _local_test() -> None:
    print("=== EMMA Local Test ===\n")

    workflow = EmmaWorkflow()
    test_conversation = [
        "Hello, I'd like to see a doctor",
        "My name is James Kumar, date of birth 15th March 1982",
        "I've had a bad cough for about 4 days now, and a mild fever",
        "Actually, I also have some chest tightness and it's hard to breathe",
    ]

    for patient_turn in test_conversation:
        print(f"Patient: {patient_turn}")
        emma_response = await workflow.process_turn(patient_turn)
        print(f"EMMA:    {emma_response}\n")

        if workflow.should_end_call():
            print("[Call ended]")
            break


if __name__ == "__main__":
    asyncio.run(_local_test())

from typing import List, Union, Literal
from dataclasses import dataclass

# --- Domain Data Structures ---
@dataclass
class RequestAgent:
    name: str
    instruction: str

@dataclass
class RequestAgentResult:
    answer: str

@dataclass
class RequestHumanInput:
    question: str


@dataclass
class RequestHumanInputResult:
    answer: str

# --- Event & Thread Models ---
@dataclass
class Event:
    type: Literal[
        "request_agent",
        "request_agent_result",
        "request_human_input",
        "request_human_input_result"
    ]
    data: Union[
        RequestAgent,
        RequestAgentResult,
        RequestHumanInput,
        RequestHumanInputResult,
        # str  # Fallback for raw strings
    ]

@dataclass
class Thread:
    id: str
    events: List[Event]
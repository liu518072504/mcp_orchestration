from pydantic import BaseModel, Field
from typing import Optional

# Planning agent schema
class PlanStep(BaseModel):
    agent: str = Field(..., description="Which agent performs this step")
    instruction: str = Field(..., description="The instruction for the agent to perform this step")

class Plan(BaseModel):
    steps: list[PlanStep]

# Orchestrator schema
class OrchestratorSchema(BaseModel):
    finished: bool = Field(..., description="True if the orchestration is complete and no more agents need to be called, False otherwise")
    agent:  Optional[str] = Field(None, description="The name of the agent that should perform the next step")
    instruction: Optional[str] = Field(None, description="The instruction for the agent to perform the next step")

# Function call schema
class FunctionCall(BaseModel):
    function: str = Field(..., description="The name of the function to call")
    args: dict = Field(..., description="The arguments to pass to the function")
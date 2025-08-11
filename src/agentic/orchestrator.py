from typing import List
from agentic.agent import Agent
import inspect
from langchain.output_parsers import PydanticOutputParser

from agentic.schemas.schemas import OrchestratorSchema, Plan, FunctionCall
from agentic.database.filedb import FileDB

class Orchestrator:
    TRUNCATE = 500

    def __init__(self, available_agents: List[Agent], functions: List[callable]):
        self.agents = {agent.name: agent for agent in available_agents}
        self.functions = functions
        exact_params = [f"{func.__name__}: {inspect.signature(func)}" for func in self.functions]

        self.db = FileDB()

        # Create special agents
        self.function_agent = Agent(
            name="function_calling_agent",
            instruction=f"""You are a function calling agent. 
            Your job is to return a JSON object following your schema containing
            the exact function name and the arguments.

            Here are the list of functions with their exact parameters:
            {exact_params}
            """,
            use_memory=False,
            db=self.db,
        )
        self.agents["function_calling_agent"] = self.function_agent

        self.planning_agent = Agent(
            name="planner",
            instruction=f"""You are an expert planner. Given an objective task and a list of Agents (LLMs) with access to various tools, 
            your job is to break down the objective into a series of steps, which can be performed by the available agents. 
            Review the functions available to the function_calling_agent and use it when appropriate.

            You must only return a JSON object adhering to the schema, and nothing else.

            If you deem no agents are neccecary to answer the query, you can respond directly by returning one step with agent 'end' and an appropriate response in 'instruction'.

            """,
            db=self.db,
            use_memory=False
        )

        self.orchestrator_agent = Agent(
            name="orchestrator",
            instruction=f"""You are an orchestrator agent. Given a pre-defined plan in chronological order, your job is to decide the next agent to execute with a set of instructions.
            The plan is just a guideline, and you can choose the best agent to execute the next step, but follow the plan to the best of your ability.
            Review the functions available to the function_calling_agent and use it when appropriate.

            In the instruction, ensure to provide sufficient context as agents do not have access to previous messages.

            Return in a JSON object with the following fields (and nothing else):
            - finished: True if the orchestration is complete and no more agents need to be called or there is a critical error that cannot be resolved. False otherwise
            - agent: the name of the agent that should perform the next step (review the plan when deciding this agent)
            - instruction: the instruction for the agent to perform the next step (Answer to the user query if finished)
            """,
            db=self.db,
        )

        # Create custom output parsers for the custom agents
        self.parsers = {
            "orchestrator": PydanticOutputParser(pydantic_object=OrchestratorSchema),
            "planner": PydanticOutputParser(pydantic_object=Plan),
            "function_call": PydanticOutputParser(pydantic_object=FunctionCall),
        }

        self.available_agents_prompt = "\n".join([f"{agent.name}: {agent.instruction}" for agent in self.agents.values()]
        )

    async def _generate_plan(self, query: str):
        """Generate a plan using the planning agent."""
        parser = self.parsers["planner"]
        format_instructions = parser.get_format_instructions()

        query_with_format = f"{query}\n\nAvailable agents:\n{self.available_agents_prompt}"
        result = await self.planning_agent.generate_str(query_with_format, format_instructions)
        plan: Plan = parser.parse(result)
        return plan.model_dump()

    # Depreciated
    async def _plan_cleanup(self, plan: Plan):
        """Merge adjacent steps with the same agent in the plan except for the function_calling_agent."""
        cleaned_steps = []
        for step in plan.steps:
            if cleaned_steps and cleaned_steps[-1].agent == step.agent and step.agent != "function_calling_agent":
                # Merge with the last step if the agent is the same
                cleaned_steps[-1].instruction += " Then, " + step.instruction
            else:
                cleaned_steps.append(step)
        #return json
        return Plan(steps=cleaned_steps).model_dump()

    async def _output_cleanup(self, raw: str, parser: PydanticOutputParser) -> str:
        """Grab only the json part of the output. By grabbing the first { and the last }."""
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("Invalid output format, no JSON found.")
        try:
            # Check if the output is valid JSON
            return parser.parse(raw[start:end + 1]).model_dump()
        except Exception as e:
            print(f"Error cleaning up output: {e}")
            raise ValueError("Invalid output format, no JSON found.")

    async def _run_function(self, instruction: str):
        parser = self.parsers["function_call"]
        format_instructions = parser.get_format_instructions()

        prompt = f"""
        You are a function calling agent. Your job is to return a JSON object with the exact function name and the arguments.
        Instruction: {instruction}
        """

        raw = await self.function_agent.generate_str(prompt, format_instructions)

        # Clean up the output to get the JSON part
        result = await self._output_cleanup(raw, parser)


        function_name = result.get("function")
        function_args = result.get("args")

        
        for func in self.functions:
            if func.__name__ == function_name:
                print(f"Executing function {function_name}")
                result = func(**function_args)

                message = f"Function {function_name} executed successfully with arguments {function_args}"
                if result is not None:
                    message += f" and returned {result}"
                return message

    async def orchestrate(self, query: str, max_steps: int = 10):
        """Orchestrate the execution of the plan."""

        # Clear the orchestrator thread before starting a new orchestration, can remove this if you want to keep the history
        await self.db.delete_thread("orchestrator")

        plan = await self._generate_plan(query)
        # plan = await self._plan_cleanup(plan)

        #if the agent is end, get the instruction:
        if plan.get("steps")[0].get("agent") == "end":
            instruction = plan.get("steps")[0].get("instruction")
            return instruction

        print("Generated plan:")
        for step in plan.values():
            for s in step:
                print(f"Agent: {s.get('agent')}, Instruction: {s.get('instruction')[:self.TRUNCATE]}...")


        format_instructions = self.parsers["orchestrator"].get_format_instructions()
        for _ in range(max_steps):
            query_with_format = f"{query}\n\nPlan: {plan}\n\nAvailable agents:\n{self.available_agents_prompt}"
            raw = await self.orchestrator_agent.generate_str(query_with_format, format_instructions)

            # Clean up the output to get the JSON part
            next_step_json = await self._output_cleanup(raw, self.parsers["orchestrator"])
            print("\nOrchestrator response:")
            #if finished is true return entire response
            if next_step_json.get("finished"):
                print("Orchestrator finished with response:", next_step_json.get("instruction"))
                return next_step_json.get("instruction")
            else:
                print("agent:", next_step_json.get("agent"), ", instruction:", next_step_json.get("instruction")[:self.TRUNCATE] + "...,", "finished:", next_step_json.get("finished"))

            agent_name = next_step_json.get("agent")
            if agent_name not in self.agents:
                # TODO reloop
                raise ValueError(f"Agent {agent_name} not found in available agents.")
            
            instruction = next_step_json.get("instruction")

            if agent_name == "function_calling_agent":
                result = await self._run_function(instruction)
                
            else:
                prompt = f"Instruction: {instruction}"
                result = await self.agents[agent_name].generate_str(prompt)

            print(f"\nAgent {agent_name} returned: {result[:self.TRUNCATE]}...")

            # Update the orchestrator thread with the result
            await self.orchestrator_agent.update_thread(
                prompt=instruction,
                result=result,
                name=agent_name
            )
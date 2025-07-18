from typing import List
from agentic.agent import Agent
import inspect
from langchain.output_parsers import PydanticOutputParser

from agentic.schemas.schemas import OrchestratorSchema, Plan, FunctionCall
from agentic.database.filedb import FileDB

class Orchestrator:
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
            instruction="""You are an expert planner. Given an objective task and a list of Agents (LLMs) with access to various tools, 
            your job is to break down the objective into a series of steps, which can be performed by the available agents. 
            Review the functions available to the function_calling_agent and use it when appropriate.""",
            db=self.db,
            use_memory=False
        )

        self.orchestrator_agent = Agent(
            name="orchestrator",
            instruction=f"""You are an orchestrator agent. Given a pre-defined plan, your job is to decide the next agent to execute with a set of instructions.
            The plan is just a guideline, and you can choose the best agent to execute the next step.
            Review the functions available to the function_calling_agent and use it when appropriate.
            
            Return in a JSON object with the following fields:
            - finished: True if the orchestration is complete and no more agents need to be called, False otherwise
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
    
    async def _run_function(self, instruction: str):
        parser = self.parsers["function_call"]
        format_instructions = parser.get_format_instructions()

        prompt = f"""
        You are a function calling agent. Your job is to return a JSON object with the exact function name and the arguments.
        Instruction: {instruction}
        """

        raw = await self.function_agent.generate_str(prompt, format_instructions)
        print(f"\nFunction call raw output: {raw}\n")

        result = parser.parse(raw).model_dump()

        function_name = result.get("function")
        function_args = result.get("args")

        
        for func in self.functions:
            if func.__name__ == function_name:
                print(f"Executing function {function_name} with args {function_args}")
                result = func(**function_args)

                message = f"Function {function_name} executed successfully with arguments {function_args}"
                if result is not None:
                    message += f" and returned {result}"
                return message

    async def orchestrate(self, query: str, max_steps: int = 10):
        """Orchestrate the execution of the plan."""
        plan = await self._generate_plan(query)


        format_instructions = self.parsers["orchestrator"].get_format_instructions()
        for _ in range(max_steps):
            query_with_format = f"{query}\n\nPlan: {plan}\n\nAvailable agents:\n{self.available_agents_prompt}"
            raw = await self.orchestrator_agent.generate_str(query_with_format, format_instructions)

            print(f"Orchestrator output: {raw}")

            next_step_json = self.parsers["orchestrator"].parse(raw).model_dump()

            if next_step_json.get("finished"):
                print("MCP Orchestration: Orchestration finished.")
                break

            agent_name = next_step_json.get("agent")
            if agent_name not in self.agents:
                # TODO reloop
                raise ValueError(f"Agent {agent_name} not found in available agents.")
            
            instruction = next_step_json.get("instruction")

            if agent_name == "function_calling_agent":
                print(f"Running function call with instruction: {instruction}")
                result= await self._run_function(instruction)
                
            else:
                print(f"Running agent {agent_name} with instruction: {instruction}")
                prompt = f"Instruction: {instruction}"
                result = await self.agents[agent_name].generate_str(prompt)
            
            # Update the orchestrator thread with the result
            await self.orchestrator_agent.update_thread(
                prompt=instruction,
                result=result,
                name=agent_name
            )
        
#test
if __name__ == "__main__":
    import asyncio
    from agentic.USER.functions import create_python_file, execute_python_file, obtain_csv_header
    from agentic.USER.agents import filesystem_agent, summary_agent
    orchestrator = Orchestrator(
        available_agents=[],
        functions=[create_python_file, execute_python_file, obtain_csv_header]
    )
    query = """Make a graph of the revenue per month from the CSV file 'database/example_datasets/company1_sales_new.csv'. 
        Save the graph as 'revenue_graph.png' in the current working directory."""
    # query = "Create a python file that contains a function to add two numbers and return the result."
    asyncio.run(orchestrator.orchestrate(query))
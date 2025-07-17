import os
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from mcp_use import MCPClient, MCPAgent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

import agentic
from agentic.database.models import Event, RequestAgent, RequestAgentResult
from agentic.database.filedb import FileDB
from agentic.database.serializer import thread_to_prompt


class Agent:
    def __init__(
            self, name: str,
            instruction: str,
            servers: list[str],
            db: FileDB,
            model_name: str = "gemini-2.0-flash",
            # model_name: str = "claude-3-haiku-20240307",
            temperature: float = 0.0, 
            config_path: str = Path(agentic.__file__).parent.parent.parent / "mcp_config.json",
            max_steps: int = 15,
            use_memory: bool = True
        ):
        
        load_dotenv()
        self.name = name
        self.system_prompt = instruction
        self.db = db
        self.thread = None
        self.use_memory = use_memory

        # Load & filter MCP servers
        with open(config_path, "r") as f:
            cfg = json.load(f)
        all_servers = cfg["mcpServers"]

        selected = {srv: all_servers[srv] for srv in servers}
        self.client = MCPClient.from_dict({"mcpServers": selected})
        


        self.llm = None
        if "gemini" in model_name.lower():
            # Initialize google llm
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                # max_tokens=None,
                # timeout=None,
            )
        elif "claude" in model_name.lower():
            # Initialize Anthropic llm
            self.llm = ChatAnthropic(
                model=model_name,
                temperature=temperature,
            )
        
        self.agent = None
        
        if len(servers) > 0:
            # Build the MCPAgent
            self.agent = MCPAgent(
                llm=self.llm,
                client=self.client,
                max_steps= max_steps,
                memory_enabled=False,
                use_server_manager= True,
            )

    async def init(self):
        """Load saved agent history from database. Creates a new thread if none exists."""
        self.thread = await self.db.load_thread(self.name)

    async def save(self):
        """Save the current history to the file database."""
        if self.thread:
            await self.db.save_thread(self.thread)

    async def _run(self, prompt: str) -> str:
        history = ""
        if self.use_memory:
            history = thread_to_prompt(self.thread)
            system_message = f"{self.system_prompt}\n\nYour history:\n{history} \n\nCurrent task:\n{prompt}\n"
        else:
            system_message = f"{self.system_prompt}\n\nCurrent task:\n{prompt}\n"
        
        result = ""
        if not self.agent:
            # Use LLM directly if no agent is defined (Used when no servers are provided)
            result = await self.llm.ainvoke(system_message)
            result = result.content
        else:
            # Use the MCPAgent to run the task when servers are provided
            result = await self.agent.run(system_message)
            if "Final Answer:" in result:
                result = result.split("Final Answer:")[-1].strip()
        return result
        

    
    async def _update_thread(self, prompt: str, result: str):
        """Update the thread with the new prompt and result."""
        if self.use_memory:
            # Update the thread with the new events
            new_event = Event(
                type="request_agent",
                data= RequestAgent(
                    name=self.name,
                    instruction=prompt
                )
            )
            new_event_result = Event(
                type="request_agent_result",
                data=RequestAgentResult(
                    answer=result
                )
            )

            self.thread.events.append(new_event)
            self.thread.events.append(new_event_result)
        

    async def generate_str(self, prompt: str) -> str:
        """Run your agent on the prompt and return the text."""
        await self.init()  # Ensure thread is loaded

        result = await self._run(prompt)

        await self._update_thread(prompt, result)  # Update thread with new events
        await self.save()  # Save the updated thread

        return result

async def main():
    planning_agent = Agent(
        name="planner",
        instruction="You are an expert planner.",
        servers=["filesystem"],
        db=FileDB(),
        use_memory=False
    )
    # structure = {"Answer": "summary of execution"}
    # query = "Use the search_file function in the rag server with this input: What is the total amount of purchases? To answer this question, what dataset should we refer to?"
    # query = "index page2.pdf"
    query = "Create a python file that contains a function to add two numbers and return the result."
    result = await planning_agent.generate_str(query)

    print("Agent output:\n", result)

if __name__ == "__main__":
    asyncio.run(main())
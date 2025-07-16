import os
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from mcp_use import MCPClient, MCPAgent
from langchain_google_genai import ChatGoogleGenerativeAI
from models import Event, RequestAgent, RequestAgentResult

from filedb import FileDB
from serializer import thread_to_prompt


class Agent:
    def __init__(
            self, name: str,
            instruction: str,
            servers: list[str],
            db: FileDB,
            model_name: str = "gemini-2.0-flash",
            temperature: float = 0.0, 
            config_path: str = Path(__file__).parent.parent / "mcp_config.json",
            max_steps: int = 10,
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

        # Initialize google llm
        self.llm = ChatGoogleGenerativeAI(
            model = model_name,
            temperature = temperature,
            max_tokens = None,
            timeout = None,
        )

        # Build the MCPAgent
        self.agent = MCPAgent(
            llm=self.llm,
            client=self.client,
            max_steps=max_steps,
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
    


    async def generate_str(self, prompt: str) -> str:
        """Run your agent on the prompt and return the text."""
        await self.init()  # Ensure thread is loaded

        history = ""

        if self.use_memory:
            history = thread_to_prompt(self.thread)

        system_message = f"{self.system_prompt}\n\nYour history:\n{history} \n\nCurrent task:\n{prompt}\n"

        # Run the agent
        result = await self.agent.run(system_message)
        
        # Only obtain final answer
        # if "Final Answer:" in result:
        #     result = result.split("Final Answer:")[-1].strip()
        # else:
            # result = result.strip()

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

        await self.save()  

        return result

async def main():
    planning_agent = Agent(
        name="planner",
        instruction="You are an expert planner.",
        servers=["rag"],
        db=FileDB(),
        use_memory=False
    )
    # structure = {"Answer": "summary of execution"}
    # query = "Use the search_file function in the rag server with this input: What is the total amount of purchases? To answer this question, what dataset should we refer to?"
    query = "index page2.pdf"
    result = await planning_agent.generate_str(query)

    print("Agent output:\n", result)

if __name__ == "__main__":
    asyncio.run(main())
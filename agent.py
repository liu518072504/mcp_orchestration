import os
import json
import asyncio
from dotenv import load_dotenv
from mcp_use import MCPClient, MCPAgent
from langchain_google_genai import ChatGoogleGenerativeAI


class Agent:
    def __init__(
            self, name: str,
            instruction: str,
            servers: list[str],
            model_name: str = "gemini-2.0-flash",
            temperature: float = 0.0, 
            config_path: str = "mcp_config.json", 
            max_steps: int = 20
        ):
        
        load_dotenv()
        self.name = name
        self.system_prompt = instruction

        # Load & filter MCP servers
        with open(config_path, "r") as f:
            cfg = json.load(f)
        all_servers = cfg["mcp"]["servers"]
        selected = {srv: all_servers[srv] for srv in servers}
        self.client = MCPClient.from_dict({"mcpServers": selected})

        # Initialize google llm
        self.llm = ChatGoogleGenerativeAI(
            model = model_name,
            temperature = temperature,
            max_tokens = None,
            timeout = None,
        )

        # 4️⃣ Build the MCPAgent
        self.agent = MCPAgent(
            llm=self.llm,
            client=self.client,
            max_steps=max_steps,
            auto_initialize=True,
            memory_enabled=False
        )

    async def generate_str(self, prompt: str) -> str:
        """Run your agent on the prompt and return the text."""
        return await self.agent.run(prompt)

    async def close(self):
        """Clean up background MCP processes."""
        await self.client.close_all_sessions()

async def main():
    planning_agent = Agent(
        name="planner",
        instruction="You are an expert planner.",
        servers=["playwright"],
    )
    result = await planning_agent.generate_str(
        "Plan a two-day trip to Toronto with a focus on art galleries. Choose four from https://artguide.artforum.com/artguide/place/toronto?category=galleries." \
        "Explain in detail if any errors occurred"

    )
    print("Agent output:\n", result)
    await planning_agent.close()

if __name__ == "__main__":
    asyncio.run(main())
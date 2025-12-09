# AI-Powered Analytics Platform​
### Automatically creates and executes workflows

Think of it as a chatbot that doesn't just return words: it can execute actions

You have full control over the the external capabilities of the orchestrator:
1. Create your own custom agents with access to mcp servers (eg: mcp playwright for web browser automation)
2. Create your own custom functions that the orchestration has access to (eg: create a python file, execute a python file)

## Setup:
clone the repository
```bash
git clone https://github.com/QixinL/mcp_orchestration.git
cd mcp_orchestration
```

Create venv (optional, recommended):
```bash
python -m venv venv #venv/bin/activate on linux
venv\scripts\activate
```

Install dependencies (Takes around 3 minutes > TODO change to UV package management for faster install maybe?) 
```bash
pip install -e .
```

### LLM Config

Create a .env file in the (root) current folder (mcp_orchestration)
and put your api key in the .env like so:
GOOGLE_API_KEY=...

You can grab the free version that I am currently using at: https://aistudio.google.com/app/apikey

Anthropic setup is the same:
ANTHROPIC_API_KEY=...
(Try not to use anthropic right now not sure why it breaks rather frequently, will need to fix this later)

Currently using google "gemini-2.0-flash" should yield decent results, if you want to change the model you can change the default in the init of:
src/agentic/agent.py


Now try running it! You can change the query in the main.py file. Some examples are provided
```bash
cd test
python main.py
```

## Customization

There is a folder called USER in test (same folder as main.py)
Define your agents and functions in the agents.py and functions.py folder
There are already some examples in there to copy

Import them into main.py and pass them into the orchestrator.

## Memory

Each agent you create will have its own memory stored in a .json file in src/agentic/database/threads
The memory for the agents will be maintained even if the code is rerun.

To clear the memory, you can:
1. delete the .json file of specific agents or replace the entire .json file with an empty array []
2. or delete the entire threads/ folder.

They will be automatically created again when the code is run.
You can set your own agents not to use memory with use_memory=False










### Used mcp-use for integration with mcp servers, not sure where to put this so will leave here for now:
```bibtex
@software{mcp_use2025,
  author = {Zullo, Pietro},
  title = {MCP-Use: MCP Library for Python},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/pietrozullo/mcp-use}
}
```


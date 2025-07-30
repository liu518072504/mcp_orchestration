from agentic.agent import Agent
from agentic.database.filedb import FileDB

# Only need to initialize one database instance that will 
db = FileDB()

# You can create agents with different functionalities. Each agent will have its own memory.

# Access the internet

# I HIGHLY RECOMMEND PROVIDING RELEVANT FUNCTIONS IN THE INSTRUCTION. This will save a LOT of tokens and time.
playwright_agent = Agent(
    name="playwright_agent",
    instruction="""You are a Playwright agent.
    You have access to the playwright mcp server, which has tools that allow you to interact with web pages.
    
    Here are some useful tools in the playwright mcp server:
    browser_navigate(url: str) - Navigate to a URL in the browser.
    browser_snapshot() - obtain an LLM‑friendly DOM representation
    """,
    db=db,
    servers=["playwright"],
    use_memory=False, # You can set this to True if you want the agent to remember its own previous context.
)

rag_agent = Agent(
    name="rag_agent",
    instruction="""You are a RAG (Retrieval-Augmented Generation) agent.
    You have access to the rag mcp server, which has tools that allow you to retrieve and manipulate documents.
    """,
    db=db,
    servers=["rag"],
    use_memory=False,
)

# Specialized agent example
schema_agent = Agent(
    name="schema_agent",
    instruction="""You are an expert in evaluating the quality of database schemas.
    Given a database schema, your job is to evaluate its quality based on these four criteria:
    1. Normalization: Ensure the schema is normalized to at least 3NF.
    2. Redundancy: Identify and eliminate any redundant data.
    3. Relationships: Ensure that relationships between tables are properly defined and enforced.
    4. Indexing: Suggest appropriate indexing strategies to optimize query performance.

    """,
    db=db,
)
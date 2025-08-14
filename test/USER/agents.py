from agentic.agent import Agent
from agentic.database.filedb import FileDB

# Only need to initialize one database instance, I feel like this is bad coding practice since this is never called explicitly hmm
db = FileDB()

# You can create agents with different functionalities. Each agent will have its own memory.
# I HIGHLY RECOMMEND PROVIDING RELEVANT FUNCTIONS IN THE INSTRUCTION. This will save a LOT of tokens and time.

# Access the internet
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
    You have access to the rag mcp server, which has tools that allow you to find tables relevant to a query and obtain the path and schema.

    Here are some useful tools in the rag mcp server:

    Looking for relevant tables (This MUST be used first):
    search_file(query: str) - Find the relevant tables to the query by searching though RAG server documentation.

    Storing data into the rag server:
    index_pdf(pdf_path: str) - Index a PDF file into the rag server.
    index_local_files(directory: str) - Index all CSV, xlsx, JSON, and TXT files in a directory into the rag server.

    Searching for data in the rag server:
    search_local_files(query: str) - Find relevant CSV, xlsx, JSON, and TXT files in the rag server to obtain the path and schema
    find_table(query: str, data_source: str) - Find the path and schema of a table in a specific database. Data_source should contain the database name. If unsure, use "ragDatabase/sql/supply_chain.db".

    If search_file returns an xlsx file, ensure to use search_local_files. If it returns a table name with no suffix, use find_table.
    """,
    db=db,
    servers=["rag"],
    use_memory=False,
)

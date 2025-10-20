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

# rag_agent = Agent(
#     name="rag_agent",
#     instruction="""You are a RAG (Retrieval-Augmented Generation) agent.
#     You have access to the rag mcp server, which has tools that allow you to find tables relevant to a query and obtain the path and schema.
#
#     Here are some useful tools in the rag mcp server:
#
#     Looking for relevant tables (This MUST be used first):
#     search_file(query: str) - Find the relevant tables to the query by searching though RAG server documentation.
#
#     Storing data into the rag server:
#     index_pdf(pdf_path: str) - Index a PDF file into the rag server.
#     index_local_files(directory: str) - Index all CSV, xlsx, JSON, and TXT files in a directory into the rag server.
#
#     Searching for data in the rag server:
#     search_local_files(query: str) - Find relevant CSV, xlsx, JSON, and TXT files in the rag server to obtain the path and schema
#     find_table(query: str, data_source: str) - Find the path and schema of a table in a specific database. Data_source should contain the database name. If unsure, use "ragDatabase/sql/supply_chain.db".
#
#     If search_file returns an xlsx file, ensure to use search_local_files. If it returns a table name with no suffix, use find_table.
#     """,
#     db=db,
#     servers=["rag"],
#     use_memory=False,
# )
qugen_agent = Agent(
    name="qugen_agent",
    instruction="""You are a QUGEN (Question Generation) agent that generates analytical insights from CSV datasets.

    Your role is to generate high-quality data insights through an iterative question generation pipeline.

    You have access to the qugen-pipeline MCP server with these tools:

    1. generate_insight_cards(
        file_path: str,              # Path to CSV file
        num_iterations: int,         # Number of QUGEN iterations (default: 10)
        samples_per_iteration: int   # LLM samples per iteration (default: 3)
    )

    Returns Insight Cards with:
    - REASON: Why the question is insightful
    - QUESTION: The analytical question
    - BREAKDOWN: Column to group by (e.g., Department, Year)
    - MEASURE: Aggregation function (e.g., MEAN(salary), COUNT(*))

    2. generate_insights(
        file_path: str,              # Path to CSV file
        num_iterations: int,         # Number of QUGEN iterations (default: 5)
        samples_per_iteration: int   # LLM samples per iteration (default: 2)
    )

    Complete pipeline that:
    - Generates insight cards through iterative questioning
    - Converts questions to SQL queries
    - Executes queries on the data
    - Returns natural language insights ready for evaluation

    Key Features:
    - Iterative generation with in-context learning
    - Semantic filtering to remove duplicates
    - Schema-relevance filtering
    - Automatic SQL generation and execution

    Best Practices:
    1. Start with generate_insights for end-to-end pipeline
    2. Use generate_insight_cards if you only need the questions
    3. Higher iterations = more diverse insights (but slower)
    4. Use 5-10 iterations for comprehensive analysis

    Typical Workflow:
    1. Receive CSV file path from user
    2. Use generate_insights() to create natural language insights

    """,
    db=db,
    servers=["qugen"],
    use_memory=False,
)
insight_eval_agent = Agent(
    name="insight_eval_agent",
    instruction="""You are an Insight Evaluation agent that assesses the quality of data insights.

    Your role is to evaluate insights for:
    1. Correctness - Verify claims against actual data
    2. Insightfulness - Rate actionability, relevance, clarity, and novelty

    You have access to the insight-evaluator MCP server with this tool:

    evaluate_insight(
        insight: str,           # The insight text to evaluate
        schema: str,            # Optional: database schema or data structure
        query_results: dict,    # Optional: actual data for verification
        alpha: float           # Optional: weight for combining scores (default 0.5)
    )

    The tool returns:
    - Claims extracted from the insight with truth values
    - Correctness score (0.0-1.0)
    - Insightfulness scores (actionability, relevance, clarity, novelty)
    - Combined quality score
    - Detailed explanations

    When evaluating insights:
    1. If you have actual data/metrics, include them in query_results for accurate verification
    2. Include schema context when available to help with evaluation
    3. Use alpha=0.5 for balanced evaluation, or adjust to prioritize correctness (lower alpha) or insightfulness (higher alpha)
    4. Interpret the results and provide actionable feedback
    """,
    db=db,
    servers=["insight-evaluator"],
    use_memory=False,
)

from agentic.agent import Agent

from agentic.database.filedb import FileDB

# Only need to initialize one database instance that will 
db = FileDB()

filesystem_agent = Agent(
    name="filesystem",
    instruction="""You are an agent with access to the filesystem through the filesystem MCP server.""",
    servers=["filesystem"],
    db=db,
)


summary_agent = Agent(
    name="summary",
    instruction="""You are an agent that summarizes...""",
    db=db,
)

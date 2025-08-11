import asyncio
from agentic.orchestrator import Orchestrator
from USER.functions import create_python_file, execute_python_file
from USER.agents import rag_agent


if __name__ == "__main__":
    # query = "Index the file p2,3.pdf using the rag server"
    # query = "Which dataset in the rag server should be referred to when it comes to the total amount of purchases? " \
    # "Use search_file in the rag server to find the answer."
    query = "How many of the expedited orders were high priority?"
    # query = "Hello"


    orchestrator = Orchestrator(
        available_agents=[rag_agent],
        functions=[create_python_file, execute_python_file],
    )

    asyncio.run(orchestrator.orchestrate(query))

'''
Simple query examples:
    query = "Create a python file that contains a function to add two numbers and return the result."

    Playwright mcp needs a bit of help with the link, DOES NOT WORK WITH CLAUDE
    query = "What is the current S&P500 index value? The direct website is https://ca.finance.yahoo.com/quote/%5EGSPC/"

    More complex query example (Also buggy with Claude):
    query = """Make a graph of the revenue per month from the CSV file 'example_datasets/company1_sales_new.csv' and 
    save the graph as 'revenue_graph.png' in the current working directory."""
'''

import asyncio
from agentic.orchestrator import Orchestrator
from USER.functions import create_python_file, execute_python_file, obtain_csv_header
from USER.agents import playwright_agent


if __name__ == "__main__":
    
    orchestrator = Orchestrator(
        available_agents=[playwright_agent],
        functions=[create_python_file, execute_python_file, obtain_csv_header]
    )
    # Simple query examples:
    query = "Create a python file that contains a function to add two numbers and return the result."

    # Playwright mcp needs a bit of help with the link, DOES NOT WORK WITH CLAUDE
    # query = "What is the current S&P500 index value? The direct website is https://ca.finance.yahoo.com/quote/%5EGSPC/"

    # More complex query example (Also buggy with Claude):
    # query = """Make a graph of the revenue per month from the CSV file 'example_datasets/company1_sales_new.csv' and 
    # save the graph as 'revenue_graph.png' in the current working directory."""
    
    
    asyncio.run(orchestrator.orchestrate(query))
import asyncio
from agentic.orchestrator import Orchestrator
from USER.functions import create_python_file, execute_python_file
# from USER.agents import rag_agent
from USER.agents import insight_eval_agent
from USER.agents import qugen_agent
if __name__ == "__main__":
    # query = "Index the file p2,3.pdf using the rag server"
    # "Use search_file in the rag server to find the answer."
    #query = "How many of the expedited orders were high priority?"
    #query = "What is the current S&P500 index value? The direct website is https://ca.finance.yahoo.com/quote/%5EGSPC/"
    query = """
    Objective: Generate and evaluate insights from datasource.csv

    Step 1: Generate Insights
    Use qugen_agent to generate insights from datasource.csv.
    Tell the agent: "Generate insights from the file datasource.csv using 1 iteration and 2 samples per iteration."

    Step 2: Evaluate Insights  
    Once you have the insights from Step 1, use insight_eval_agent to evaluate each insight.

    Step 3: Present Results
    Show the top 3 insights with the highest combined scores with COMPLETE ORIGINAL TEXT.
    Display them in a ranked list with their scores.

    Do NOT save any files - just print the results directly.
    """

    # query = "Hello"


    orchestrator = Orchestrator(
        available_agents=[insight_eval_agent,qugen_agent],
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

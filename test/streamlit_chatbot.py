# To run: streamlit run streamlit_chatbot.py

import streamlit as st
import asyncio

from agentic.orchestrator import Orchestrator
from USER.functions import create_python_file, execute_python_file
from USER.agents import rag_agent

from agentic.database.models import Event, HumanInput, HumanInputResult
from agentic.database.filedb import FileDB
from agentic.database.serializer import thread_to_prompt


# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

st.title("🗨️ Agentic AI Orchestrator")

# Display existing chat messages
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

# Input box for user query
query = st.chat_input("Type your message...")

if query:
    # Append user message
    st.session_state.messages.append({'role': 'user', 'content': query})
    st.chat_message('user').write(query)

    # Build full context string for LLM
    context = "chat_history:\n"
    context += "\n".join(
        f"{m['role']}: {m['content']}" for m in st.session_state.messages
    )

    print(f"\n\n{context}\n\n")

    # Show thinking spinner
    with st.spinner('Thinking...'):
        orchestrator = Orchestrator(
            available_agents=[rag_agent],
            functions=[create_python_file, execute_python_file],
        )
        # Run orchestrator and get response
        response = asyncio.run(orchestrator.orchestrate(context))

    # Append and display planner response
    st.session_state.messages.append({'role': 'planner', 'content': response})
    st.chat_message('planner').write(response)



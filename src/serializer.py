import yaml
from models import Event, Thread
from dataclasses import asdict

# Public Method(s)
def thread_to_prompt(thread: Thread) -> str:
    """
    Serialize an entire Thread (list of Events) into one concise prompt.
    """
    return "\n\n".join(_event_to_prompt(evt) for evt in thread.events)

# Private Helper Function(s)
def _event_to_prompt(event: Event) -> str:
    """
    Serialize a single Event to a compact prompt string.
    """
  
    payload = yaml.safe_dump(
        asdict(event.data),
        default_flow_style=False,
        sort_keys=False
    ).strip()

    return f"<{event.type}>\n{payload}\n</{event.type}>"


#example usage
# if __name__ == "__main__":
#     # Example Event and Thread
#     example_event = Event(
#         type="request_agent",
#         data={"name": "Agent Smith", "instruction": "Find Neo"}
#     )
#     example_event2 = Event(
#         type="request_agent_result",
#         data={"answer": "Neo is the One"}
#     )

#     example_thread = Thread(events=[example_event, example_event2], id="orchestrator")
    
#     # Convert to prompt
#     prompt = thread_to_prompt(example_thread)
#     print(prompt)
#     print("--- End of Prompt ---")

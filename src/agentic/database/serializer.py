import yaml
from dataclasses import asdict

from agentic.database.models import Event, Thread

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

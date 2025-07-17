import os
import json
from typing import Optional
from dataclasses import asdict

import agentic
from agentic.database.models import Thread
from agentic.database.utils import deserialize_event

class FileDB:
    # use path relative to agentic module
    def __init__(self, base_dir: str = os.path.join(os.path.dirname(agentic.__file__), "database", "threads")):
        os.makedirs(base_dir, exist_ok=True)
        self.base_dir = base_dir

    def _path(self, thread_id: str) -> str:
        return os.path.join(self.base_dir, f"{thread_id}.json")

    async def save_thread(self, thread: Thread) -> None:
        """
        Serialize thread.events to a JSON file named <thread.id>.json.
        """
        payload = []
        for e in thread.events:
            #figure out what e.data is
            payload.append({"type": e.type, "data": asdict(e.data)})

        with open(self._path(thread.id), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    async def load_thread(self, thread_id: str) -> Optional[Thread]:
        """
        Read the JSON file back into a Thread object.
        Returns empty Thread if file does not exist.
        """
        p = self._path(thread_id)
        if not os.path.exists(p):
            return Thread(id=thread_id, events=[])

        with open(p, "r", encoding="utf-8") as f:
            raw_events = json.load(f)

        events = [deserialize_event(e) for e in raw_events]
        return Thread(id=thread_id, events=events)
    
    # To implement (to clear errors that have been resolved)!!!
    async def clear_last_n_events(self, thread_id: str, n: int) -> None:
        """
        Clear the last n events from the thread.
        """
        thread = await self.load_thread(thread_id)
        if not thread or n <= 0:
            return

        # Keep only the first len(events) - n events
        thread.events = thread.events[:-n] if len(thread.events) > n else []
        await self.save_thread(thread)
    
# Example usage
# if __name__ == "__main__":
    # db = FileDB()
    
    # # Create a sample thread
    # example_event = Event(
    #     type="request_agent",
    #     data={"name": "Agent Smith", "instruction": "Find Neo"}
    # )
    # example_event2 = Event(
    #     type="request_agent_result",
    #     data={"answer": "Neo is the One"}
    # )

    # example_thread = Thread(events=[example_event, example_event2], id="orchestrator")
    
    # # Save the thread
    # import asyncio
    # asyncio.run(db.save_thread(example_thread))
    
    # # Load the thread back
    # loaded_thread = asyncio.run(db.load_thread("orchestrator"))
    # print(loaded_thread)

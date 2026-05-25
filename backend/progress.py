"""
Simple in-memory progress tracker using asyncio.Queue.
Processing thread pushes updates, SSE endpoint streams them to client.
"""
import asyncio
import json
from typing import AsyncGenerator


class ProgressTracker:
    """Track progress of document processing tasks."""

    _instances: dict = {}

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.queue: asyncio.Queue = asyncio.Queue()
        self.current_state = {"stage": "pending", "progress": 0, "message": "Waiting..."}
        ProgressTracker._instances[task_id] = self

    @classmethod
    def get(cls, task_id: str) -> "ProgressTracker":
        if task_id not in cls._instances:
            cls._instances[task_id] = cls(task_id)
        return cls._instances[task_id]

    @classmethod
    def exists(cls, task_id: str) -> bool:
        return task_id in cls._instances

    def update(self, stage: str, progress: int, message: str):
        """Called from processing thread to push a progress update."""
        self.current_state = {"stage": stage, "progress": progress, "message": message}
        try:
            self.queue.put_nowait(self.current_state.copy())
        except asyncio.QueueFull:
            pass  # Skip if queue is full

    async def stream(self) -> AsyncGenerator[str, None]:
        """SSE generator - yields formatted events until 'done' stage."""
        while True:
            try:
                event = await asyncio.wait_for(self.queue.get(), timeout=30.0)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event["stage"] == "done" or event["stage"] == "error":
                    break
            except asyncio.TimeoutError:
                # Send keepalive
                yield f"data: {json.dumps({'stage': 'keepalive', 'progress': self.current_state['progress'], 'message': 'Processing...'})}\n\n"

        # Cleanup
        if self.task_id in ProgressTracker._instances:
            del ProgressTracker._instances[self.task_id]

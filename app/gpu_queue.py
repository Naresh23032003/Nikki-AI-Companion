"""ONE shared GPU job queue for all renders (studio TTS, covers, video lines).

Priorities: lower = sooner. Voice notes (0) beat covers (10). Jobs run one at
a time so the 6GB card never hosts two heavy models simultaneously.
"""
from __future__ import annotations

import asyncio
import itertools
import logging

logger = logging.getLogger("companion.gpuq")

PRIORITY_VOICE_NOTE = 0
PRIORITY_BENCH = 5
PRIORITY_COVER = 10

_counter = itertools.count()


class GPUJobQueue:
    def __init__(self):
        self._q: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._worker: asyncio.Task | None = None
        self.current: str | None = None

    def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run())

    async def submit(self, name: str, coro_factory, priority: int = 5):
        """Enqueue a job; returns an asyncio.Future with its result."""
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        await self._q.put((priority, next(_counter), name, coro_factory, fut))
        logger.info("gpu queue: +%s (prio %d, depth %d)", name, priority,
                    self._q.qsize())
        return fut

    async def _run(self) -> None:
        while True:
            priority, _, name, factory, fut = await self._q.get()
            self.current = name
            try:
                result = await factory()
                if not fut.cancelled():
                    fut.set_result(result)
            except Exception as e:  # noqa: BLE001
                logger.exception("gpu job %s failed", name)
                if not fut.cancelled():
                    fut.set_exception(e)
            finally:
                self.current = None
                self._q.task_done()

    def status(self) -> dict:
        return {"depth": self._q.qsize(), "current": self.current}

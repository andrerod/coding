from collections import deque
from dataclasses import dataclass, field
from heapq import heappush, heappop
from threading import Lock
from time import monotonic
from typing import Any, Callable
import unittest


@dataclass(frozen=True)
class Lease:
    task_id: int
    worker_id: str
    token: int
    attempt: int
    payload: Any


@dataclass
class Task:
    task_id: int
    payload: Any
    state: str = "READY"
    attempts: int = 0
    lease: Lease | None = None
    error: str | None = None


class WorkQueue:
    """
    In-memory, thread-safe, at-least-once work queue.

    max_attempts counts total deliveries, including the initial delivery.
    Expired or explicitly failed deliveries consume an attempt.
    """

    def __init__(
        self,
        lease_seconds: float = 30.0,
        max_attempts: int = 3,
        clock: Callable[[], float] = monotonic,
    ):
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        self.lease_seconds = lease_seconds
        self.max_attempts = max_attempts
        self.clock = clock

        self._lock = Lock()
        self._next_task_id = 1
        self._next_token = 1

        self._tasks: dict[int, Task] = {}
        self._ready: deque[int] = deque()

        # Entries: (expiration_time, token, task_id)
        self._expiry_heap: list[tuple[float, int, int]] = []

    def submit(self, payload: Any) -> int:
        with self._lock:
            task_id = self._next_task_id
            self._next_task_id += 1

            self._tasks[task_id] = Task(task_id, payload)
            self._ready.append(task_id)
            return task_id

    def take(self, worker_id: str) -> Lease | None:
        with self._lock:
            self._reap_expired_locked()

            while self._ready:
                task_id = self._ready.popleft()
                task = self._tasks[task_id]

                # Ignore obsolete queue entries.
                if task.state != "READY":
                    continue

                token = self._next_token
                self._next_token += 1
                task.attempts += 1

                lease = Lease(
                    task_id=task_id,
                    worker_id=worker_id,
                    token=token,
                    attempt=task.attempts,
                    payload=task.payload,
                )

                task.state = "LEASED"
                task.lease = lease

                heappush(
                    self._expiry_heap,
                    (
                        self.clock() + self.lease_seconds,
                        token,
                        task_id,
                    ),
                )
                return lease

            return None

    def ack(self, lease: Lease) -> bool:
        """Return False if this lease is stale or already completed."""
        with self._lock:
            self._reap_expired_locked()
            task = self._owned_task_locked(lease)

            if task is None:
                return False

            task.state = "DONE"
            task.lease = None
            return True

    def fail(self, lease: Lease, error: str) -> bool:
        """Record failure; retry or move to DEAD."""
        with self._lock:
            self._reap_expired_locked()
            task = self._owned_task_locked(lease)

            if task is None:
                return False

            self._retry_or_dead_locked(task, error)
            return True

    def reap_expired(self) -> int:
        """Recover all currently expired leases; return count recovered."""
        with self._lock:
            return self._reap_expired_locked()

    def get_status(self, task_id: int) -> dict:
        with self._lock:
            task = self._tasks[task_id]
            return {
                "task_id": task.task_id,
                "payload": task.payload,
                "state": task.state,
                "attempts": task.attempts,
                "error": task.error,
            }

    def dead_letters(self) -> list[dict]:
        with self._lock:
            return [
                self.get_status(task_id)
                for task_id, task in self._tasks.items()
                if task.state == "DEAD"
            ]

    def _owned_task_locked(self, lease: Lease) -> Task | None:
        task = self._tasks.get(lease.task_id)

        if (
            task is None
            or task.state != "LEASED"
            or task.lease != lease
        ):
            return None

        return task

    def _retry_or_dead_locked(self, task: Task, error: str) -> None:
        task.lease = None
        task.error = error

        if task.attempts >= self.max_attempts:
            task.state = "DEAD"
        else:
            task.state = "READY"
            self._ready.append(task.task_id)

    def _reap_expired_locked(self) -> int:
        now = self.clock()
        recovered = 0

        while self._expiry_heap and self._expiry_heap[0][0] <= now:
            _, token, task_id = heappop(self._expiry_heap)
            task = self._tasks[task_id]

            # Old heap entries cannot invalidate a newer lease.
            if (
                task.state != "LEASED"
                or task.lease is None
                or task.lease.token != token
            ):
                continue

            self._retry_or_dead_locked(
                task, "lease expired; worker presumed unavailable"
            )
            recovered += 1

        return recovered

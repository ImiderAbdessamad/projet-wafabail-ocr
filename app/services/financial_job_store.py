"""Stockage mémoire thread-safe des jobs d'analyse financière (TTL)."""
from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque

from app.config import DIRECT_FINANCIAL_JOB_TTL_MINUTES
from app.schemas.direct_financial_extraction import (
    FinancialDocumentAnalysisResult,
    FinancialJobProgress,
    JobStatus,
)


ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass
class FinancialJob:
    job_id: str
    status: JobStatus = "queued"
    progress_pct: int = 0
    current_step: str = "queued"
    current_page: int | None = None
    pages_total: int | None = None
    pages_financial: int = 0
    pages_skipped: int = 0
    pages_failed: int = 0
    message: str = ""
    error: str | None = None
    result: FinancialDocumentAnalysisResult | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    events: Deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=500))
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    pdf_bytes: bytes | None = None
    filename: str = "document.pdf"
    include_markdown: bool = False
    max_pages: int | None = None


class FinancialJobStore:
    """Abstraction mémoire remplaçable ultérieurement par Redis."""

    def __init__(self, *, ttl_minutes: int | None = None) -> None:
        self._ttl = (ttl_minutes or DIRECT_FINANCIAL_JOB_TTL_MINUTES) * 60
        self._jobs: dict[str, FinancialJob] = {}
        self._lock = threading.RLock()

    def create(
        self,
        *,
        pdf_bytes: bytes,
        filename: str,
        include_markdown: bool = False,
        max_pages: int | None = None,
    ) -> FinancialJob:
        self.cleanup()
        job_id = uuid.uuid4().hex
        job = FinancialJob(
            job_id=job_id,
            pdf_bytes=pdf_bytes,
            filename=filename,
            include_markdown=include_markdown,
            max_pages=max_pages,
            message="Job en file d'attente",
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> FinancialJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if time.time() - job.updated_at > self._ttl:
                del self._jobs[job_id]
                return None
            return job

    def update(self, job_id: str, **kwargs: Any) -> FinancialJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            job.updated_at = time.time()
            return job

    def emit(self, job_id: str, event_type: str, data: dict[str, Any] | None = None) -> None:
        event = {
            "event": event_type,
            "job_id": job_id,
            "ts": time.time(),
            **(data or {}),
        }
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.events.append(event)
            job.updated_at = time.time()
            subscribers = list(job.subscribers)
        for queue in subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def subscribe(self, job_id: str) -> asyncio.Queue | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            queue: asyncio.Queue = asyncio.Queue(maxsize=200)
            # Replay buffered events
            for event in list(job.events):
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    break
            job.subscribers.append(queue)
            return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if queue in job.subscribers:
                job.subscribers.remove(queue)

    def to_progress(self, job: FinancialJob) -> FinancialJobProgress:
        return FinancialJobProgress(
            job_id=job.job_id,
            status=job.status,
            progress_pct=job.progress_pct,
            current_step=job.current_step,
            current_page=job.current_page,
            pages_total=job.pages_total,
            pages_financial=job.pages_financial,
            pages_skipped=job.pages_skipped,
            pages_failed=job.pages_failed,
            message=job.message,
            error=job.error,
            stream_url=f"/api/v1/financial-documents/jobs/{job.job_id}/stream",
            result_url=f"/api/v1/financial-documents/jobs/{job.job_id}/result",
        )

    def cleanup(self) -> None:
        now = time.time()
        with self._lock:
            expired = [
                jid
                for jid, job in self._jobs.items()
                if now - job.updated_at > self._ttl
            ]
            for jid in expired:
                del self._jobs[jid]


# Singleton processus
job_store = FinancialJobStore()

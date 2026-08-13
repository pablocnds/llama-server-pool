from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import secrets
import signal
import socket
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .config import Settings
from .errors import (
    CapacityError,
    DuplicateModelError,
    ModelConflictError,
    ModelNotFoundError,
    PoolShuttingDownError,
    StartupError,
)
from .memory import MemoryReader, SystemMemory
from .models import (
    ModelStatus,
    ModelView,
    PoolStatsView,
    RegisterModelRequest,
    SystemMemoryView,
)

logger = logging.getLogger(__name__)

_FORBIDDEN_ARGUMENTS = {
    "-m",
    "--model",
    "--alias",
    "--host",
    "--port",
    "--api-key",
    "--api-key-file",
}
_FORBIDDEN_PREFIXES = tuple(f"{argument}=" for argument in _FORBIDDEN_ARGUMENTS)


@dataclass(slots=True)
class ModelRecord:
    id: str
    model_path: str
    args: tuple[str, ...]
    priority: int
    predicted_memory_bytes: int
    created_at: float = field(default_factory=time.time)
    status: ModelStatus = ModelStatus.REGISTERED
    active_requests: int = 0
    last_used_at: float | None = None
    process: asyncio.subprocess.Process | None = None
    internal_port: int | None = None
    api_key: str | None = None
    actual_memory_bytes: int | None = None
    last_error: str | None = None
    watcher: asyncio.Task[None] | None = None
    output_tasks: tuple[asyncio.Task[None], ...] = ()

    @property
    def identity(self) -> tuple[str, tuple[str, ...]]:
        return self.model_path, self.args


@dataclass(frozen=True, slots=True)
class RouteLease:
    manager: PoolManager
    model_id: str
    port: int
    api_key: str

    async def release(self) -> None:
        await self.manager.release_route(self.model_id)


class _StartupAttemptError(Exception):
    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class _StartInterrupted(Exception):
    pass


class PoolManager:
    def __init__(
        self,
        settings: Settings,
        *,
        memory_reader: MemoryReader | None = None,
    ) -> None:
        self.settings = settings
        self.memory = memory_reader or MemoryReader()
        self._records: dict[str, ModelRecord] = {}
        self._condition = asyncio.Condition()
        self._initialization_lock = asyncio.Lock()
        self._ports_in_use: set[int] = set()
        self._port_cursor = settings.internal_port_min
        self._monitor_task: asyncio.Task[None] | None = None
        self._closing = False
        self._health_client = httpx.AsyncClient(
            timeout=httpx.Timeout(1.0), trust_env=False
        )

    async def start(self) -> None:
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(), name="memory-monitor"
        )

    async def shutdown(self) -> None:
        async with self._condition:
            self._closing = True
            self._condition.notify_all()

        if self._monitor_task is not None:
            self._monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitor_task

        async with self._condition:
            records = [record for record in self._records.values() if record.process]
        await asyncio.gather(
            *(self._stop_record(record, "pool shutdown") for record in records),
            return_exceptions=True,
        )
        await self._health_client.aclose()

    async def register(self, request: RegisterModelRequest) -> ModelView:
        self._validate_args(request.args)
        try:
            model_path = Path(request.model_path).expanduser().resolve(strict=True)
        except OSError as exc:
            raise ValueError(
                f"model path cannot be accessed: {request.model_path}"
            ) from exc
        if not model_path.is_file():
            raise ValueError(f"model path is not a regular file: {model_path}")

        prediction = request.estimated_memory_bytes
        if prediction is None:
            prediction = (
                model_path.stat().st_size + self.settings.model_size_margin_bytes
            )
        identity = (str(model_path), tuple(request.args))

        async with self._condition:
            if self._closing:
                raise PoolShuttingDownError("the pool is shutting down")
            if request.id in self._records:
                raise ModelConflictError(
                    f"model ID {request.id!r} is already registered"
                )
            duplicate = next(
                (
                    record
                    for record in self._records.values()
                    if record.identity == identity
                ),
                None,
            )
            if duplicate is not None:
                raise DuplicateModelError(
                    f"model path and arguments are already registered as {duplicate.id!r}"
                )
            record = ModelRecord(
                id=request.id,
                model_path=str(model_path),
                args=tuple(request.args),
                priority=request.priority,
                predicted_memory_bytes=prediction,
            )
            self._records[record.id] = record
            self._condition.notify_all()

        logger.info("registered model %s from %s", record.id, record.model_path)
        logger.debug(
            "model %s memory prediction: model_size=%d margin_or_override_total=%d",
            record.id,
            model_path.stat().st_size,
            prediction,
        )
        if request.initialize:
            await self.start_model(record.id, force=request.force)
        return await self.get_model(record.id, refresh_memory=True)

    async def remove(self, model_id: str) -> None:
        record = await self._require_record(model_id)
        await self._stop_record(record, "explicit removal")
        async with self._condition:
            if self._records.get(model_id) is record:
                del self._records[model_id]
                self._condition.notify_all()
        logger.info("removed model registration %s", model_id)

    async def unload(self, model_id: str) -> ModelView:
        record = await self._require_record(model_id)
        await self._stop_record(record, "explicit unload")
        return await self.get_model(model_id)

    async def update_priority(self, model_id: str, priority: int) -> ModelView:
        async with self._condition:
            record = self._records.get(model_id)
            if record is None:
                raise ModelNotFoundError(f"model {model_id!r} is not registered")
            record.priority = priority
            self._condition.notify_all()
        return self._view(record)

    async def start_model(self, model_id: str, *, force: bool = False) -> ModelView:
        async with self._initialization_lock:
            while True:
                async with self._condition:
                    if self._closing:
                        raise PoolShuttingDownError("the pool is shutting down")
                    record = self._records.get(model_id)
                    if record is None:
                        raise ModelNotFoundError(
                            f"model {model_id!r} is not registered"
                        )
                    if (
                        record.status is ModelStatus.RUNNING
                        and record.process is not None
                    ):
                        return self._view(record)
                    if record.status is ModelStatus.STOPPING:
                        await self._condition.wait()
                        continue
                    record.status = ModelStatus.STARTING
                    record.last_error = None
                    self._condition.notify_all()
                    break

            logger.info(
                "initializing model %s (predicted memory %d bytes)",
                model_id,
                record.predicted_memory_bytes,
            )
            try:
                await self._ensure_capacity(record, force=force)
                await self._spawn_with_port_retries(record)
            except asyncio.CancelledError:
                await self._reset_failed_start(
                    record, "initialization cancelled", failed=False
                )
                raise
            except CapacityError as exc:
                await self._reset_failed_start(record, str(exc), failed=False)
                raise
            except _StartInterrupted as exc:
                await self._reset_failed_start(record, str(exc), failed=False)
                raise ModelConflictError(str(exc)) from exc
            except Exception as exc:
                await self._reset_failed_start(record, str(exc), failed=True)
                if isinstance(exc, StartupError):
                    raise
                raise StartupError(
                    f"could not initialize model {model_id!r}: {exc}"
                ) from exc

            return await self.get_model(model_id, refresh_memory=True)

    async def acquire_route(self, model_id: str) -> RouteLease:
        while True:
            await self.start_model(model_id)
            async with self._condition:
                record = self._records.get(model_id)
                if record is None:
                    raise ModelNotFoundError(f"model {model_id!r} is not registered")
                if (
                    record.status is ModelStatus.RUNNING
                    and record.process is not None
                    and record.internal_port is not None
                    and record.api_key is not None
                ):
                    record.active_requests += 1
                    return RouteLease(
                        manager=self,
                        model_id=model_id,
                        port=record.internal_port,
                        api_key=record.api_key,
                    )

    async def release_route(self, model_id: str) -> None:
        async with self._condition:
            record = self._records.get(model_id)
            if record is not None:
                record.active_requests = max(0, record.active_requests - 1)
                record.last_used_at = time.time()
            self._condition.notify_all()

    async def get_model(
        self, model_id: str, *, refresh_memory: bool = False
    ) -> ModelView:
        record = await self._require_record(model_id)
        if refresh_memory:
            await self._refresh_record_memory(record)
        async with self._condition:
            return self._view(record)

    async def list_models(self, *, refresh_memory: bool = False) -> list[ModelView]:
        async with self._condition:
            records = list(self._records.values())
        if refresh_memory:
            await asyncio.gather(*(self._refresh_record_memory(r) for r in records))
        async with self._condition:
            return [self._view(record) for record in self._records.values()]

    async def stats(self) -> PoolStatsView:
        system, usage = await asyncio.gather(self.memory.system(), self._pool_usage())
        async with self._condition:
            records = list(self._records.values())
            views = [self._view(record) for record in records]
        return PoolStatsView(
            system=SystemMemoryView(
                total_bytes=system.total_bytes,
                available_bytes=system.available_bytes,
                used_bytes=system.used_bytes,
                free_bytes=(
                    system.free_bytes
                    if system.free_bytes is not None
                    else system.available_bytes
                ),
                outside_pool_resident_bytes=max(
                    0,
                    system.total_bytes
                    - (
                        system.free_bytes
                        if system.free_bytes is not None
                        else system.available_bytes
                    )
                    - usage,
                ),
                normal_headroom_bytes=self.settings.normal_headroom_bytes,
                critical_headroom_bytes=self.settings.critical_headroom_bytes,
            ),
            pool_usage_bytes=usage,
            pool_budget_bytes=self.settings.pool_memory_budget_bytes,
            registered_models=len(records),
            running_models=sum(r.status is ModelStatus.RUNNING for r in records),
            starting_models=sum(r.status is ModelStatus.STARTING for r in records),
            active_requests=sum(r.active_requests for r in records),
            processes=views,
        )

    async def _ensure_capacity(self, target: ModelRecord, *, force: bool) -> None:
        prediction = target.predicted_memory_bytes
        system = await self.memory.system()
        if prediction + self.settings.normal_headroom_bytes > system.total_bytes:
            raise CapacityError(
                "the predicted model size cannot fit while preserving system headroom"
            )
        budget = self.settings.pool_memory_budget_bytes
        if budget is not None and prediction > budget:
            raise CapacityError(
                "the predicted model size exceeds the pool memory budget"
            )

        logged_wait = False
        while True:
            if self._closing:
                raise PoolShuttingDownError("the pool is shutting down")
            async with self._condition:
                if (
                    self._records.get(target.id) is not target
                    or target.status is not ModelStatus.STARTING
                ):
                    raise _StartInterrupted(
                        f"initialization of model {target.id!r} was interrupted"
                    )
            system, usage = await asyncio.gather(
                self.memory.system(), self._pool_usage()
            )
            if self._has_capacity(system, usage, prediction):
                return

            victim = await self._choose_victim(
                exclude_id=target.id,
                include_active=force,
                include_starting=force,
            )
            if victim is not None:
                await self._stop_record(
                    victim,
                    f"capacity for model {target.id}" + (" (forced)" if force else ""),
                )
                continue
            if force:
                raise CapacityError(
                    "no process can be evicted to satisfy memory limits"
                )

            if not logged_wait:
                logger.info(
                    "model %s is waiting for an idle eviction candidate", target.id
                )
                logged_wait = True
            async with self._condition:
                try:
                    await asyncio.wait_for(
                        self._condition.wait(),
                        timeout=self.settings.monitor_interval_seconds,
                    )
                except TimeoutError:
                    pass

    def _has_capacity(
        self, system: SystemMemory, pool_usage: int, prediction: int
    ) -> bool:
        if system.available_bytes - prediction < self.settings.normal_headroom_bytes:
            return False
        budget = self.settings.pool_memory_budget_bytes
        return budget is None or pool_usage + prediction <= budget

    async def _choose_victim(
        self,
        *,
        exclude_id: str | None = None,
        include_active: bool,
        include_starting: bool,
    ) -> ModelRecord | None:
        async with self._condition:
            candidates: list[ModelRecord] = []
            for record in self._records.values():
                if record.id == exclude_id or record.process is None:
                    continue
                if record.status is ModelStatus.STARTING:
                    if include_starting:
                        candidates.append(record)
                elif record.status is ModelStatus.RUNNING and (
                    record.active_requests == 0 or include_active
                ):
                    candidates.append(record)

            def eviction_key(record: ModelRecord) -> tuple[int, int, float]:
                if record.status is ModelStatus.RUNNING and record.active_requests == 0:
                    activity_rank = 0
                elif record.status is ModelStatus.STARTING:
                    activity_rank = 1
                else:
                    activity_rank = 2
                return (
                    record.priority,
                    activity_rank,
                    record.last_used_at if record.last_used_at is not None else 0.0,
                )

            return min(candidates, key=eviction_key, default=None)

    async def _spawn_with_port_retries(self, record: ModelRecord) -> None:
        attempts = self.settings.internal_port_max - self.settings.internal_port_min + 1
        last_error = "no internal port was available"
        for _ in range(attempts):
            port = await self._allocate_port()
            if port is None:
                break
            try:
                await self._spawn_attempt(record, port)
                return
            except _StartupAttemptError as exc:
                last_error = str(exc)
                await self._cleanup_failed_attempt(record)
                if not exc.retryable:
                    break
                logger.warning(
                    "model %s could not bind internal port %d; retrying: %s",
                    record.id,
                    port,
                    exc,
                )
        raise StartupError(f"could not initialize model {record.id!r}: {last_error}")

    async def _spawn_attempt(self, record: ModelRecord, port: int) -> None:
        async with self._condition:
            if (
                self._records.get(record.id) is not record
                or record.status is not ModelStatus.STARTING
            ):
                self._ports_in_use.discard(port)
                raise _StartInterrupted(
                    f"initialization of model {record.id!r} was interrupted"
                )
        # The prefix prevents a leading '-' from being interpreted as another
        # command-line option by llama-server's argument parser.
        api_key = f"pool_{secrets.token_urlsafe(32)}"
        command = [
            self.settings.llama_server_executable,
            "--model",
            record.model_path,
            "--alias",
            record.id,
            *record.args,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--api-key",
            api_key,
        ]
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            async with self._condition:
                self._ports_in_use.discard(port)
            raise _StartupAttemptError(str(exc)) from exc

        stderr_tail: deque[str] = deque(maxlen=30)
        output_tasks = (
            asyncio.create_task(
                self._drain_output(record.id, "stdout", process.stdout),
                name=f"{record.id}-stdout",
            ),
            asyncio.create_task(
                self._drain_output(record.id, "stderr", process.stderr, stderr_tail),
                name=f"{record.id}-stderr",
            ),
        )
        async with self._condition:
            interrupted = (
                self._records.get(record.id) is not record
                or record.status is not ModelStatus.STARTING
            )
            if interrupted:
                self._ports_in_use.discard(port)
            else:
                record.process = process
                record.internal_port = port
                record.api_key = api_key
                record.output_tasks = output_tasks
                record.watcher = asyncio.create_task(
                    self._watch_process(record, process, port),
                    name=f"{record.id}-watcher",
                )
                self._condition.notify_all()
        if interrupted:
            await self._terminate_process(process)
            await asyncio.gather(*output_tasks, return_exceptions=True)
            raise _StartInterrupted(
                f"initialization of model {record.id!r} was interrupted"
            )

        deadline = (
            asyncio.get_running_loop().time() + self.settings.startup_timeout_seconds
        )
        url = f"http://127.0.0.1:{port}/health"
        while asyncio.get_running_loop().time() < deadline:
            if process.returncode is not None:
                stderr = "\n".join(stderr_tail)
                retryable = any(
                    phrase in stderr.lower()
                    for phrase in (
                        "address already in use",
                        "failed to bind",
                        "bind failed",
                    )
                )
                detail = (
                    stderr.strip() or f"process exited with code {process.returncode}"
                )
                raise _StartupAttemptError(detail, retryable=retryable)
            async with self._condition:
                if record.status is ModelStatus.STOPPING:
                    raise _StartupAttemptError("startup was interrupted by eviction")
            try:
                response = await self._health_client.get(
                    url, headers={"Authorization": f"Bearer {api_key}"}
                )
                if response.status_code == 200:
                    async with self._condition:
                        if not (
                            self._records.get(record.id) is record
                            and record.process is process
                            and record.status is ModelStatus.STARTING
                        ):
                            raise _StartupAttemptError(
                                "startup was interrupted by an unload or eviction"
                            )
                        record.status = ModelStatus.RUNNING
                        record.last_error = None
                        self._condition.notify_all()
                    actual = await self.memory.process(process.pid)
                    async with self._condition:
                        if record.process is process:
                            record.actual_memory_bytes = actual
                    logger.info(
                        "initialized model %s as pid %d on internal port %d",
                        record.id,
                        process.pid,
                        port,
                    )
                    logger.debug(
                        "model %s memory after initialization: predicted=%d actual_pss_or_rss=%d",
                        record.id,
                        record.predicted_memory_bytes,
                        actual,
                    )
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.1)

        raise _StartupAttemptError(
            f"health check did not become ready within "
            f"{self.settings.startup_timeout_seconds:g} seconds"
        )

    async def _cleanup_failed_attempt(self, record: ModelRecord) -> None:
        await self._stop_record(record, "failed startup")
        async with self._condition:
            if self._records.get(record.id) is record:
                record.status = ModelStatus.STARTING
                self._condition.notify_all()

    async def _reset_failed_start(
        self, record: ModelRecord, error: str, *, failed: bool
    ) -> None:
        if record.process is not None:
            await self._stop_record(record, "initialization failure")
        async with self._condition:
            if self._records.get(record.id) is record:
                record.status = ModelStatus.FAILED if failed else ModelStatus.REGISTERED
                record.last_error = error
                self._condition.notify_all()

    async def _stop_record(self, record: ModelRecord, reason: str) -> None:
        async with self._condition:
            if self._records.get(record.id) is not record:
                return
            process = record.process
            if process is None:
                if record.status is not ModelStatus.REGISTERED:
                    record.status = ModelStatus.REGISTERED
                    self._condition.notify_all()
                return
            if record.status is not ModelStatus.STOPPING:
                record.status = ModelStatus.STOPPING
                self._condition.notify_all()
            port = record.internal_port

        logger.info("stopping model %s (pid %d): %s", record.id, process.pid, reason)
        await self._terminate_process(process)

        async with self._condition:
            if record.process is process:
                record.process = None
                record.internal_port = None
                record.api_key = None
                record.watcher = None
                record.output_tasks = ()
                record.actual_memory_bytes = None
                record.status = ModelStatus.REGISTERED
            if port is not None:
                self._ports_in_use.discard(port)
            self._condition.notify_all()

    async def _terminate_process(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(
                asyncio.shield(process.wait()),
                timeout=self.settings.shutdown_timeout_seconds,
            )
        except TimeoutError:
            logger.warning(
                "process group %d did not stop; sending SIGKILL", process.pid
            )
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()

    async def _watch_process(
        self, record: ModelRecord, process: asyncio.subprocess.Process, port: int
    ) -> None:
        return_code = await process.wait()
        async with self._condition:
            if record.process is not process:
                return
            if record.status is ModelStatus.STOPPING:
                self._condition.notify_all()
                return
            record.process = None
            record.internal_port = None
            record.api_key = None
            record.actual_memory_bytes = None
            self._ports_in_use.discard(port)
            if self._closing:
                record.status = ModelStatus.REGISTERED
            else:
                record.status = ModelStatus.FAILED
                record.last_error = (
                    f"llama-server exited unexpectedly with code {return_code}"
                )
                logger.error("model %s %s", record.id, record.last_error)
            self._condition.notify_all()

    async def _drain_output(
        self,
        model_id: str,
        stream_name: str,
        stream: asyncio.StreamReader | None,
        tail: deque[str] | None = None,
    ) -> None:
        if stream is None:
            return
        while line := await stream.readline():
            text = line.decode(errors="replace").rstrip()
            if tail is not None:
                tail.append(text)
            logger.debug("llama-server[%s] %s: %s", model_id, stream_name, text)

    async def _monitor_loop(self) -> None:
        while True:
            try:
                await self._monitor_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("memory monitor iteration failed")
            await asyncio.sleep(self.settings.monitor_interval_seconds)

    async def _monitor_once(self) -> None:
        system, usage = await asyncio.gather(self.memory.system(), self._pool_usage())
        budget = self.settings.pool_memory_budget_bytes
        critical = system.available_bytes < self.settings.critical_headroom_bytes
        pressured = system.available_bytes < self.settings.normal_headroom_bytes
        over_budget = budget is not None and usage > budget
        if not (pressured or over_budget):
            return

        if critical:
            logger.error(
                "critical memory pressure: %d bytes available; active processes may be evicted",
                system.available_bytes,
            )
        else:
            logger.warning(
                "memory threshold exceeded: available=%d pool_usage=%d",
                system.available_bytes,
                usage,
            )

        allow_busy = critical
        while pressured or over_budget:
            victim = await self._choose_victim(
                include_active=allow_busy,
                include_starting=allow_busy,
            )
            if victim is None:
                logger.warning(
                    "memory pressure remains but no eligible process can be evicted"
                )
                return
            await self._stop_record(victim, "memory pressure")
            system, usage = await asyncio.gather(
                self.memory.system(), self._pool_usage()
            )
            pressured = system.available_bytes < self.settings.normal_headroom_bytes
            over_budget = budget is not None and usage > budget

    async def _pool_usage(self) -> int:
        async with self._condition:
            process_records = [
                (record, record.process)
                for record in self._records.values()
                if record.process is not None
            ]
        usages = await asyncio.gather(
            *(self.memory.process(process.pid) for _, process in process_records)
        )
        total = 0
        async with self._condition:
            for (record, process), usage in zip(process_records, usages, strict=True):
                if record.process is process:
                    record.actual_memory_bytes = usage
                    total += usage
        return total

    async def _refresh_record_memory(self, record: ModelRecord) -> None:
        async with self._condition:
            process = record.process
        if process is None:
            return
        usage = await self.memory.process(process.pid)
        async with self._condition:
            if record.process is process:
                record.actual_memory_bytes = usage

    async def _allocate_port(self) -> int | None:
        count = self.settings.internal_port_max - self.settings.internal_port_min + 1
        for _ in range(count):
            async with self._condition:
                port = self._port_cursor
                self._port_cursor += 1
                if self._port_cursor > self.settings.internal_port_max:
                    self._port_cursor = self.settings.internal_port_min
                if port in self._ports_in_use:
                    continue
            if not self._port_is_available(port):
                continue
            async with self._condition:
                if port not in self._ports_in_use:
                    self._ports_in_use.add(port)
                    return port
        return None

    @staticmethod
    def _port_is_available(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                return False
        return True

    async def _require_record(self, model_id: str) -> ModelRecord:
        async with self._condition:
            record = self._records.get(model_id)
        if record is None:
            raise ModelNotFoundError(f"model {model_id!r} is not registered")
        return record

    @staticmethod
    def _validate_args(args: list[str]) -> None:
        for argument in args:
            if "\x00" in argument:
                raise ValueError("llama-server arguments cannot contain NUL bytes")
            if argument in _FORBIDDEN_ARGUMENTS or argument.startswith(
                _FORBIDDEN_PREFIXES
            ):
                raise ValueError(
                    f"argument {argument!r} is controlled by the pool and cannot be set"
                )

    @staticmethod
    def _view(record: ModelRecord) -> ModelView:
        return ModelView(
            id=record.id,
            model_path=record.model_path,
            args=list(record.args),
            priority=record.priority,
            status=record.status,
            active_requests=record.active_requests,
            created_at=record.created_at,
            last_used_at=record.last_used_at,
            pid=record.process.pid if record.process is not None else None,
            internal_port=record.internal_port,
            predicted_memory_bytes=record.predicted_memory_bytes,
            actual_memory_bytes=record.actual_memory_bytes,
            last_error=record.last_error,
        )

from __future__ import annotations

import asyncio
import socket
from dataclasses import replace
from pathlib import Path

import pytest

from llama_server_pool.config import Settings
from llama_server_pool.errors import ModelConflictError
from llama_server_pool.manager import PoolManager
from llama_server_pool.memory import SystemMemory
from llama_server_pool.models import RegisterModelRequest


class FixedMemoryReader:
    def __init__(self, *, available: int = 10_000, process_usage: int = 100) -> None:
        self.available = available
        self.process_usage = process_usage

    async def system(self) -> SystemMemory:
        return SystemMemory(
            total_bytes=10_000,
            available_bytes=self.available,
            used_bytes=10_000 - self.available,
        )

    async def process(self, _pid: int) -> int:
        return self.process_usage


def manager_settings(*, budget: int | None = None) -> Settings:
    executable = Path(__file__).with_name("fake_llama_server.py")
    return Settings(
        llama_server_executable=str(executable),
        internal_port_min=19_200,
        internal_port_max=19_299,
        normal_headroom_bytes=0,
        critical_headroom_bytes=0,
        pool_memory_budget_bytes=budget,
        model_size_margin_bytes=0,
        monitor_interval_seconds=60,
        startup_timeout_seconds=5,
        shutdown_timeout_seconds=1,
    )


@pytest.mark.asyncio
async def test_budget_evicts_idle_low_priority_model(model_file: Path) -> None:
    manager = PoolManager(
        manager_settings(budget=150), memory_reader=FixedMemoryReader()
    )
    await manager.start()
    try:
        await manager.register(
            RegisterModelRequest(
                id="low",
                model_path=str(model_file),
                priority=0,
                estimated_memory_bytes=100,
            )
        )
        await manager.register(
            RegisterModelRequest(
                id="high",
                model_path=str(model_file),
                args=["--ctx-size", "2"],
                priority=10,
                estimated_memory_bytes=100,
            )
        )
        await manager.start_model("low")
        await manager.start_model("high")

        assert (await manager.get_model("low")).status == "registered"
        assert (await manager.get_model("high")).status == "running"
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_force_can_evict_an_active_model(model_file: Path) -> None:
    manager = PoolManager(
        manager_settings(budget=150), memory_reader=FixedMemoryReader()
    )
    await manager.start()
    try:
        await manager.register(
            RegisterModelRequest(
                id="active", model_path=str(model_file), estimated_memory_bytes=100
            )
        )
        await manager.register(
            RegisterModelRequest(
                id="forced",
                model_path=str(model_file),
                args=["--ctx-size", "2"],
                estimated_memory_bytes=100,
            )
        )
        lease = await manager.acquire_route("active")
        await manager.start_model("forced", force=True)

        assert (await manager.get_model("active")).status == "registered"
        assert (await manager.get_model("forced")).status == "running"
        await lease.release()
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_unload_interrupts_a_start_waiting_for_an_active_model(
    model_file: Path,
) -> None:
    manager = PoolManager(
        manager_settings(budget=150), memory_reader=FixedMemoryReader()
    )
    await manager.start()
    try:
        await manager.register(
            RegisterModelRequest(
                id="active", model_path=str(model_file), estimated_memory_bytes=100
            )
        )
        await manager.register(
            RegisterModelRequest(
                id="waiting",
                model_path=str(model_file),
                args=["--ctx-size", "2"],
                estimated_memory_bytes=100,
            )
        )
        lease = await manager.acquire_route("active")
        start = asyncio.create_task(manager.start_model("waiting"))
        for _ in range(100):
            if (await manager.get_model("waiting")).status == "starting":
                break
            await asyncio.sleep(0.01)

        await manager.unload("waiting")
        with pytest.raises(ModelConflictError):
            await start
        waiting = await manager.get_model("waiting")
        assert waiting.status == "registered"
        assert waiting.pid is None
        await lease.release()
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_critical_pressure_can_evict_an_active_model(model_file: Path) -> None:
    memory = FixedMemoryReader()
    settings = replace(
        manager_settings(),
        normal_headroom_bytes=500,
        critical_headroom_bytes=100,
    )
    manager = PoolManager(settings, memory_reader=memory)
    await manager.start()
    try:
        await manager.register(
            RegisterModelRequest(
                id="active", model_path=str(model_file), estimated_memory_bytes=100
            )
        )
        lease = await manager.acquire_route("active")
        memory.available = 50
        await manager._monitor_once()

        assert (await manager.get_model("active")).status == "registered"
        await lease.release()
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_an_occupied_internal_port_is_skipped(model_file: Path) -> None:
    occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    occupied.bind(("127.0.0.1", 0))
    occupied.listen()
    port = occupied.getsockname()[1]
    if port == 65_535:
        occupied.close()
        pytest.skip("no following TCP port is available for this test")

    settings = replace(
        manager_settings(), internal_port_min=port, internal_port_max=port + 1
    )
    manager = PoolManager(settings, memory_reader=FixedMemoryReader())
    await manager.start()
    try:
        await manager.register(
            RegisterModelRequest(id="model", model_path=str(model_file))
        )
        started = await manager.start_model("model")
        assert started.internal_port == port + 1
    finally:
        await manager.shutdown()
        occupied.close()


@pytest.mark.asyncio
async def test_generated_api_key_cannot_be_parsed_as_an_option(
    model_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "llama_server_pool.manager.secrets.token_urlsafe", lambda _size: "-leading"
    )
    manager = PoolManager(manager_settings(), memory_reader=FixedMemoryReader())
    await manager.start()
    try:
        await manager.register(
            RegisterModelRequest(id="model", model_path=str(model_file))
        )
        lease = await manager.acquire_route("model")
        assert lease.api_key == "pool_-leading"
        await lease.release()
    finally:
        await manager.shutdown()

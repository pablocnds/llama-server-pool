from __future__ import annotations

import asyncio
import json
import socket
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from llama_server_pool.config import Settings
from llama_server_pool.errors import ModelConflictError, ProfileStoreError, StartupError
from llama_server_pool.manager import PoolManager
from llama_server_pool.memory import ProcessMemory, SystemMemory
from llama_server_pool.models import RegisterModelRequest


class FixedMemoryReader:
    def __init__(
        self,
        *,
        available: int = 10_000,
        free: int | None = None,
        cached: int = 0,
        process_usage: int = 100,
        process_file_usage: int = 0,
        drm_system_usage: int = 0,
        drm_vram_usage: int = 0,
    ) -> None:
        self.available = available
        self.free = free
        self.cached = cached
        self.process_usage = process_usage
        self.process_file_usage = process_file_usage
        self.drm_system_usage = drm_system_usage
        self.drm_vram_usage = drm_vram_usage

    async def system(self) -> SystemMemory:
        return SystemMemory(
            total_bytes=10_000,
            available_bytes=self.available,
            used_bytes=10_000 - self.available,
            free_bytes=self.free,
            cached_bytes=self.cached,
        )

    async def process(self, _pid: int) -> ProcessMemory:
        return ProcessMemory(
            process_bytes=self.process_usage,
            process_anon_bytes=self.process_usage - self.process_file_usage,
            process_file_bytes=self.process_file_usage,
            drm_system_bytes=self.drm_system_usage,
            drm_vram_bytes=self.drm_vram_usage,
        )


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
async def test_running_route_bypasses_unrelated_capacity_wait(model_file: Path) -> None:
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
        first_lease = await manager.acquire_route("active")
        waiting_start = asyncio.create_task(manager.start_model("waiting"))
        for _ in range(100):
            if (await manager.get_model("waiting")).status == "starting":
                break
            await asyncio.sleep(0.01)

        started = await asyncio.wait_for(manager.start_model("active"), timeout=0.5)
        second_lease = await asyncio.wait_for(
            manager.acquire_route("active"), timeout=0.5
        )
        assert started.status == "running"
        assert (await manager.get_model("active")).active_requests == 2

        waiting_start.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting_start
        await second_lease.release()
        await first_lease.release()
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_eviction_reservation_prevents_a_new_route(model_file: Path) -> None:
    manager = PoolManager(manager_settings(), memory_reader=FixedMemoryReader())
    await manager.start()
    try:
        await manager.register(
            RegisterModelRequest(id="model", model_path=str(model_file))
        )
        await manager.start_model("model")

        victim = await manager._reserve_victim(
            include_active=False, include_starting=False
        )

        assert victim is not None
        assert victim.status == "stopping"
        assert await manager._lease_running_route("model") is None
        await manager._stop_record(victim, "test cleanup")
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_reserved_eviction_finishes_after_caller_cancellation(
    model_file: Path,
) -> None:
    manager = PoolManager(manager_settings(), memory_reader=FixedMemoryReader())
    await manager.start()
    try:
        await manager.register(
            RegisterModelRequest(id="model", model_path=str(model_file))
        )
        await manager.start_model("model")
        victim = await manager._reserve_victim(
            include_active=False, include_starting=False
        )
        assert victim is not None
        original_terminate = manager._terminate_process
        terminate_started = asyncio.Event()
        allow_termination = asyncio.Event()

        async def delayed_termination(process):
            terminate_started.set()
            await allow_termination.wait()
            await original_terminate(process)

        manager._terminate_process = delayed_termination
        stop = asyncio.create_task(manager._stop_record(victim, "test cancellation"))
        await terminate_started.wait()
        stop.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stop
        allow_termination.set()
        assert victim.stop_task is not None
        await victim.stop_task

        assert (await manager.get_model("model")).status == "registered"
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_stale_lease_does_not_release_reused_model_id(model_file: Path) -> None:
    manager = PoolManager(manager_settings(), memory_reader=FixedMemoryReader())
    await manager.start()
    try:
        await manager.register(
            RegisterModelRequest(id="reused", model_path=str(model_file))
        )
        stale_lease = await manager.acquire_route("reused")
        await manager.remove("reused")
        await manager.register(
            RegisterModelRequest(
                id="reused", model_path=str(model_file), args=["--ctx-size", "2"]
            )
        )
        current_lease = await manager.acquire_route("reused")

        await stale_lease.release()

        assert (await manager.get_model("reused")).active_requests == 1
        await current_lease.release()
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
async def test_active_eviction_stops_when_pressure_is_no_longer_critical(
    model_file: Path,
) -> None:
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
                id="first", model_path=str(model_file), estimated_memory_bytes=100
            )
        )
        await manager.register(
            RegisterModelRequest(
                id="second",
                model_path=str(model_file),
                args=["--ctx-size", "2"],
                estimated_memory_bytes=100,
            )
        )
        first_lease = await manager.acquire_route("first")
        second_lease = await manager.acquire_route("second")
        original_stop = manager._stop_record
        stop_count = 0

        async def stop_and_relieve_pressure(record, reason):
            nonlocal stop_count
            await original_stop(record, reason)
            stop_count += 1
            memory.available = 200

        manager._stop_record = stop_and_relieve_pressure
        memory.available = 50

        await manager._monitor_once()

        assert stop_count == 1
        assert (
            sum(model.status == "running" for model in await manager.list_models()) == 1
        )
        await first_lease.release()
        await second_lease.release()
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


@pytest.mark.parametrize(
    "argument",
    ["-a", "--alias", "--api_key", "--api_key=known", "--api-key-file"],
)
def test_all_pool_owned_argument_spellings_are_rejected(argument: str) -> None:
    with pytest.raises(ValueError, match="controlled by the pool"):
        PoolManager._validate_args([argument])


@pytest.mark.asyncio
async def test_controlled_llama_environment_is_not_inherited(
    model_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured_environment = None

    async def reject_spawn(*_command, **options):
        nonlocal captured_environment
        captured_environment = options["env"]
        raise OSError("test spawn")

    for name in (
        "LLAMA_ARG_ALIAS",
        "LLAMA_ARG_API_KEY_FILE",
        "LLAMA_ARG_HOST",
        "LLAMA_ARG_MODEL",
        "LLAMA_ARG_PORT",
        "LLAMA_API_KEY",
    ):
        monkeypatch.setenv(name, "untrusted")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", reject_spawn)
    manager = PoolManager(manager_settings(), memory_reader=FixedMemoryReader())
    await manager.start()
    try:
        await manager.register(
            RegisterModelRequest(id="model", model_path=str(model_file))
        )
        with pytest.raises(StartupError, match="test spawn"):
            await manager.start_model("model")
        assert captured_environment is not None
        assert (
            not {
                "LLAMA_ARG_ALIAS",
                "LLAMA_ARG_API_KEY_FILE",
                "LLAMA_ARG_HOST",
                "LLAMA_ARG_MODEL",
                "LLAMA_ARG_PORT",
                "LLAMA_API_KEY",
            }
            & captured_environment.keys()
        )
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_termination_cleans_surviving_process_group(tmp_path: Path) -> None:
    child_pid_file = tmp_path / "child.pid"
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        (
            "import subprocess, sys; "
            "child = subprocess.Popen([sys.executable, '-c', "
            "'import time; time.sleep(60)']); "
            "open(sys.argv[1], 'w').write(str(child.pid))"
        ),
        str(child_pid_file),
        start_new_session=True,
    )
    await process.wait()
    child_pid = int(child_pid_file.read_text())
    manager = PoolManager(manager_settings(), memory_reader=FixedMemoryReader())

    await manager._terminate_process(process)

    try:
        stat = Path(f"/proc/{child_pid}/stat").read_text()
    except FileNotFoundError:
        return
    assert stat.split()[2] == "Z"


@pytest.mark.asyncio
async def test_stats_charge_drm_system_memory_and_separate_cache(
    model_file: Path,
) -> None:
    memory = FixedMemoryReader(
        available=7_000,
        free=4_000,
        cached=3_000,
        process_usage=1_000,
        process_file_usage=400,
        drm_system_usage=2_000,
        drm_vram_usage=500,
    )
    manager = PoolManager(manager_settings(), memory_reader=memory)
    await manager.start()
    try:
        await manager.register(
            RegisterModelRequest(id="model", model_path=str(model_file))
        )
        await manager.start_model("model")
        stats = await manager.stats()
        model = stats.processes[0]

        assert model.actual_memory_bytes == 3_000
        assert model.process_memory_bytes == 1_000
        assert model.process_file_memory_bytes == 400
        assert model.gpu_shared_memory_bytes == 2_000
        assert model.gpu_dedicated_memory_bytes == 500
        assert stats.pool_usage_bytes == 3_000
        assert stats.system.outside_pool_cache_bytes == 2_600
        assert stats.system.outside_pool_resident_bytes == 400
        assert (
            stats.pool_usage_bytes
            + stats.system.outside_pool_cache_bytes
            + stats.system.outside_pool_resident_bytes
            + stats.system.free_bytes
            == stats.system.total_bytes
        )
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_profiles_persist_across_manager_instances(
    model_file: Path, tmp_path: Path
) -> None:
    profiles_file = tmp_path / "config" / "profiles.json"
    settings = replace(manager_settings(), profiles_file=str(profiles_file))
    first = PoolManager(settings, memory_reader=FixedMemoryReader())
    await first.start()
    try:
        await first.register(
            RegisterModelRequest(
                id="persistent",
                model_path=str(model_file),
                args=["--ctx-size", "4096"],
                priority=3,
                estimated_memory_bytes=321,
            )
        )
        await first.update_priority("persistent", 7)
        await first.start_model("persistent")
    finally:
        await first.shutdown()

    stored = json.loads(profiles_file.read_text())
    assert stored == {
        "version": 1,
        "profiles": [
            {
                "id": "persistent",
                "model_path": str(model_file.resolve()),
                "args": ["--ctx-size", "4096"],
                "priority": 7,
                "estimated_memory_bytes": 321,
            }
        ],
    }
    assert profiles_file.stat().st_mode & 0o777 == 0o600

    second = PoolManager(settings, memory_reader=FixedMemoryReader())
    await second.start()
    try:
        restored = await second.get_model("persistent")
        assert restored.status == "registered"
        assert restored.pid is None
        assert restored.priority == 7
        assert restored.predicted_memory_bytes == 321
    finally:
        await second.shutdown()


@pytest.mark.asyncio
async def test_removing_a_profile_updates_the_store(
    model_file: Path, tmp_path: Path
) -> None:
    profiles_file = tmp_path / "profiles.json"
    settings = replace(manager_settings(), profiles_file=str(profiles_file))
    manager = PoolManager(settings, memory_reader=FixedMemoryReader())
    await manager.start()
    try:
        await manager.register(
            RegisterModelRequest(id="temporary", model_path=str(model_file))
        )
        await manager.remove("temporary")
    finally:
        await manager.shutdown()

    assert json.loads(profiles_file.read_text()) == {"version": 1, "profiles": []}


@pytest.mark.asyncio
async def test_invalid_profile_store_fails_startup(tmp_path: Path) -> None:
    profiles_file = tmp_path / "profiles.json"
    profiles_file.write_text('{"version": 99, "profiles": []}')
    manager = PoolManager(
        replace(manager_settings(), profiles_file=str(profiles_file)),
        memory_reader=FixedMemoryReader(),
    )

    with pytest.raises(ProfileStoreError, match="is invalid"):
        await manager.start()
    await manager.shutdown()


@pytest.mark.asyncio
async def test_failed_profile_write_rolls_back_registration(
    model_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profiles_file = tmp_path / "profiles.json"
    manager = PoolManager(
        replace(manager_settings(), profiles_file=str(profiles_file)),
        memory_reader=FixedMemoryReader(),
    )
    await manager.start()
    try:

        def fail_write(_path, _contents):
            raise OSError("disk unavailable")

        monkeypatch.setattr(manager, "_write_profiles_file", fail_write)
        with pytest.raises(ProfileStoreError, match="could not write"):
            await manager.register(
                RegisterModelRequest(id="rollback", model_path=str(model_file))
            )
        assert await manager.list_models() == []
    finally:
        await manager.shutdown()

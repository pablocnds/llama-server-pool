from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import psutil


@dataclass(frozen=True, slots=True)
class SystemMemory:
    total_bytes: int
    available_bytes: int
    used_bytes: int
    free_bytes: int | None = None
    cached_bytes: int = 0


@dataclass(frozen=True, slots=True)
class ProcessMemory:
    """Memory attributed to one managed process tree.

    ``process_bytes`` is PSS where Linux exposes it, with RSS as a fallback.
    DRM system memory is separate because GPU drivers do not necessarily map
    resident buffers into the client's CPU page tables, so it is absent from
    both PSS and RSS.
    """

    process_bytes: int = 0
    process_anon_bytes: int = 0
    process_file_bytes: int = 0
    process_shmem_bytes: int = 0
    drm_system_bytes: int = 0
    drm_vram_bytes: int = 0

    @property
    def system_bytes(self) -> int:
        return self.process_bytes + self.drm_system_bytes

    def __add__(self, other: ProcessMemory) -> ProcessMemory:
        if not isinstance(other, ProcessMemory):
            return NotImplemented
        return ProcessMemory(
            process_bytes=self.process_bytes + other.process_bytes,
            process_anon_bytes=self.process_anon_bytes + other.process_anon_bytes,
            process_file_bytes=self.process_file_bytes + other.process_file_bytes,
            process_shmem_bytes=self.process_shmem_bytes + other.process_shmem_bytes,
            drm_system_bytes=self.drm_system_bytes + other.drm_system_bytes,
            drm_vram_bytes=self.drm_vram_bytes + other.drm_vram_bytes,
        )


class MemoryReader:
    def __init__(self, *, proc_root: str | Path = "/proc") -> None:
        self.proc_root = Path(proc_root)

    async def system(self) -> SystemMemory:
        return await asyncio.to_thread(self._read_system)

    async def process(self, pid: int) -> ProcessMemory:
        return await asyncio.to_thread(self._read_process_tree, pid)

    @staticmethod
    def _read_system() -> SystemMemory:
        memory = psutil.virtual_memory()
        return SystemMemory(
            total_bytes=memory.total,
            available_bytes=memory.available,
            used_bytes=memory.total - memory.available,
            free_bytes=memory.free,
            cached_bytes=memory.cached,
        )

    def _read_process_tree(self, pid: int) -> ProcessMemory:
        try:
            root = psutil.Process(pid)
            processes = [root, *root.children(recursive=True)]
        except psutil.NoSuchProcess, psutil.AccessDenied:
            return ProcessMemory()

        usage = ProcessMemory()
        pids: list[int] = []
        for process in processes:
            pids.append(process.pid)
            usage += self._read_process_resident(process)
        drm_system, drm_vram = self._read_drm_clients(pids)
        return usage + ProcessMemory(
            drm_system_bytes=drm_system,
            drm_vram_bytes=drm_vram,
        )

    def _read_process_resident(self, process: psutil.Process) -> ProcessMemory:
        try:
            rollup = self._read_key_values(
                self.proc_root / str(process.pid) / "smaps_rollup"
            )
            pss = rollup.get("Pss", 0)
            if pss:
                anon = rollup.get("Pss_Anon", 0)
                file = rollup.get("Pss_File", 0)
                shmem = rollup.get("Pss_Shmem", 0)
                # Older kernels may expose total PSS without its three-way
                # breakdown. Treat the unknown portion as non-reclaimable.
                anon += max(0, pss - anon - file - shmem)
                return ProcessMemory(
                    process_bytes=pss,
                    process_anon_bytes=anon,
                    process_file_bytes=file,
                    process_shmem_bytes=shmem,
                )
        except OSError, ValueError:
            pass

        try:
            full_info = process.memory_full_info()
            resident = getattr(full_info, "pss", 0) or full_info.rss
        except psutil.NoSuchProcess, psutil.AccessDenied:
            try:
                resident = process.memory_info().rss
            except psutil.NoSuchProcess, psutil.AccessDenied:
                return ProcessMemory()
        return ProcessMemory(
            process_bytes=resident,
            process_anon_bytes=resident,
        )

    def _read_drm_clients(self, pids: list[int]) -> tuple[int, int]:
        """Return resident DRM system and dedicated-VRAM bytes.

        A process commonly holds multiple descriptors for the same DRM client.
        The kernel repeats complete client totals on each descriptor, so client
        identity is deduplicated across the whole managed process tree.
        """

        clients: dict[tuple[str, str, str], tuple[int, int]] = {}
        for pid in pids:
            fdinfo_directory = self.proc_root / str(pid) / "fdinfo"
            try:
                entries = list(fdinfo_directory.iterdir())
            except OSError:
                continue
            for entry in entries:
                try:
                    values = self._read_key_values(entry)
                except OSError, ValueError:
                    continue
                client_id = values.get("drm-client-id")
                if client_id is None:
                    continue
                identity = (
                    str(values.get("drm-driver", "")),
                    str(values.get("drm-pdev", "")),
                    str(client_id),
                )
                regions: dict[str, int] = {}
                for key, value in values.items():
                    if key.startswith("drm-resident-") and isinstance(value, int):
                        regions[key.removeprefix("drm-resident-")] = value
                # drm-memory-* is the deprecated amdgpu alias. Prefer the
                # standard resident key whenever both are present.
                for key, value in values.items():
                    if key.startswith("drm-memory-") and isinstance(value, int):
                        regions.setdefault(key.removeprefix("drm-memory-"), value)

                system = sum(
                    value
                    for region, value in regions.items()
                    if region in {"system", "gtt", "cpu"}
                )
                vram = sum(
                    value
                    for region, value in regions.items()
                    if region == "vram" or region.startswith("vram")
                )
                previous = clients.get(identity, (0, 0))
                clients[identity] = (max(previous[0], system), max(previous[1], vram))
        return (
            sum(system for system, _ in clients.values()),
            sum(vram for _, vram in clients.values()),
        )

    @staticmethod
    def _read_key_values(path: Path) -> dict[str, int | str]:
        result: dict[str, int | str] = {}
        with path.open(encoding="utf-8", errors="replace") as source:
            for line in source:
                key, separator, raw_value = line.partition(":")
                if not separator:
                    continue
                value = raw_value.strip()
                pieces = value.split()
                if not pieces:
                    continue
                try:
                    number = int(pieces[0])
                except ValueError:
                    result[key] = value
                    continue
                unit = pieces[1] if len(pieces) > 1 else "B"
                multipliers = {
                    "B": 1,
                    "kB": 1024,  # Linux procfs spelling; values are kibibytes.
                    "KiB": 1024,
                    "MiB": 1024**2,
                    "GiB": 1024**3,
                }
                multiplier = multipliers.get(unit)
                result[key] = number * multiplier if multiplier is not None else value
        return result

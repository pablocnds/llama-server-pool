from __future__ import annotations

import asyncio
from dataclasses import dataclass

import psutil


@dataclass(frozen=True, slots=True)
class SystemMemory:
    total_bytes: int
    available_bytes: int
    used_bytes: int


class MemoryReader:
    async def system(self) -> SystemMemory:
        return await asyncio.to_thread(self._read_system)

    async def process(self, pid: int) -> int:
        return await asyncio.to_thread(self._read_process_tree, pid)

    @staticmethod
    def _read_system() -> SystemMemory:
        memory = psutil.virtual_memory()
        return SystemMemory(
            total_bytes=memory.total,
            available_bytes=memory.available,
            used_bytes=memory.total - memory.available,
        )

    @staticmethod
    def _read_process_tree(pid: int) -> int:
        try:
            root = psutil.Process(pid)
            processes = [root, *root.children(recursive=True)]
        except psutil.NoSuchProcess, psutil.AccessDenied:
            return 0

        usage = 0
        for process in processes:
            try:
                full_info = process.memory_full_info()
                proportional = getattr(full_info, "pss", 0)
                usage += proportional or full_info.rss
            except psutil.NoSuchProcess, psutil.AccessDenied:
                try:
                    usage += process.memory_info().rss
                except psutil.NoSuchProcess, psutil.AccessDenied:
                    pass
        return usage

from __future__ import annotations

from pathlib import Path

from llama_server_pool.memory import MemoryReader, ProcessMemory


def write_fdinfo(root: Path, pid: int, fd: int, contents: str) -> None:
    directory = root / str(pid) / "fdinfo"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / str(fd)).write_text(contents)


def test_drm_memory_is_deduplicated_and_regions_are_classified(
    tmp_path: Path,
) -> None:
    client = """\
drm-driver:\tamdgpu
drm-client-id:\t6
drm-pdev:\t0000:c5:00.0
drm-resident-gtt:\t2048 KiB
drm-memory-gtt:\t2048 KiB
drm-resident-vram:\t3 MiB
drm-memory-vram:\t3 MiB
"""
    write_fdinfo(tmp_path, 10, 3, client)
    write_fdinfo(tmp_path, 10, 4, client)
    write_fdinfo(
        tmp_path,
        11,
        3,
        """\
drm-driver: xe
drm-client-id: 9
drm-pdev: 0000:03:00.0
drm-resident-system: 1024 KiB
drm-resident-vram0: 4 MiB
""",
    )

    system, vram = MemoryReader(proc_root=tmp_path)._read_drm_clients([10, 11])

    assert system == 3 * 1024**2
    assert vram == 7 * 1024**2


def test_deprecated_amdgpu_memory_alias_is_supported(tmp_path: Path) -> None:
    write_fdinfo(
        tmp_path,
        20,
        7,
        """\
drm-driver: amdgpu
drm-client-id: 3
drm-pdev: 0000:01:00.0
drm-memory-gtt: 512 KiB
drm-memory-vram: 2 MiB
""",
    )

    assert MemoryReader(proc_root=tmp_path)._read_drm_clients([20]) == (
        512 * 1024,
        2 * 1024**2,
    )


def test_non_drm_fdinfo_and_unknown_regions_are_ignored(tmp_path: Path) -> None:
    write_fdinfo(tmp_path, 30, 1, "pos: 0\nflags: 0100000\n")
    write_fdinfo(
        tmp_path,
        30,
        8,
        """\
drm-driver: example
drm-client-id: 1
drm-pdev: virtual
drm-resident-mystery: 99 MiB
""",
    )

    assert MemoryReader(proc_root=tmp_path)._read_drm_clients([30]) == (0, 0)


def test_process_memory_total_adds_only_system_drm() -> None:
    memory = ProcessMemory(
        process_bytes=100,
        process_anon_bytes=60,
        process_file_bytes=40,
        drm_system_bytes=200,
        drm_vram_bytes=300,
    )

    assert memory.system_bytes == 300

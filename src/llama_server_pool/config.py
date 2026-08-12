from __future__ import annotations

import os
from dataclasses import dataclass

GIB = 1024**3
MIB = 1024**2


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


@dataclass(frozen=True, slots=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8080
    llama_server_executable: str = "llama-server"
    internal_port_min: int = 10_000
    internal_port_max: int = 11_000
    normal_headroom_bytes: int = 2 * GIB
    critical_headroom_bytes: int = 512 * MIB
    pool_memory_budget_bytes: int | None = None
    model_size_margin_bytes: int = 512 * MIB
    monitor_interval_seconds: float = 1.0
    startup_timeout_seconds: float = 300.0
    shutdown_timeout_seconds: float = 10.0
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if not 0 < self.port <= 65_535:
            raise ValueError("port must be between 1 and 65535")
        if not 0 < self.internal_port_min <= self.internal_port_max <= 65_535:
            raise ValueError("invalid internal port range")
        if self.critical_headroom_bytes > self.normal_headroom_bytes:
            raise ValueError("critical headroom cannot exceed normal headroom")
        for name in (
            "normal_headroom_bytes",
            "critical_headroom_bytes",
            "model_size_margin_bytes",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if (
            self.pool_memory_budget_bytes is not None
            and self.pool_memory_budget_bytes <= 0
        ):
            raise ValueError("pool memory budget must be positive or unset")
        if self.monitor_interval_seconds <= 0:
            raise ValueError("monitor interval must be positive")
        if self.startup_timeout_seconds <= 0 or self.shutdown_timeout_seconds <= 0:
            raise ValueError("timeouts must be positive")

    @classmethod
    def from_env(cls) -> Settings:
        budget = _env_int("LLAMA_POOL_MEMORY_BUDGET_BYTES", 0)
        return cls(
            host=os.getenv("LLAMA_POOL_HOST", "127.0.0.1"),
            port=_env_int("LLAMA_POOL_PORT", 8080),
            llama_server_executable=os.getenv(
                "LLAMA_POOL_LLAMA_SERVER_EXECUTABLE", "llama-server"
            ),
            internal_port_min=_env_int("LLAMA_POOL_INTERNAL_PORT_MIN", 10_000),
            internal_port_max=_env_int("LLAMA_POOL_INTERNAL_PORT_MAX", 11_000),
            normal_headroom_bytes=_env_int("LLAMA_POOL_NORMAL_HEADROOM_BYTES", 2 * GIB),
            critical_headroom_bytes=_env_int(
                "LLAMA_POOL_CRITICAL_HEADROOM_BYTES", 512 * MIB
            ),
            pool_memory_budget_bytes=budget or None,
            model_size_margin_bytes=_env_int(
                "LLAMA_POOL_MODEL_SIZE_MARGIN_BYTES", 512 * MIB
            ),
            monitor_interval_seconds=_env_float(
                "LLAMA_POOL_MONITOR_INTERVAL_SECONDS", 1.0
            ),
            startup_timeout_seconds=_env_float(
                "LLAMA_POOL_STARTUP_TIMEOUT_SECONDS", 300.0
            ),
            shutdown_timeout_seconds=_env_float(
                "LLAMA_POOL_SHUTDOWN_TIMEOUT_SECONDS", 10.0
            ),
            log_level=os.getenv("LLAMA_POOL_LOG_LEVEL", "INFO"),
        )

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelStatus(StrEnum):
    REGISTERED = "registered"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"


class RegisterModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9._:-]+$")
    model_path: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    priority: int = 0
    initialize: bool = False
    force: bool = False
    estimated_memory_bytes: int | None = Field(default=None, gt=0)


class StoredProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9._:-]+$")
    model_path: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    priority: int = 0
    estimated_memory_bytes: int | None = Field(default=None, gt=0)


class ProfileStore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    profiles: list[StoredProfile] = Field(default_factory=list)


class StartModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    force: bool = False


class UpdateModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    priority: int


class ModelView(BaseModel):
    id: str
    model_path: str
    args: list[str]
    priority: int
    status: ModelStatus
    active_requests: int
    created_at: float
    last_used_at: float | None
    pid: int | None
    internal_port: int | None
    predicted_memory_bytes: int
    actual_memory_bytes: int | None
    process_memory_bytes: int | None
    process_file_memory_bytes: int | None
    gpu_shared_memory_bytes: int | None
    gpu_dedicated_memory_bytes: int | None
    last_error: str | None


class SystemMemoryView(BaseModel):
    total_bytes: int
    available_bytes: int
    used_bytes: int
    free_bytes: int
    cached_bytes: int
    outside_pool_resident_bytes: int
    outside_pool_cache_bytes: int
    normal_headroom_bytes: int
    critical_headroom_bytes: int


class PoolStatsView(BaseModel):
    system: SystemMemoryView
    pool_usage_bytes: int
    pool_budget_bytes: int | None
    registered_models: int
    running_models: int
    starting_models: int
    active_requests: int
    processes: list[ModelView]


class DiscoveredModelView(BaseModel):
    path: str
    name: str
    relative_path: str
    size_bytes: int


class ModelDiscoveryView(BaseModel):
    enabled: bool
    models: list[DiscoveredModelView]

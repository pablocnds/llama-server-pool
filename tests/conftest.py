from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from llama_server_pool.app import create_app
from llama_server_pool.config import Settings


@pytest.fixture
def model_file(tmp_path: Path) -> Path:
    path = tmp_path / "model.gguf"
    path.write_bytes(b"fake model")
    return path


@pytest.fixture
def settings() -> Settings:
    executable = Path(__file__).with_name("fake_llama_server.py")
    return Settings(
        llama_server_executable=str(executable),
        internal_port_min=19_100,
        internal_port_max=19_199,
        normal_headroom_bytes=0,
        critical_headroom_bytes=0,
        model_size_margin_bytes=0,
        monitor_interval_seconds=60,
        startup_timeout_seconds=5,
        shutdown_timeout_seconds=1,
    )


@pytest_asyncio.fixture
async def client(settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(settings)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as test_client,
    ):
        yield test_client

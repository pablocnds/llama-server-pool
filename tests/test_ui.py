from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from llama_server_pool.app import create_app
from llama_server_pool.config import Settings


@pytest.mark.asyncio
async def test_ui_and_assets_are_served_by_default(client: httpx.AsyncClient) -> None:
    redirect = await client.get("/ui", follow_redirects=False)
    assert redirect.status_code == 307
    assert redirect.headers["location"] == "/ui/"

    page = await client.get("/ui/")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert "llama-server pool" in page.text
    assert "New profile" in page.text

    javascript = await client.get("/ui/app.js")
    stylesheet = await client.get("/ui/styles.css")
    assert javascript.status_code == 200
    assert stylesheet.status_code == 200
    assert "/control/stats" in javascript.text
    assert "/control/model-files" in javascript.text
    assert "/v1/chat/completions" in javascript.text
    assert "gpu_shared_memory_bytes" in javascript.text
    assert "reclaimable" not in page.text.lower()
    assert "outside_pool_cache_bytes" not in javascript.text
    assert (
        page.text.index('id="model-segments"')
        < page.text.index('id="memory-free"')
        < page.text.index('id="memory-other"')
    )
    assert ".memory-track" in stylesheet.text
    assert "overflow: hidden" in stylesheet.text
    assert javascript.text.index('$("#memory-free")') < javascript.text.index(
        '$("#memory-other")'
    )


@pytest.mark.asyncio
async def test_ui_can_be_disabled(settings: Settings) -> None:
    app = create_app(replace(settings, ui_enabled=False))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (await client.get("/ui")).status_code == 404
        assert (await client.get("/ui/")).status_code == 404


@pytest.mark.asyncio
async def test_discovery_is_explicitly_disabled_without_a_root(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/control/model-files")
    assert response.status_code == 200
    assert response.json() == {"enabled": False, "models": []}


@pytest.mark.asyncio
async def test_discovery_lists_only_ggufs_resolving_inside_root(
    settings: Settings, tmp_path: Path
) -> None:
    root = tmp_path / "models"
    nested = root / "nested"
    nested.mkdir(parents=True)
    first = root / "one.gguf"
    second = nested / "TWO.GGUF"
    ignored = root / "notes.txt"
    outside = tmp_path / "outside.gguf"
    first.write_bytes(b"one")
    second.write_bytes(b"second")
    ignored.write_text("not a model")
    outside.write_bytes(b"outside")
    (root / "outside-link.gguf").symlink_to(outside)
    (root / "duplicate.gguf").symlink_to(first)

    app = create_app(replace(settings, model_discovery_root=str(root)))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client,
    ):
        response = await client.get("/control/model-files")

    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is True
    assert [model["relative_path"] for model in body["models"]] == [
        "nested/TWO.GGUF",
        "one.gguf",
    ]
    assert [model["size_bytes"] for model in body["models"]] == [6, 3]
    assert all(Path(model["path"]).is_relative_to(root) for model in body["models"])


@pytest.mark.asyncio
async def test_stats_include_additive_memory_fields(client: httpx.AsyncClient) -> None:
    stats = (await client.get("/control/stats")).json()
    assert stats["system"]["free_bytes"] >= 0
    assert stats["system"]["cached_bytes"] >= 0
    assert stats["system"]["outside_pool_resident_bytes"] >= 0
    assert stats["system"]["outside_pool_cache_bytes"] >= 0

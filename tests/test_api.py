from __future__ import annotations

import httpx
import pytest


@pytest.mark.asyncio
async def test_register_start_proxy_unload_and_lazy_restart(
    client: httpx.AsyncClient, model_file
) -> None:
    registration = await client.post(
        "/control/models",
        json={
            "id": "qwen-test",
            "model_path": str(model_file),
            "args": ["--ctx-size", "16000"],
            "initialize": True,
        },
    )
    assert registration.status_code == 201, registration.text
    first = registration.json()
    assert first["status"] == "running"
    assert first["pid"] is not None

    response = await client.post(
        "/v1/chat/completions",
        json={"model": "qwen-test", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert response.json()["model"] == "qwen-test"

    models = await client.get("/v1/models")
    assert [item["id"] for item in models.json()["data"]] == ["qwen-test"]

    unloaded = await client.post("/control/models/qwen-test/unload")
    assert unloaded.status_code == 200
    assert unloaded.json()["status"] == "registered"

    restarted = await client.post(
        "/v1/chat/completions",
        json={"model": "qwen-test", "messages": []},
    )
    assert restarted.status_code == 200
    status = await client.get("/control/models/qwen-test")
    assert status.json()["status"] == "running"
    assert status.json()["pid"] != first["pid"]


@pytest.mark.asyncio
async def test_streaming_proxy(client: httpx.AsyncClient, model_file) -> None:
    registered = await client.post(
        "/control/models",
        json={"id": "streamer", "model_path": str(model_file)},
    )
    assert registered.status_code == 201

    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "streamer", "messages": [], "stream": True},
    ) as response:
        content = b"".join([chunk async for chunk in response.aiter_bytes()])
    assert response.status_code == 200
    assert b"data: [DONE]" in content
    assert b"hello" in content

    model = await client.get("/control/models/streamer")
    assert model.json()["active_requests"] == 0
    assert model.json()["last_used_at"] is not None


@pytest.mark.asyncio
async def test_duplicate_identity_and_controlled_args_are_rejected(
    client: httpx.AsyncClient, model_file
) -> None:
    first = await client.post(
        "/control/models",
        json={"id": "first", "model_path": str(model_file), "args": ["--temp", "1"]},
    )
    assert first.status_code == 201

    duplicate = await client.post(
        "/control/models",
        json={"id": "second", "model_path": str(model_file), "args": ["--temp", "1"]},
    )
    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["code"] == "duplicate_model"

    forbidden = await client.post(
        "/control/models",
        json={"id": "bad", "model_path": str(model_file), "args": ["--port=9999"]},
    )
    assert forbidden.status_code == 422

    key_file = await client.post(
        "/control/models",
        json={
            "id": "also-bad",
            "model_path": str(model_file),
            "args": ["--api-key-file", "/tmp/keys"],
        },
    )
    assert key_file.status_code == 422

    missing_path = await client.post(
        "/control/models",
        json={"id": "missing", "model_path": str(model_file) + ".missing"},
    )
    assert missing_path.status_code == 422


@pytest.mark.asyncio
async def test_remove_and_missing_model_errors(
    client: httpx.AsyncClient, model_file
) -> None:
    await client.post(
        "/control/models", json={"id": "temporary", "model_path": str(model_file)}
    )
    removed = await client.delete("/control/models/temporary")
    assert removed.status_code == 204
    missing = await client.post(
        "/v1/chat/completions", json={"model": "temporary", "messages": []}
    )
    assert missing.status_code == 404

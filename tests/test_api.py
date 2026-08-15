from __future__ import annotations

import httpx
import pytest
from starlette.responses import Response

from llama_server_pool.app import _append_response_headers, _filtered_headers


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

    chat_request = {
        "model": "qwen-test",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.4,
        "reasoning_effort": "high",
        "chat_template_kwargs": {"enable_thinking": True},
    }
    response = await client.post("/v1/chat/completions", json=chat_request)
    assert response.status_code == 200
    assert response.json()["model"] == "qwen-test"
    assert response.json()["request"] == chat_request

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
async def test_health_is_pool_scoped_and_does_not_load_models(
    client: httpx.AsyncClient, model_file
) -> None:
    await client.post(
        "/control/models", json={"id": "healthy", "model_path": str(model_file)}
    )

    for path in ("/health", "/v1/health"):
        response = await client.get(path)
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    model = await client.get("/control/models/healthy")
    assert model.json()["status"] == "registered"


@pytest.mark.asyncio
async def test_arbitrary_llama_server_paths_and_request_options_are_forwarded(
    client: httpx.AsyncClient, model_file
) -> None:
    await client.post(
        "/control/models", json={"id": "forwarded", "model_path": str(model_file)}
    )
    body = {
        "model": "forwarded",
        "input": "hello",
        "temperature": 0.37,
        "top_p": 0.81,
        "reasoning_effort": "high",
        "chat_template_kwargs": {
            "enable_thinking": True,
            "custom_template_value": "untouched",
        },
        "future_llama_option": {"nested": [1, 2, 3]},
    }

    response = await client.post("/v1/responses?custom=query", json=body)

    assert response.status_code == 200
    result = response.json()
    assert result["path"] == "/v1/responses?custom=query"
    assert result["request"] == body


@pytest.mark.asyncio
async def test_client_credentials_are_replaced_with_internal_credentials(
    client: httpx.AsyncClient, model_file
) -> None:
    await client.post(
        "/control/models", json={"id": "anthropic", "model_path": str(model_file)}
    )

    response = await client.post(
        "/v1/messages",
        headers={"Authorization": "Bearer external", "X-Api-Key": "external"},
        json={"model": "anthropic", "messages": [], "max_tokens": 10},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["authorization"].startswith("Bearer pool_")
    assert result["x_api_key"] == result["authorization"].removeprefix("Bearer ")


@pytest.mark.asyncio
async def test_query_model_routes_bodies_without_a_model_field(
    client: httpx.AsyncClient, model_file
) -> None:
    await client.post(
        "/control/models", json={"id": "utility", "model_path": str(model_file)}
    )

    tokenized = await client.post(
        "/tokenize?model=utility&trace=yes",
        json={"content": "hello", "with_pieces": True},
    )
    assert tokenized.status_code == 200
    assert tokenized.json()["path"] == "/tokenize?model=utility&trace=yes"
    assert tokenized.json()["request"] == {
        "content": "hello",
        "with_pieces": True,
    }

    props = await client.get("/props?model=utility&detail=full")
    assert props.status_code == 200
    assert props.json() == {
        "path": "/props?model=utility&detail=full",
        "method": "GET",
    }


@pytest.mark.asyncio
async def test_multipart_model_field_routes_and_preserves_upload(
    client: httpx.AsyncClient, model_file
) -> None:
    await client.post(
        "/control/models", json={"id": "audio", "model_path": str(model_file)}
    )

    response = await client.post(
        "/v1/audio/transcriptions",
        data={"model": "audio", "language": "en"},
        files={"file": ("sample.wav", b"fake-wave-data", "audio/wav")},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["path"] == "/v1/audio/transcriptions"
    assert "multipart/form-data" in result["request"]["content_type"]
    assert 'name="model"' in result["request"]["raw_body"]
    assert "fake-wave-data" in result["request"]["raw_body"]


@pytest.mark.asyncio
async def test_model_is_required_for_model_scoped_proxy_requests(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post("/v1/responses", json={"input": "hello"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


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


def test_proxy_header_filtering_is_connection_aware_and_preserves_duplicates() -> None:
    headers = [
        ("Connection", "close, X-Private"),
        ("X-Private", "remove me"),
        ("Trailer", "Expires"),
        ("Set-Cookie", "first=1"),
        ("Set-Cookie", "second=2"),
        ("X-Public", "keep me"),
    ]

    filtered = _filtered_headers(headers, {"connection", "trailer"})
    response = _append_response_headers(Response(), filtered)

    assert filtered == [
        ("Set-Cookie", "first=1"),
        ("Set-Cookie", "second=2"),
        ("X-Public", "keep me"),
    ]
    assert response.headers.getlist("set-cookie") == ["first=1", "second=2"]

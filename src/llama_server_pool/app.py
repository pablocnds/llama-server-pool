from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Iterable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import (
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from starlette.requests import ClientDisconnect

from .config import Settings
from .errors import PoolError
from .manager import PoolManager, RouteLease
from .models import (
    DiscoveredModelView,
    ModelDiscoveryView,
    ModelView,
    PoolStatsView,
    RegisterModelRequest,
    StartModelRequest,
    UpdateModelRequest,
)

logger = logging.getLogger(__name__)

_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
    "trailers",
    "transfer-encoding",
    "upgrade",
}
_REQUEST_HEADERS_TO_DROP = _HOP_BY_HOP_HEADERS | {
    "authorization",
    "content-length",
    "host",
}
_RESPONSE_HEADERS_TO_DROP = _HOP_BY_HOP_HEADERS | {"content-length"}


def create_app(settings: Settings | None = None) -> FastAPI:
    configured = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        manager = PoolManager(configured)
        proxy_client = httpx.AsyncClient(
            timeout=httpx.Timeout(None, connect=10.0), trust_env=False
        )
        app.state.manager = manager
        app.state.proxy_client = proxy_client
        try:
            await manager.start()
            yield
        finally:
            await proxy_client.aclose()
            await manager.shutdown()

    app = FastAPI(
        title="llama-server pool",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.exception_handler(PoolError)
    async def pool_error_handler(_request: Request, exc: PoolError) -> JSONResponse:
        return _error_response(exc.status_code, exc.message, exc.code)

    @app.exception_handler(ValueError)
    async def value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
        return _error_response(422, str(exc), "invalid_request")

    @app.post("/control/models", response_model=ModelView, status_code=201)
    async def register_model(
        request: RegisterModelRequest, raw_request: Request
    ) -> ModelView:
        return await _await_or_disconnect(
            raw_request, _manager(raw_request).register(request)
        )

    @app.get("/control/models", response_model=list[ModelView])
    async def list_control_models(request: Request) -> list[ModelView]:
        return await _manager(request).list_models(refresh_memory=True)

    @app.get("/control/models/{model_id}", response_model=ModelView)
    async def get_control_model(model_id: str, request: Request) -> ModelView:
        return await _manager(request).get_model(model_id, refresh_memory=True)

    @app.post("/control/models/{model_id}/start", response_model=ModelView)
    async def start_model(
        model_id: str, body: StartModelRequest, request: Request
    ) -> ModelView:
        return await _await_or_disconnect(
            request, _manager(request).start_model(model_id, force=body.force)
        )

    @app.post("/control/models/{model_id}/unload", response_model=ModelView)
    async def unload_model(model_id: str, request: Request) -> ModelView:
        return await _manager(request).unload(model_id)

    @app.patch("/control/models/{model_id}", response_model=ModelView)
    async def update_model(
        model_id: str, body: UpdateModelRequest, request: Request
    ) -> ModelView:
        return await _manager(request).update_priority(model_id, body.priority)

    @app.delete("/control/models/{model_id}", status_code=204)
    async def remove_model(model_id: str, request: Request) -> Response:
        await _manager(request).remove(model_id)
        return Response(status_code=204)

    @app.get("/control/stats", response_model=PoolStatsView)
    async def pool_stats(request: Request) -> PoolStatsView:
        return await _manager(request).stats()

    @app.get("/control/model-files", response_model=ModelDiscoveryView)
    async def discover_model_files() -> ModelDiscoveryView:
        if configured.model_discovery_root is None:
            return ModelDiscoveryView(enabled=False, models=[])
        models = await asyncio.to_thread(
            _discover_models, Path(configured.model_discovery_root)
        )
        return ModelDiscoveryView(enabled=True, models=models)

    @app.get("/v1/models")
    async def openai_models(request: Request) -> dict[str, Any]:
        models = await _manager(request).list_models()
        return {
            "object": "list",
            "data": [
                {
                    "id": model.id,
                    "object": "model",
                    "created": int(model.created_at),
                    "owned_by": "llama-server-pool",
                }
                for model in models
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        raw_body = await request.body()
        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError, UnicodeDecodeError:
            return _error_response(
                400, "request body must be valid JSON", "invalid_request"
            )
        if not isinstance(body, dict) or not isinstance(body.get("model"), str):
            return _error_response(
                400, "request body must contain a string model field", "invalid_request"
            )

        manager = _manager(request)
        lease = await _await_or_disconnect(
            request, manager.acquire_route(body["model"])
        )
        proxy_client: httpx.AsyncClient = request.app.state.proxy_client
        headers = _filtered_headers(
            (
                (key.decode("latin-1"), value.decode("latin-1"))
                for key, value in request.headers.raw
            ),
            _REQUEST_HEADERS_TO_DROP,
        )
        headers.append(("Authorization", f"Bearer {lease.api_key}"))
        upstream_request = proxy_client.build_request(
            "POST",
            f"http://127.0.0.1:{lease.port}/v1/chat/completions",
            headers=headers,
            content=raw_body,
        )
        try:
            upstream = await proxy_client.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            await lease.release()
            logger.warning(
                "proxy connection to model %s failed: %s", body["model"], exc
            )
            return _error_response(
                502, "the model server is unavailable", "upstream_error"
            )

        response_headers = _filtered_headers(
            upstream.headers.multi_items(), _RESPONSE_HEADERS_TO_DROP
        )
        if body.get("stream") is True:
            return _append_response_headers(
                StreamingResponse(
                    _stream_upstream(upstream, lease),
                    status_code=upstream.status_code,
                    media_type=None,
                ),
                response_headers,
            )

        try:
            content = b"".join([chunk async for chunk in upstream.aiter_raw()])
        except httpx.HTTPError as exc:
            logger.warning(
                "proxy response from model %s failed: %s", body["model"], exc
            )
            return _error_response(
                502, "the model server disconnected", "upstream_error"
            )
        finally:
            await upstream.aclose()
            await lease.release()
        return _append_response_headers(
            Response(content=content, status_code=upstream.status_code),
            response_headers,
        )

    if configured.ui_enabled:
        static_directory = Path(__file__).with_name("static")

        @app.get("/ui", include_in_schema=False)
        async def ui_redirect() -> RedirectResponse:
            return RedirectResponse("/ui/", status_code=307)

        app.mount(
            "/ui",
            StaticFiles(directory=static_directory, html=True),
            name="ui",
        )

    return app


async def _stream_upstream(
    upstream: httpx.Response, lease: RouteLease
) -> AsyncIterator[bytes]:
    try:
        async for chunk in upstream.aiter_raw():
            yield chunk
    except httpx.HTTPError as exc:
        logger.warning("streaming upstream disconnected: %s", exc)
    finally:
        await upstream.aclose()
        await lease.release()


def _manager(request: Request) -> PoolManager:
    return request.app.state.manager


async def _await_or_disconnect[T](request: Request, operation: Awaitable[T]) -> T:
    """Cancel capacity waits and startup if the requesting client goes away."""
    task = asyncio.create_task(operation)
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=0.25)
            if done:
                return await task
            if await request.is_disconnected():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                raise ClientDisconnect()
    except BaseException:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        raise


def _error_response(status_code: int, message: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": code,
                "code": code,
            }
        },
    )


def _filtered_headers(
    headers: Iterable[tuple[str, str]], excluded: set[str]
) -> list[tuple[str, str]]:
    items = list(headers)
    connection_options = {
        option.strip().lower()
        for key, value in items
        if key.lower() == "connection"
        for option in value.split(",")
        if option.strip()
    }
    blocked = excluded | connection_options
    return [(key, value) for key, value in items if key.lower() not in blocked]


def _append_response_headers[T: Response](
    response: T, headers: Iterable[tuple[str, str]]
) -> T:
    response.raw_headers.extend(
        (key.lower().encode("latin-1"), value.encode("latin-1"))
        for key, value in headers
    )
    return response


def _discover_models(root: Path) -> list[DiscoveredModelView]:
    root = root.resolve(strict=True)
    discovered: list[DiscoveredModelView] = []
    seen: set[Path] = set()
    for candidate in root.rglob("*"):
        if candidate.suffix.lower() != ".gguf":
            continue
        try:
            resolved = candidate.resolve(strict=True)
            if not resolved.is_file() or not resolved.is_relative_to(root):
                continue
            size = resolved.stat().st_size
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        discovered.append(
            DiscoveredModelView(
                path=str(resolved),
                name=resolved.name,
                relative_path=str(resolved.relative_to(root)),
                size_bytes=size,
            )
        )
    return sorted(discovered, key=lambda item: item.relative_path.casefold())

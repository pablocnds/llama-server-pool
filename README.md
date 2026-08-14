# Llama-server Pool

Process manager and supervisor of multiple llama-servers. Keeps multiple llms loaded within the system's memory avoiding it from overfilling. Provides an OpenAI-compatible proxy for communication with the models.

Instances can be added, unloaded or removed on request. Otherwise, processes are automatically removed only when the memory threshold is about to fill. The choice of which process is evicted is made according to a mix of configurable factors.

## Requirements and installation

- Python 3.14 or newer
- Linux (PSS and available-memory accounting use Linux process facilities)
- Recent version of llama.cpp with `llama-server`

```console
python3.14 -m venv .venv
.venv/bin/pip install -e '.[test]'
```

## Running

```console
LLAMA_POOL_LLAMA_SERVER_EXECUTABLE=/path/to/llama-server \
  .venv/bin/llama-server-pool
```

The manager binds to `127.0.0.1:8080` by default. Interactive API documentation
is available at `/docs` while it is running. The optional monitoring and control
panel is available at <http://127.0.0.1:8080/ui/>.

Register and initialize a model:

```console
curl -X POST http://127.0.0.1:8080/control/models \
  -H 'content-type: application/json' \
  -d '{
    "id": "qwen-16k",
    "model_path": "/models/qwen.gguf",
    "args": ["--ctx-size", "16000", "--temp", "1.0"],
    "priority": 10,
    "initialize": true
  }'
```

Route a completion through its stable ID:

```console
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen-16k","messages":[{"role":"user","content":"Hello"}]}'
```

Both regular and streaming chat completions are supported. Requesting an
unloaded registered model starts it automatically.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/control/models` | Register a model; optionally initialize it |
| `GET` | `/control/models` | List registrations and process state |
| `GET` | `/control/models/{id}` | Get one registration and process state |
| `POST` | `/control/models/{id}/start` | Initialize it; body is `{"force": false}` |
| `POST` | `/control/models/{id}/unload` | Stop its process but retain registration |
| `PATCH` | `/control/models/{id}` | Change priority with `{"priority": 10}` |
| `DELETE` | `/control/models/{id}` | Stop and remove the registration |
| `GET` | `/control/stats` | Get system, pool, and per-process memory data |
| `GET` | `/control/model-files` | List GGUF files inside the optional discovery root |
| `GET` | `/v1/models` | List all registered stable model IDs |
| `POST` | `/v1/chat/completions` | Proxy an OpenAI-compatible request |

Registration requires `id` and `model_path`. `args` defaults to an empty array,
and `priority` defaults to zero. Lower numeric priorities are evicted first.
`estimated_memory_bytes` may override the default prediction for split or
unusual models. Otherwise, the prediction is the model file size plus the
configured margin.

The pool owns the llama-server `--model`, `--alias`, `--host`, `--port`,
`--api-key`, and `--api-key-file` options. Supplying any of those through
registration arguments is rejected. Duplicate registrations with the same
resolved model path and exact argument list return HTTP 409.

## Monitoring and control panel

The lightweight browser panel at `/ui/` is enabled by default. It is a static,
same-origin client of the documented pool API and has no privileged backend
access or separate state. It provides:

- A system-memory bar with a segment for each loaded profile, available
  capacity in the middle, other non-cache system use anchored at the right,
  and pool-budget and headroom markers
- Live profile status, PSS/RSS usage, active request counts, priority, and
  last-used time
- Load, force-load, unload, and unregister controls
- Profile creation using discovered or already-registered model files
- Ephemeral streaming chats kept only in the current browser tab

Set `LLAMA_POOL_MODEL_DISCOVERY_ROOT` to allow the creation form to list GGUF
files recursively beneath one directory. Resolved files must remain inside that
root, so symlinks cannot expose files elsewhere. The UI deliberately has no
free-form model-path field. Without a discovery root, it can only create another
configuration using a model file already referenced by a registered profile.

The panel has exactly the same access as the API. It shows a warning when opened
through a non-loopback hostname because this MVP does not yet provide
authentication. Disabling the panel removes `/ui` but does not disable or alter
any control endpoint.

## Configuration

Configuration uses environment variables. Byte values are integer byte counts.

| Variable | Default |
| --- | --- |
| `LLAMA_POOL_HOST` | `127.0.0.1` |
| `LLAMA_POOL_PORT` | `8080` |
| `LLAMA_POOL_LLAMA_SERVER_EXECUTABLE` | `llama-server` |
| `LLAMA_POOL_INTERNAL_PORT_MIN` | `10000` |
| `LLAMA_POOL_INTERNAL_PORT_MAX` | `11000` |
| `LLAMA_POOL_NORMAL_HEADROOM_BYTES` | `2147483648` (2 GiB) |
| `LLAMA_POOL_CRITICAL_HEADROOM_BYTES` | `536870912` (512 MiB) |
| `LLAMA_POOL_MEMORY_BUDGET_BYTES` | `0` (unlimited) |
| `LLAMA_POOL_MODEL_SIZE_MARGIN_BYTES` | `536870912` (512 MiB) |
| `LLAMA_POOL_MONITOR_INTERVAL_SECONDS` | `1` |
| `LLAMA_POOL_STARTUP_TIMEOUT_SECONDS` | `300` |
| `LLAMA_POOL_SHUTDOWN_TIMEOUT_SECONDS` | `10` |
| `LLAMA_POOL_LOG_LEVEL` | `INFO` |
| `LLAMA_POOL_UI_ENABLED` | `true` |
| `LLAMA_POOL_MODEL_DISCOVERY_ROOT` | unset (discovery disabled) |
| `LLAMA_POOL_PROFILES_FILE` | `$XDG_CONFIG_HOME/llama-server-pool/profiles.json` or `~/.config/llama-server-pool/profiles.json` |

Set the log level to `DEBUG` to record each model's predicted memory and its
measured PSS (or RSS fallback) immediately after initialization.

## Memory and eviction behavior

Pool usage is the sum of the child process trees' PSS where available, falling
back to RSS, plus DRM buffers resident in system-memory regions such as GTT.
DRM client totals are read from Linux process `fdinfo` and deduplicated when a
client owns multiple descriptors. Dedicated VRAM is reported separately and is
not charged to the system-memory pool budget. System capacity uses available
memory. Before startup, idle models are evicted until both normal system
headroom and the optional pool budget can accommodate the prediction. If every
eligible model is active, initialization waits; `force: true` permits active
eviction but never bypasses a memory limit.

Eviction uses the lowest numeric priority, then least-recently-used order.
Normal pressure only evicts idle processes. Critical pressure may also stop a
starting or active process to protect the host. Forced or critical eviction of
an active process interrupts its in-flight response or stream.

The pool terminates every managed process group during shutdown. VRAM
accounting is not yet implemented.

## Tests

The test suite uses a real subprocess that mimics the relevant llama-server
health and chat APIs:

```console
.venv/bin/pytest
```

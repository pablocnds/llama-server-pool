### PROJECT REQUIREMENTS

- Design and implementation is simple and straightforward. Avoids unnecessary complexity that drifts from these requirements.

- Listens for requests to register and initialize llama-server processes. Each registered model has a stable model ID that outlives any particular subprocess, so an evicted model is automatically initialized again when its model ID is requested through the inference endpoint. Avoids duplicate servers with the same model and args. Only one subprocess initialization is performed at a time, and eviction decisions are synchronized with it.

- Can unload specific active servers on API request while retaining their registered models, or remove their registrations entirely.

- Other API requests include getting stats like memory usage, individual process memory usage, etc.

- The service hosts a transparent OpenAI-compatible and native llama-server proxy and routes model-scoped requests to each corresponding subprocess according to the stable model ID. It extracts only the routing model, forwards request and response data without maintaining duplicate endpoint schemas, and supports streaming on every proxied path. Pool-wide `/v1/models` and health endpoints are handled by the pool itself. Each subprocess is hosted on its own internal port on 127.0.0.1 and protected by a manager-generated API key, as it is meant to communicate only through the pool. By default, this pool is also hosted on localhost, unless otherwise specified.

- Subprocess ports are dynamically allocated from a configurable range and are not part of a model's public identity. A port can be reused after its previous process has exited, and port conflicts are retried with another port.

- The service additionally hosts a control plane for server management requests, such as unloading one of the servers or getting usage metrics.

- An instance is active while at least one request is being proxied to it. Streaming requests remain active until the stream finishes or disconnects. The instance's last-used time is updated when a request finishes. All intended inference traffic passes through the pool.

- Constantly monitors system memory usage. If system memory is about to fill, it can kill one or more llama-server processes. It should leave some overall room for the system. A normal headroom threshold evicts only idle instances; a lower critical threshold may also kill starting or active instances to protect the system. Both thresholds are configurable.

- The service can be configured to use only up to a given memory budget. By default the pool budget is unlimited, while the system headroom still applies. Overall memory pressure can cause the eviction of processes even while the pool is under its own budget. System capacity is based on available rather than merely free memory, and pool usage is based on the subprocesses' proportional memory usage where available, falling back to resident memory usage.

- Predicts the memory usage of a model before adding it. If it would cross the system headroom or pool budget, it kills another process first to leave room. It only predicts the initial model size plus a configurable margin and does not predict the full KV cache size, as memory is monitored continuously.

- If a new process is requested while there is no room for it and all eligible instances are processing requests, then by default it waits until one is finished. The request can instead specify a `force` argument, which permits evicting one or more active instances as necessary but does not bypass the memory limits.

- When choosing which process to evict, it chooses the lowest-priority eligible instance first and then the least recently used instance within that priority. It does not evict starting or active processes except under critical memory pressure, a forced initialization, or an explicit unload request. When either starting or active instances must be considered, starting instances are preferred over active ones. A model's priority can be set when registering it and changed with an independent request.

- Provides sensible defaults. Global configuration includes the system headroom thresholds, pool memory budget, model-size margin, monitoring interval, subprocess port range, llama-server executable, and shutdown timeout. Model registrations can specify llama-server arguments and priority, while initialization requests can specify `force` behavior.

- The configured llama-server executable is invoked directly with an argument list, without shell interpretation. The pool controls and prevents callers from overriding subprocess host, port, and API key settings. It manages the full subprocess lifetime and terminates its subprocesses when the pool shuts down.

- Initially it will work with regular system memory, but it will be expanded to VRAM eventually.

- Logs errors, unexpected behavior from the instances (e.g. one instance dies on its own), unexpected events such as memory going through the budget, when a server is created or killed and why. Keeps logging relatively light.

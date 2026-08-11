# Llama-server Pool

This project is meant to provide a service that can dynamically maintain multiple LLMs loaded within the system's memory limit.

Llama servers can be added, unloaded or removed on request. Otherwise, processes are automatically removed or unloaded only when the memory threshold is about to fill. The choice of which model is removed is made according to a mix of configurable factors.

This service constantly monitors system memory and the growing cache size of the models to avoid memory from completely filling up.

Existing llama.cpp and llama-swap tooling can load and route between multiple models, but do not dynamically manage model residency according to live system memory usage and a configurable memory budget. This project is meant to fulfill that niche.

# Llama-server Pool

This project is meant to provide a service that can dynamically mantain multiple llms loaded within the system's memory limit.

Llama servers can be added, unloaded or removed on request. Otherwise, processes are automatically removed or unloaded only when the memory threshold is about to fill. The choice of which model is removed is made according to a mix of configurable factors.

This service constantly monitors system memory and the growing cache size of the models to avoid memory from completely filling up.

As of writing, neither native llama.cpp tools nor llama-swap provide the posibility of dynamically loading and managing multiple llms in memory simultaneously. They normally require to unload the current model before initializing the new one. This project is meant to fulfill that niche.

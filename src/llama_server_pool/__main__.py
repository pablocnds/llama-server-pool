import logging

import uvicorn

from .config import Settings


def main() -> None:
    settings = Settings.from_env()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Health polling and proxy traffic otherwise produce one dependency log entry
    # per request, obscuring the pool's lifecycle and memory events.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    uvicorn.run(
        "llama_server_pool.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()

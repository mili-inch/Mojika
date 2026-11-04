import argparse
import logging
from pathlib import Path

import uvicorn

from .app import create_app
from .config import ServerConfig
from .engine import Engine


def main():
    parser = argparse.ArgumentParser(description="Mojika generation studio")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = ServerConfig.read(args.config)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    uvicorn.run(
        create_app(config, Engine),
        host=config.host,
        port=config.port,
        proxy_headers=True,
        forwarded_allow_ips=f"127.0.0.1,{config.host}",
        ws_max_size=4096,
        ws_per_message_deflate=False,
        ws_ping_interval=15,
        ws_ping_timeout=15,
        timeout_graceful_shutdown=40,
        access_log=False,
    )


if __name__ == "__main__":
    main()

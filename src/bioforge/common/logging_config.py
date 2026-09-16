"""One-line structured logging setup shared by every service entrypoint."""
from __future__ import annotations

import logging
import sys


def configure_logging(service_name: str, level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format=f"%(asctime)s %(levelname)s [{service_name}] %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )

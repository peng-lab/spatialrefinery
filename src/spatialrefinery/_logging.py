from __future__ import annotations

import logging


def _setup_logger() -> logging.Logger:
    from rich.console import Console
    from rich.logging import RichHandler

    # Attach the rich handler to the "spatialrefinery" *package* logger, not this
    # submodule's own logger. Every other module (registry, downloader, converter,
    # utils, io.xenium) does `logging.getLogger(__name__)` and relies on
    # propagation to reach a handler; attaching it here but on a leaf logger with
    # `propagate = False` would make every one of those calls invisible unless the
    # host application configured logging itself.
    logger = logging.getLogger("spatialrefinery")
    logger.setLevel(logging.INFO)
    console = Console(force_terminal=True)
    if console.is_jupyter is True:
        console.is_jupyter = False
    ch = RichHandler(show_path=False, console=console, show_time=False)
    logger.addHandler(ch)

    # this prevents double outputs (e.g. via a root logger handler)
    logger.propagate = False
    return logger


logger = _setup_logger()

"""Запуск: python -m aldk_light [--tray]"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from .settings import data_dir


def _setup_logging() -> None:
    # Приложение запускается без консоли, поэтому всё пишем в файл.
    handler = RotatingFileHandler(data_dir() / "log.txt", maxBytes=512_000, backupCount=1, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    logging.getLogger("bleak").setLevel(logging.WARNING)
    logging.getLogger("pywebview").setLevel(logging.WARNING)

    def on_exception(exc_type, exc, tb) -> None:
        logging.getLogger("aldk_light").critical("unhandled exception", exc_info=(exc_type, exc, tb))

    sys.excepthook = on_exception


def main() -> int:
    _setup_logging()
    from .app import run  # после настройки журнала: ошибки импорта тоже попадут в файл

    try:
        return run(start_hidden="--tray" in sys.argv)
    except Exception:
        logging.getLogger("aldk_light").exception("fatal error")
        return 1


if __name__ == "__main__":
    sys.exit(main())

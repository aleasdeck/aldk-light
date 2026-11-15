"""Запуск вместе с Windows: запись в разделе Run текущего пользователя."""

from __future__ import annotations

import logging
import sys
import winreg
from pathlib import Path

log = logging.getLogger(__name__)

_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
_VALUE_NAME = "AldkLight"


def command() -> str:
    """Команда автозапуска: приложение стартует свёрнутым в трей."""
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}" --tray'
    # Запуск из исходников: pythonw рядом с текущим интерпретатором + скрипт-лаунчер.
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    launcher = Path(__file__).resolve().parent.parent / "aldk_light_app.py"
    return f'"{pythonw}" "{launcher}" --tray'


def _read() -> str | None:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY) as key:
            value, _ = winreg.QueryValueEx(key, _VALUE_NAME)
            return str(value)
    except FileNotFoundError:
        return None


def is_enabled() -> bool:
    return _read() is not None


def set_enabled(enabled: bool) -> None:
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
        if enabled:
            winreg.SetValueEx(key, _VALUE_NAME, 0, winreg.REG_SZ, command())
        else:
            try:
                winreg.DeleteValue(key, _VALUE_NAME)
            except FileNotFoundError:
                pass


def refresh() -> None:
    """Если автозапуск включён, а программу перенесли — поправить путь в записи."""
    try:
        current = _read()
        if current is not None and current != command():
            set_enabled(True)
    except OSError:
        log.exception("autostart refresh failed")

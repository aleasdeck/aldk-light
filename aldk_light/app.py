"""Окно (WebView2 через pywebview), значок в трее и защита от второго экземпляра.

Окно существует, только пока оно открыто: в трее приложение работает без него.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import threading
from pathlib import Path

import pystray
import webview

from . import autostart, protocol, sun
from .engine import Engine
from .icon import make_icon
from .screen import describe_monitors
from .settings import Settings

log = logging.getLogger(__name__)

APP_NAME = "Aldk Light"

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.CreateMutexW.restype = ctypes.c_void_p
_kernel32.CreateEventW.restype = ctypes.c_void_p
_kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
_kernel32.SetEvent.argtypes = [ctypes.c_void_p]
_ERROR_ALREADY_EXISTS = 183
_INFINITE = 0xFFFFFFFF


def _web_dir() -> Path:
    # В собранном exe файлы распаковываются во временную папку sys._MEIPASS.
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / "aldk_light" / "web"


class Api:
    """Методы, которые вызывает страница. Всё, что не для страницы, начинается с подчёркивания."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._window = None          # окно появляется позже, чем этот объект
        self._fullscreen = False

    def _set_fullscreen(self, wanted: bool) -> None:
        if self._window is None or wanted == self._fullscreen:
            return
        try:
            self._window.toggle_fullscreen()
            self._fullscreen = wanted
        except Exception:
            log.exception("fullscreen switch failed")

    def init(self) -> dict:
        settings = self._engine.s
        try:
            monitors = describe_monitors()
        except Exception:
            log.exception("monitor list failed")
            monitors = []
        log.info("interface loaded, monitors: %d", len(monitors))
        return {
            "settings": {
                key: getattr(settings, key)
                for key in (
                    "power", "mode", "hue", "sat", "brightness", "effect", "effect_speed", "reverse",
                    "screen_displays", "screen_cycle", "cycle_period", "smoothing", "saturation", "min_brightness", "dark_dimming", "screen_calibration", "sensitivity",
                    "music_rainbow", "off_on_exit", "schedule_enabled", "schedule_on", "schedule_off",
                    "schedule_days", "schedule_sun", "schedule_margin", "schedule_city", "schedule_lat",
                    "schedule_lon", "device_name",
                )
            },
            "cities": [[name, lat, lon] for name, lat, lon, _offset in sorted(sun.CITIES)],
            "effects": [[effect_id, name] for effect_id, name in protocol.EFFECTS],
            "monitors": monitors,
            "autostart": autostart.is_enabled(),
        }

    def poll(self) -> dict:
        return self._engine.live()

    def sun_info(self) -> dict:
        return self._engine.sun_info()

    def monitors(self) -> list:
        """Свежий список мониторов: после сна или переподключения он мог измениться."""
        try:
            return describe_monitors()
        except Exception:
            log.exception("monitor list failed")
            return []

    def set_power(self, on: bool) -> None:
        self._engine.set_power(on)

    def set_mode(self, mode: int) -> None:
        self._engine.set_mode(mode)

    def set_brightness(self, percent: int) -> None:
        self._engine.set_brightness(percent)

    def set_color(self, hue: int, sat: int) -> None:
        self._engine.set_color(hue, sat)

    def set_effect(self, effect_id: int) -> None:
        self._engine.set_effect(effect_id)

    def set(self, key: str, value):
        return self._engine.set(key, value)

    def calibration_start(self) -> dict:
        """Калибровка цветов: образец занимает весь экран, чтобы его было удобно сравнивать со стеной."""
        info = self._engine.calibration_start()
        self._set_fullscreen(True)
        return info

    def calibration_show(self, hue: int) -> None:
        self._engine.calibration_show(hue)

    def calibration_finish(self, values=None) -> bool:
        self._set_fullscreen(False)
        return self._engine.calibration_finish(values)

    def calibration_reset(self) -> None:
        self._engine.calibration_reset()

    def set_autostart(self, enabled: bool) -> bool:
        autostart.set_enabled(bool(enabled))
        return autostart.is_enabled()


def run(start_hidden: bool) -> int:
    # Два экземпляра дрались бы за одно Bluetooth-подключение: второй просто показывает окно первого.
    mutex = _kernel32.CreateMutexW(None, False, "AldkLight.SingleInstance")
    already_running = ctypes.get_last_error() == _ERROR_ALREADY_EXISTS
    show_event = _kernel32.CreateEventW(None, False, False, "AldkLight.ShowWindow")
    if already_running:
        _kernel32.SetEvent(show_event)
        return 0

    engine = Engine(Settings.load())
    engine.start()
    autostart.refresh()

    # Окно — это встроенный браузер (WebView2), он держит около 450 МБ памяти. Поэтому в трее окна
    # нет вообще: оно создаётся, когда его просят показать, и уничтожается крестиком. Лампами
    # управляет движок, ему окно не нужно. Замер: с окном ~670 МБ, без него ~100 МБ.
    quitting = threading.Event()
    wake = threading.Event()            # попросили показать окно или выйти
    hint_shown = threading.Event()
    lock = threading.Lock()
    shown: list = []                    # окно, которое сейчас открыто (не больше одного)

    def current_window():
        with lock:
            return shown[0] if shown else None

    def show_window(*_args) -> None:
        window = current_window()
        if window is None:
            wake.set()                  # главный поток создаст окно
            return
        try:
            window.show()
            window.restore()
        except Exception:
            log.exception("show window failed")

    def quit_app(*_args) -> None:
        quitting.set()
        wake.set()
        window = current_window()
        if window is not None:
            try:
                window.destroy()
            except Exception:
                log.exception("window destroy failed")

    tray = pystray.Icon(
        "AldkLight",
        make_icon(64),
        APP_NAME,
        pystray.Menu(
            pystray.MenuItem("Открыть", show_window, default=True),
            pystray.MenuItem("Включить / выключить лампы", lambda *_: engine.toggle_power()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Выход", quit_app),
        ),
    )
    tray.run_detached()

    def on_closing() -> bool:
        # Крестик закрывает окно совсем, приложение остаётся в трее. С этого момента просьба
        # «показать окно» должна создавать новое, а не трогать закрывающееся.
        with lock:
            shown.clear()
        if not quitting.is_set() and not hint_shown.is_set():
            hint_shown.set()
            try:
                tray.notify("Приложение работает в трее. Выход — через меню значка.", APP_NAME)
            except Exception:
                log.exception("tray notify failed")
        return True

    def wait_for_second_instance() -> None:
        while _kernel32.WaitForSingleObject(show_event, _INFINITE) == 0:
            if quitting.is_set():
                return
            show_window()

    threading.Thread(target=wait_for_second_instance, name="show-signal", daemon=True).start()

    if not start_hidden:
        wake.set()
    try:
        while True:
            wake.wait()
            wake.clear()
            if quitting.is_set():
                break
            api = Api(engine)
            window = webview.create_window(
                APP_NAME,
                url=str(_web_dir() / "index.html"),
                js_api=api,
                width=1000,
                height=700,
                min_size=(860, 620),
                background_color="#0d0d12",
            )
            api._window = window
            window.events.closing += on_closing
            with lock:
                shown[:] = [window]
            try:
                webview.start()  # возвращается, когда окно закрыли
            finally:
                with lock:
                    shown.clear()
                engine.calibration_finish(None)  # окно закрыли посреди калибровки — лампы обратно в свой режим
    finally:
        quitting.set()
        engine.shutdown()
        try:
            tray.stop()
        except Exception:
            log.exception("tray stop failed")
        del mutex  # держим ссылку до самого конца, чтобы мьютекс жил всё время работы
    return 0

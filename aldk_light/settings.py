"""Настройки приложения, хранятся в %APPDATA%\\AldkLight\\settings.json."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .protocol import DEFAULT_DEVICE_NAME
from .sun import default_location

log = logging.getLogger(__name__)

MODE_COLOR, MODE_EFFECT, MODE_SCREEN, MODE_MUSIC = range(4)


def data_dir() -> Path:
    path = Path(os.environ.get("APPDATA") or Path.home()) / "AldkLight"
    path.mkdir(parents=True, exist_ok=True)
    return path


# Версия формата настроек. 2 — ползунок «Плавность» растянут до 3 секунд.
SETTINGS_VERSION = 2

# Из чего состоят списки в настройках (по умолчанию — из целых чисел).
_LIST_ITEMS = {"screen_displays": str}

# Город по умолчанию подбирается по часовому поясу системы, без сети.
_HOME = default_location()


@dataclass
class Settings:
    version: int = SETTINGS_VERSION
    device_name: str = DEFAULT_DEVICE_NAME
    last_address: str = ""   # адрес контроллера с прошлого подключения — по нему поиск быстрее
    power: bool = True
    mode: int = MODE_COLOR
    hue: int = 30            # 0..359
    sat: int = 800           # 0..1000
    brightness: int = 100    # %
    effect: int = 0x01
    effect_speed: int = 50   # 1..100
    reverse: bool = False
    # Мониторы для подсветки, по «положению и размеру» (screen.monitor_key); пусто — все.
    # Номера мониторов здесь не годятся: Windows меняет их после сна.
    screen_displays: list[str] = field(default_factory=list)
    screen_cycle: bool = False  # показывать цвета трёх частей экрана по очереди
    cycle_period: int = 20   # десятые доли секунды на одну часть: 5..50
    smoothing: int = 60      # 0..95
    saturation: int = 130    # %, усиление насыщенности в режиме экрана
    min_brightness: int = 5  # 0..40, нижняя граница яркости на тёмных сценах (мощность = квадрат: 10 → 1%)
    dark_dimming: int = 60   # 0..100, насколько сильно глушить лампы на тёмных сценах
    blue_trim: int = 100     # %, доля синего канала: у светодиодов синий заметнее, чем на мониторе
    sensitivity: int = 100   # %, чувствительность цветомузыки
    music_rainbow: bool = True
    off_on_exit: bool = False
    schedule_enabled: bool = False
    schedule_on: str = "09:00"
    schedule_off: str = "23:30"
    schedule_days: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])  # 0 — понедельник
    # Расписание «по солнцу»: включать на закате, выключать на рассвете.
    schedule_sun: bool = False       # False — по часам (schedule_on / schedule_off)
    schedule_margin: int = 20        # минут: включать раньше заката и выключать позже рассвета
    schedule_city: str = _HOME[0]    # пусто — координаты введены вручную
    schedule_lat: float = _HOME[1]
    schedule_lon: float = _HOME[2]

    @classmethod
    def load(cls) -> "Settings":
        path = data_dir() / "settings.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return cls()
        except (OSError, ValueError):
            log.exception("settings unreadable, using defaults")
            return cls()
        if not isinstance(raw, dict):
            return cls()
        defaults = cls()
        known = {}
        for item in fields(cls):
            value = raw.get(item.name)
            expected = type(getattr(defaults, item.name))
            if expected is float and type(value) is int:
                value = float(value)  # в JSON «55» и «55.0» неразличимы
            # Берём только значения ожидаемого типа — файл могли править руками.
            if type(value) is not expected:
                continue
            if isinstance(value, list) and not all(type(x) is _LIST_ITEMS.get(item.name, int) for x in value):
                continue
            known[item.name] = value
        if raw.get("version", 1) < 2 and "smoothing" in known:
            # Раньше правый край ползунка означал 1.23 с, теперь 3 с. Пересчитываем положение,
            # чтобы у пользователя осталось то же время перехода, что он себе подобрал.
            from .fade import smoothing_for

            share = min(95, max(0, known["smoothing"])) / 95
            known["smoothing"] = smoothing_for(0.03 + share * share * 1.2)
        known["version"] = SETTINGS_VERSION
        return cls(**known)

    def save(self) -> None:
        path = data_dir() / "settings.json"
        try:
            path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            log.exception("settings save failed")

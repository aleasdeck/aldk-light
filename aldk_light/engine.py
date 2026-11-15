"""Движок приложения: режимы, источники цвета, расписание. От интерфейса не зависит.

Все публичные методы можно вызывать из любого потока.
"""

from __future__ import annotations

import colorsys
import logging
import math
import re
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Callable

import numpy as np

from . import protocol, sun
from .ble import LampLink
from .fade import ColorFader
from .schedule import desired_power, parse_hhmm
from . import screen
from .screen import ScreenFrame, ScreenSampler, lamp_hue_sat, scene_brightness
from .settings import MODE_COLOR, MODE_EFFECT, MODE_MUSIC, MODE_SCREEN, Settings

log = logging.getLogger(__name__)


def _int_in(low: int, high: int) -> Callable[[Any], int]:
    return lambda value: max(low, min(high, int(value)))


def _float_in(low: float, high: float) -> Callable[[Any], float]:
    def check(value: Any) -> float:
        number = float(value)
        if number != number:  # NaN
            raise ValueError("не число")
        return max(low, min(high, number))

    return check


def _hhmm(value: Any) -> str:
    parsed = parse_hhmm(value)
    if parsed is None:
        raise ValueError(f"не время: {value!r}")
    return f"{parsed[0]:02d}:{parsed[1]:02d}"


def _weekdays(value: Any) -> list[int]:
    return sorted({int(day) for day in value if 0 <= int(day) <= 6})


_DISPLAY_KEY = re.compile(r"-?\d{1,6},-?\d{1,6},\d{1,6},\d{1,6}")


def _display_keys(value: Any) -> list[str]:
    return sorted({str(key) for key in value if _DISPLAY_KEY.fullmatch(str(key))})[:16]


# Настройки, которые интерфейс может менять напрямую, и проверка их значений.
_SETTERS: dict[str, Callable[[Any], Any]] = {
    "smoothing": _int_in(0, 95),
    "saturation": _int_in(50, 250),
    "min_brightness": _int_in(0, 40),
    "dark_dimming": _int_in(0, 100),
    "sensitivity": _int_in(30, 300),
    "effect_speed": _int_in(1, 100),
    "cycle_period": _int_in(5, 50),
    "screen_cycle": bool,
    "reverse": bool,
    "music_rainbow": bool,
    "off_on_exit": bool,
    "schedule_enabled": bool,
    "schedule_on": _hhmm,
    "schedule_off": _hhmm,
    "schedule_days": _weekdays,
    "schedule_sun": bool,
    "schedule_margin": _int_in(0, 90),
    "schedule_city": lambda value: str(value)[:40],
    "schedule_lat": _float_in(-90.0, 90.0),
    "schedule_lon": _float_in(-180.0, 180.0),
    "screen_displays": _display_keys,
}


class Engine:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self._lock = threading.RLock()        # режимы и настройки
        self._live_lock = threading.Lock()    # то, что показывает интерфейс
        self._status = ("searching", "")
        self._zones: list[dict] = []
        self._parts: list[list[int]] = []     # цвета трёх частей экрана для интерфейса
        self._part: int | None = None         # какая часть сейчас на лампах в режиме «по очереди»
        self._lamp: tuple[int, int, int] | None = None
        self._level = 0.0
        self._music_error = ""
        self._screen: ScreenSampler | None = None
        self._fader: ColorFader | None = None
        self._audio = None
        self._music_hue = 0.0
        self._music_time = time.monotonic()
        self._dirty = False
        self._schedule_state: bool | None = None
        self._calibrating = False             # идёт калибровка цветов: лампы показывают оттенок из интерфейса
        self._stop = threading.Event()
        self._keeper: threading.Thread | None = None
        self.link = LampLink(settings.device_name, self._on_status, settings.last_address)

    # ---- жизненный цикл --------------------------------------------------

    def start(self) -> None:
        self.link.start()
        with self._lock:
            self._check_schedule()
            self._apply()
        self._keeper = threading.Thread(target=self._housekeeping, name="engine-keeper", daemon=True)
        self._keeper.start()

    def shutdown(self) -> None:
        self._stop.set()
        with self._lock:
            self._stop_sources()
            if self.s.off_on_exit:
                self.link.set_power(False)
                self.link.flush(1.5)
            self.s.save()
        self.link.stop()

    # ---- команды интерфейса ----------------------------------------------

    def set_power(self, on: bool) -> None:
        with self._lock:
            self.s.power = bool(on)
            self._dirty = True
            self._apply()

    def toggle_power(self) -> None:
        self.set_power(not self.s.power)

    def set_mode(self, mode: int) -> None:
        mode = int(mode)
        if mode not in (MODE_COLOR, MODE_EFFECT, MODE_SCREEN, MODE_MUSIC):
            raise ValueError(f"нет такого режима: {mode}")
        with self._lock:
            if mode == self.s.mode:
                return
            self.s.mode = mode
            self._dirty = True
            self._apply()

    def set_brightness(self, percent: int) -> None:
        with self._lock:
            self.s.brightness = max(1, min(100, int(percent)))
            self._dirty = True
            if self.s.power and self.s.mode in (MODE_COLOR, MODE_EFFECT):
                self.link.set_brightness(self.s.brightness * 10)

    def set_color(self, hue: int, sat: int) -> None:
        with self._lock:
            self.s.hue = int(hue) % 360
            self.s.sat = max(0, min(1000, int(sat)))
            self._dirty = True
            if self.s.power and self.s.mode == MODE_COLOR:
                self._send_color()

    def set_effect(self, effect_id: int) -> None:
        effect_id = int(effect_id)
        if effect_id not in protocol.EFFECT_IDS:
            raise ValueError(f"нет такого эффекта: {effect_id}")
        with self._lock:
            self.s.effect = effect_id
            self._dirty = True
            if self.s.power and self.s.mode == MODE_EFFECT:
                self._send_effect()

    def set(self, key: str, value: Any) -> Any:
        """Изменить простую настройку; возвращает принятое значение."""
        check = _SETTERS.get(key)
        if check is None:
            raise KeyError(key)
        value = check(value)
        with self._lock:
            setattr(self.s, key, value)
            self._dirty = True
            if key in ("reverse", "effect_speed"):
                if self.s.power and self.s.mode == MODE_EFFECT:
                    self._send_effect()
            elif key == "screen_displays":
                if self.s.power and self.s.mode == MODE_SCREEN:
                    self._apply()
            elif key.startswith("schedule_"):
                self._schedule_state = None  # расписание изменилось — применить заново
                self._check_schedule()
        return value

    # ---- калибровка цветов -------------------------------------------------

    def calibration_start(self) -> dict:
        """Остановить подсветку и отдать лампы интерфейсу: он показывает образец на экране,
        а пользователь подбирает к нему оттенок ламп. Возвращает образцы."""
        with self._lock:
            self._stop_sources()
            self._calibrating = True
            self.link.set_power(True)
            self.link.set_brightness(self.s.brightness * 10)
            saved = self.s.screen_calibration
            if not screen.calibration_is_valid(saved):
                saved = [screen.NOT_CALIBRATED] * len(screen.CALIBRATION_PATCHES)
            patches = []
            for index, (name, _screen_hue) in enumerate(screen.CALIBRATION_PATCHES):
                patches.append(
                    {
                        "name": name,
                        "rgb": [round(x * 255) for x in screen.patch_color(index)],
                        # С чего начать подбор: то, что лампы показали бы на этом образце сейчас.
                        "hue": round(screen.calibrated_hue(screen.patch_hue(index), saved)) % 360,
                        "saved": saved[index],
                    }
                )
            return {"patches": patches}

    def calibration_show(self, hue: int) -> None:
        with self._lock:
            if not self._calibrating:
                return
            hue = int(hue) % 360
            self.link.set_color(hue, 1000)
            self._set_lamp_preview(hue / 360, 1.0, 1.0)

    def calibration_finish(self, values: Any = None) -> bool:
        """Закончить калибровку и вернуть лампы в обычный режим. values — результат; None — отмена.
        Возвращает, сохранён ли результат."""
        with self._lock:
            try:
                if values is None or not self._calibrating:
                    return False
                values = [int(x) for x in values]
                if not screen.calibration_is_valid(values):
                    raise ValueError("калибровка не того вида")
                self.s.screen_calibration = values
                self._dirty = True
                return True
            finally:
                self._apply()  # лампы возвращаются в обычный режим при любом исходе

    def calibration_reset(self) -> None:
        with self._lock:
            self.s.screen_calibration = []
            self._dirty = True

    def sun_info(self) -> dict:
        """Сегодняшние рассвет и закат для интерфейса, по местному времени."""
        now = datetime.now().astimezone()
        lat, lon, margin = self.s.schedule_lat, self.s.schedule_lon, self.s.schedule_margin
        sunrise, sunset = sun.sun_events(now.replace(hour=0, minute=0, second=0, microsecond=0), lat, lon)

        def clock(moment: datetime | None, shift_minutes: int = 0) -> str | None:
            if moment is None:
                return None
            # +30 секунд — округление до ближайшей минуты, а не отбрасывание секунд.
            return (moment + timedelta(minutes=shift_minutes, seconds=30)).astimezone().strftime("%H:%M")

        return {
            "sunrise": clock(sunrise),
            "sunset": clock(sunset),
            "on_at": clock(sunset, -margin),     # включение — за «запас» до заката
            "off_at": clock(sunrise, margin),    # выключение — через «запас» после рассвета
            "dark_now": sun.lamps_wanted(now, lat, lon, margin),
            "sun_down_all_day": sunrise is None and sunset is None and sun.sun_is_down(now, lat, lon),
        }

    def live(self) -> dict:
        """То, что меняется само: связь, цвет на лампах, участки экрана, уровень звука."""
        with self._live_lock:
            state, detail = self._status
            return {
                "status": state,
                "detail": detail,
                "power": self.s.power,
                "mode": self.s.mode,
                "calibrating": self._calibrating,
                "brightness": self.s.brightness,
                "lamp": list(self._lamp) if self._lamp else None,
                "zones": self._zones,
                "parts": self._parts,
                "part": self._part,
                "level": round(self._level, 3),
                "music_error": self._music_error,
            }

    # ---- внутреннее ------------------------------------------------------

    def _on_status(self, state: str, detail: str) -> None:
        with self._live_lock:
            changed = self._status[0] != state
            self._status = (state, detail)
        if changed:
            log.info("lamp link: %s %s", state, detail)
        if state == "connected" and detail and detail != self.s.last_address:
            self.s.last_address = detail
            self._dirty = True

    def _set_lamp_preview(self, hue: float, sat: float, value: float) -> None:
        r, g, b = colorsys.hsv_to_rgb(hue, sat, max(0.2, value))
        with self._live_lock:
            self._lamp = (round(r * 255), round(g * 255), round(b * 255))

    def _send_color(self) -> None:
        self.link.set_color(self.s.hue, self.s.sat)
        self._set_lamp_preview(self.s.hue / 360, self.s.sat / 1000, 1.0)

    def _send_effect(self) -> None:
        self.link.set_effect(self.s.effect, self.s.reverse, self.s.effect_speed)

    def _stop_sources(self) -> None:
        # Дожидаемся остановки потоков: иначе запоздавший кадр экрана перезапишет
        # на лампах цвет, который только что выбрали вручную.
        workers = [w for w in (self._screen, self._fader) if w is not None]
        for worker in workers:
            worker.stop()
        for worker in workers:
            worker.join(timeout=2.0)
        self._screen = self._fader = None
        if self._audio is not None:
            self._audio.stop()
            self._audio = None
        with self._live_lock:
            self._zones = []
            self._parts = []
            self._part = None
            self._level = 0.0

    def _apply(self) -> None:
        """Привести лампы и источники цвета в соответствие с настройками. Вызывать под _lock."""
        # Любое переключение (питание из трея, смена режима) заканчивает и калибровку:
        # интерфейс увидит это в live() и закроет её экран.
        self._calibrating = False
        self._stop_sources()
        with self._live_lock:
            self._music_error = ""
            self._lamp = None
        if not self.s.power:
            self.link.set_power(False)
            return
        self.link.set_power(True)
        mode = self.s.mode
        if mode == MODE_COLOR:
            self.link.set_brightness(self.s.brightness * 10)
            self._send_color()
        elif mode == MODE_EFFECT:
            self.link.set_brightness(self.s.brightness * 10)
            self._send_effect()
        elif mode == MODE_SCREEN:
            self._fader = ColorFader(lambda: self.s.smoothing, self._on_screen_color)
            self._fader.start()
            self._screen = ScreenSampler(self.s.screen_displays, self._on_screen_frame)
            self._screen.start()
        elif mode == MODE_MUSIC:
            self._start_music()

    def _start_music(self) -> None:
        try:
            from .audio import AudioSampler

            sampler = AudioSampler(lambda: self.s.sensitivity, self._on_audio)
            sampler.start()
            self._audio = sampler
            self._music_time = time.monotonic()
        except Exception as exc:
            log.exception("audio start failed")
            with self._live_lock:
                self._music_error = f"Не удалось открыть звук компьютера: {exc}"

    # Вызывается из потока захвата экрана: новый кадр — новая цель для плавного перехода.
    def _on_screen_frame(self, frame: ScreenFrame) -> None:
        active = None
        color, light, confidence = frame.color, frame.light, frame.confidence
        if self.s.screen_cycle:
            # Контроллер красит ленту только целиком, поэтому «три цвета» показываем по очереди:
            # лампы переливаются между цветами левой, средней и правой частей картинки.
            active = int(time.monotonic() / (self.s.cycle_period / 10)) % len(frame.parts)
            color, light, confidence = frame.parts[active]
        # Насыщенность и яркость считаем до сглаживания, чтобы плавно менялось уже то,
        # что реально увидят лампы: иначе усиление насыщенности превращает ровный
        # переход в рывок посередине.
        hue, sat = lamp_hue_sat(color, self.s.saturation, self.s.screen_calibration)
        sat *= confidence  # цвета на экране почти нет — лампы остаются нейтральными, а не ловят случайный оттенок
        value = scene_brightness(light, self.s.min_brightness, self.s.dark_dimming)
        fader = self._fader
        if fader is not None:
            # Яркость ведём в «глазной» шкале (корень): в темноте глаз замечает
            # гораздо меньшие ступени, и там шаги должны быть мельче.
            fader.set_target(np.array(colorsys.hsv_to_rgb(hue, sat, 1.0), np.float32), math.sqrt(value))
        with self._live_lock:
            self._zones = frame.monitors
            self._parts = [np.rint(np.clip(part[0], 0, 1) * 255).astype(int).tolist() for part in frame.parts]
            self._part = active

    # Вызывается из потока плавного перехода, чаще, чем приходят кадры.
    def _on_screen_color(self, color, eye_light: float) -> None:
        r, g, b = (min(1.0, max(0.0, float(x))) for x in color)
        hue, sat, _ = colorsys.rgb_to_hsv(r, g, b)
        value = min(1.0, max(0.0, eye_light)) ** 2
        # Именно пара «яркость 05 + цвет 04»: только она гасит и белый тоже (проверено на лампах).
        self.link.set_brightness(round(value * self.s.brightness * 10))
        self.link.set_color(round(hue * 360) % 360, round(sat * 1000))
        self._set_lamp_preview(hue, sat, value)

    # Вызывается из потоков звука.
    def _on_audio(self, level: float, beat: bool) -> None:
        now = time.monotonic()
        elapsed, self._music_time = now - self._music_time, now
        if self.s.music_rainbow:
            self._music_hue = (self._music_hue + elapsed * 20 + (50 if beat else 0)) % 360
            hue, sat = int(self._music_hue), 1000
        else:
            hue, sat = self.s.hue, self.s.sat
        self.link.set_brightness(round((0.05 + 0.95 * level) * self.s.brightness * 10))
        self.link.set_color(hue, sat)
        self._set_lamp_preview(hue / 360, sat / 1000, 0.25 + 0.75 * level)
        with self._live_lock:
            self._level = level

    def _check_schedule(self) -> None:
        """Включить или выключить лампы, если наступило время. Вызывать под _lock."""
        if not self.s.schedule_enabled:
            self._schedule_state = None
            return
        if self._calibrating:
            return  # не прерываем калибровку закатом; расписание сработает, когда она закончится
        if self.s.schedule_sun:
            wanted = sun.lamps_wanted(
                datetime.now().astimezone(), self.s.schedule_lat, self.s.schedule_lon, self.s.schedule_margin
            )
        else:
            wanted = desired_power(datetime.now(), self.s.schedule_on, self.s.schedule_off, self.s.schedule_days)
        if wanted is None or wanted == self._schedule_state:
            return
        # Срабатываем только на смену состояния: если лампы включили руками
        # в «выключенное» время, расписание не станет гасить их каждую минуту.
        self._schedule_state = wanted
        if wanted != self.s.power:
            log.info("schedule: power %s", "on" if wanted else "off")
            self.s.power = wanted
            self._dirty = True
            self._apply()

    def _housekeeping(self) -> None:
        while not self._stop.wait(2.0):
            try:
                with self._lock:
                    self._check_schedule()
                    if self._dirty:
                        self._dirty = False
                        self.s.save()
            except Exception:
                log.exception("housekeeping failed")

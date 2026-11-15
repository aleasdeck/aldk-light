"""Плавный переход цвета.

Экран снимается ~15 раз в секунду, и если слать на лампы каждый кадр как есть,
цвет меняется ступеньками. Здесь цвет «доезжает» до цели в собственном ритме,
чаще, чем приходят кадры, — переход получается непрерывным.

Сглаживание двухступенчатое: цель → промежуточное значение → выход. Один фильтр
срывается с места рывком (первый же шаг — самый большой, глаз видит толчок),
а два подряд дают плавный разгон и плавное торможение.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

# Самое долгое время перехода, секунды (ползунок «Плавность» в крайнем правом положении).
_SLOWEST = 3.0
_FASTEST = 0.03


def time_constant(smoothing: int) -> float:
    """Настройка «Плавность» 0..95 → характерное время перехода в секундах."""
    share = min(95, max(0, smoothing)) / 95
    return _FASTEST + share * share * (_SLOWEST - _FASTEST)


def smoothing_for(seconds: float) -> int:
    """Обратное к time_constant: какое положение ползунка даёт такое время."""
    share = math.sqrt(min(1.0, max(0.0, (seconds - _FASTEST) / (_SLOWEST - _FASTEST))))
    return round(share * 95)


def ease(current, target, elapsed: float, tau: float):
    """Сдвинуть current к target за elapsed секунд. Результат не зависит от частоты вызовов."""
    return current + (target - current) * (1.0 - math.exp(-elapsed / tau))


class ColorFader(threading.Thread):
    """Ведёт цвет и яркость к последней цели и отдаёт промежуточные значения в on_output."""

    def __init__(
        self,
        get_smoothing: Callable[[], int],
        on_output: Callable[[np.ndarray, float], None],
        rate: int = 30,  # цвет + яркость = два пакета на шаг, а контроллер принимает около 60 пакетов/с
    ) -> None:
        super().__init__(name="color-fader", daemon=True)
        self._get_smoothing = get_smoothing
        self._on_output = on_output
        self._period = 1.0 / rate
        self._lock = threading.Lock()
        self._target: tuple[np.ndarray, float] | None = None
        self._stop_event = threading.Event()

    def set_target(self, color: np.ndarray, light: float) -> None:
        with self._lock:
            self._target = (np.asarray(color, np.float32), float(light))

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        middle: tuple[np.ndarray, float] | None = None   # первая ступень сглаживания
        color: np.ndarray | None = None                  # вторая ступень — то, что уходит на лампы
        light = 0.0
        last = time.perf_counter()
        deadline = last
        while not self._stop_event.is_set():
            # time.sleep на Windows точнее, чем Event.wait: тот шагает по 15.6 мс.
            deadline = max(deadline + self._period, time.perf_counter())
            time.sleep(max(0.0, deadline - time.perf_counter()))
            if self._stop_event.is_set():
                break  # остановили, пока спали — больше ничего не отправляем
            now = time.perf_counter()
            elapsed, last = now - last, now
            with self._lock:
                target = self._target
            if target is None:
                continue
            try:
                if middle is None or color is None:
                    middle = target
                    color, light = target  # первый кадр — без разгона из чёрного
                else:
                    half = time_constant(self._get_smoothing()) / 2  # две ступени по половине времени
                    middle = (ease(middle[0], target[0], elapsed, half), ease(middle[1], target[1], elapsed, half))
                    color = ease(color, middle[0], elapsed, half)
                    light = ease(light, middle[1], elapsed, half)
                self._on_output(color, light)
            except Exception:
                log.exception("fader output failed")

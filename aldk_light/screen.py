"""Цвет экрана для режима подсветки под картинку.

Каждый монитор делится на сетку участков. Для участка считаются «яркий» цвет
(насыщенные точки весят больше) и вес — насколько участок вообще цветной.
Итоговый цвет ламп — преобладающий оттенок по всем участкам, а не среднее:
среднее от красного слева и синего справа даёт грязно-фиолетовый, которого на
экране нет.
"""

from __future__ import annotations

import colorsys
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

import mss
import numpy as np

log = logging.getLogger(__name__)

# Сколько точек по ширине брать с кадра: для цвета участков больше не нужно.
_SAMPLE_WIDTH = 96
# Как часто перечитывать список мониторов, секунды.
_RESCAN_SECONDS = 5.0
# На сколько секторов делится цветовой круг при поиске преобладающего оттенка.
_HUE_BINS = 12
# Вес «серой массы» участка: белый документ с одной цветной иконкой остаётся белым.
# Умножается на яркость участка, поэтому тёмная сцена со слабым оттенком (ночное небо)
# сохраняет свой цвет, а не превращается в белый.
_GREY_WEIGHT = 0.025
# Лента светит линейно, а цвета на экране закодированы с гаммой. Без пересчёта экранный
# зелёный с небольшой долей синего (0, 230, 128) уходит на ленту как «синий вполсилы»
# и выглядит бирюзовым, хотя на мониторе синего там всего ~20% по свету.
_LED_GAMMA = 2.4
# Средняя «цветная энергия» участка, ниже и выше которой цвету сцены не верим / верим.
# Замер на экране пользователя: тёмный серый рабочий стол — 0.002…0.005, обычная цветная картинка — 0.036…0.045.
_COLOUR_WEAK = 0.005
_COLOUR_CLEAR = 0.02


def _smoothstep(low: float, high: float, x: float) -> float:
    t = min(1.0, max(0.0, (x - low) / (high - low)))
    return t * t * (3 - 2 * t)


# Образцы для калибровки цветов: название и оттенок на экране (полная насыщенность и яркость).
# Пользователь подбирает к каждому оттенок ламп, который на глаз совпадает с экраном.
CALIBRATION_PATCHES: list[tuple[str, int]] = [
    ("красный", 0), ("оранжевый", 30), ("жёлтый", 60), ("салатовый", 90),
    ("зелёный", 120), ("зелёно-бирюзовый", 150), ("голубой", 180), ("лазурный", 210),
    ("синий", 240), ("фиолетовый", 270), ("пурпурный", 300), ("розовый", 330),
]
NOT_CALIBRATED = -1  # образец пропущен: для него остаётся расчётный оттенок


def patch_color(index: int) -> tuple[float, float, float]:
    """Цвет образца на экране, RGB 0..1."""
    return colorsys.hsv_to_rgb(CALIBRATION_PATCHES[index][1] / 360, 1.0, 1.0)


def _linear_hue(rgb) -> tuple[float, float]:
    """Расчётные оттенок (градусы) и насыщенность 0..1 цвета с экрана — до калибровки."""
    r, g, b = (min(1.0, max(0.0, float(x))) ** _LED_GAMMA for x in rgb)
    hue, sat, _ = colorsys.rgb_to_hsv(r, g, b)
    return hue * 360, sat


def patch_hue(index: int) -> float:
    """Расчётный оттенок образца в градусах: от него отсчитывается поправка калибровки."""
    return _linear_hue(patch_color(index))[0]


def calibration_is_valid(calibration) -> bool:
    return len(calibration) == len(CALIBRATION_PATCHES) and all(
        type(x) is int and NOT_CALIBRATED <= x < 360 for x in calibration
    )


def calibrated_hue(hue: float, calibration) -> float:
    """Расчётный оттенок (градусы) → оттенок ламп с учётом калибровки.

    calibration — по одному числу на образец: какой оттенок ламп пользователь счёл совпадающим
    с экраном (или NOT_CALIBRATED). Для каждого образца получается поправка «сколько градусов
    добавить», между соседними образцами поправка меняется равномерно — оттенок идёт без скачков.
    """
    if not calibration_is_valid(calibration):
        return hue % 360
    anchors = []  # (расчётный оттенок образца, поправка −180..180)
    for index, chosen in enumerate(calibration):
        at = patch_hue(index)
        offset = 0.0 if chosen == NOT_CALIBRATED else (chosen - at + 180) % 360 - 180
        anchors.append((at, offset))
    hue %= 360
    for (left, left_offset), (right, right_offset) in zip(anchors, anchors[1:] + anchors[:1]):
        width = (right - left) % 360
        along = (hue - left) % 360
        if along <= width:
            return (hue + left_offset + (right_offset - left_offset) * along / width) % 360
    return hue  # недостижимо: образцы покрывают весь круг


def lamp_hue_sat(rgb, saturation: int = 100, calibration=()) -> tuple[float, float]:
    """Цвет с экрана (RGB 0..1) → оттенок 0..1 и насыщенность 0..1 для ленты.

    saturation — усиление насыщенности в процентах, calibration — калибровка цветов
    (см. calibrated_hue). Она заменила ползунок «Синий канал»: тот убавлял синий одним
    числом на все цвета и, исправляя зелёный, уводил небесно-синий в зелёную сторону.
    """
    hue, sat = _linear_hue(rgb)
    return calibrated_hue(hue, calibration) / 360 % 1.0, min(1.0, sat * saturation / 100)


def scene_brightness(light: float, floor: int, dimming: int) -> float:
    """Мощность ламп 0..1 по яркости картинки 0..1.

    Глаз воспринимает свет нелинейно: 12% мощности светодиода выглядят примерно как 40%
    яркости, поэтому «чуть убавить» на тёмной сцене не видно вовсе (проверено на лампах
    пользователя: 19% → 12% он не заметил). Чтобы лампа выглядела такой же тёмной, как
    картинка, мощность должна падать как степень ~2.4 от яркости экрана.

    dimming 0..100 задаёт эту степень: 0 — мощность пропорциональна яркости экрана
    (лампы на тёмном заметно светят), 100 — лампы темнеют наравне с картинкой.
    Светлые сцены остаются на полной яркости при любом значении.

    floor 0..40 — минимальная яркость «на глаз»: мощность берётся как квадрат,
    чтобы у нуля ползунок работал мелкими шагами (10 → 1% мощности, 40 → 16%).
    """
    base = min(1.0, max(0.0, light) * 1.3)
    value = base ** (1.0 + 1.4 * min(100, max(0, dimming)) / 100)
    low = (min(100, max(0, floor)) / 100) ** 2
    return low + (1.0 - low) * value
# Новый оттенок должен быть на столько сильнее текущего, чтобы лампы переключились:
# иначе при двух равных по силе цветах подсветка мечется между ними.
_SWITCH_MARGIN = 1.2


def grid_for(width: int, height: int) -> tuple[int, int]:
    """Сетка участков (столбцы, строки) под ориентацию монитора."""
    return (4, 3) if width >= height else (3, 4)


def _open_capture():
    return (getattr(mss, "MSS", None) or mss.mss)()  # в новых версиях mss класс называется MSS


def monitor_key(mon: dict) -> str:
    """Постоянное имя монитора: его положение и размер на рабочем столе.

    Порядковый номер для этого не годится: после сна или переподключения Windows
    перенумеровывает мониторы, и «монитор 1» оказывается другим экраном.
    """
    return f"{mon['left']},{mon['top']},{mon['width']},{mon['height']}"


def choose_monitors(monitors: list[dict], keys: list[str]) -> list[dict]:
    """Выбранные мониторы; если ни одного из выбранных сейчас нет (или выбор пуст) — все."""
    wanted = set(keys)
    return [mon for mon in monitors if monitor_key(mon) in wanted] or list(monitors)


def describe_monitors() -> list[dict]:
    """Мониторы с положением на рабочем столе и сеткой участков — для интерфейса."""
    with _open_capture() as sct:
        result = []
        for mon in sct.monitors[1:]:  # нулевой элемент — весь рабочий стол целиком
            cols, rows = grid_for(mon["width"], mon["height"])
            result.append(
                {
                    "key": monitor_key(mon),
                    "left": mon["left"],
                    "top": mon["top"],
                    "width": mon["width"],
                    "height": mon["height"],
                    # Основной монитор в Windows всегда начинается в точке (0, 0).
                    "primary": mon["left"] == 0 and mon["top"] == 0,
                    "cols": cols,
                    "rows": rows,
                }
            )
        return result


@dataclass
class ZoneStats:
    mean: np.ndarray    # N×3, обычный средний цвет участка
    vivid: np.ndarray   # N×3, цвет участка с перевесом ярких насыщенных точек
    weight: np.ndarray  # N, насколько участок цветной (0 — серый или чёрный)
    light: np.ndarray   # N, общая яркость участка


def zone_stats(rgb: np.ndarray, cols: int, rows: int) -> ZoneStats:
    """Разбить кадр H×W×3 (значения 0..1) на cols×rows участков, слева направо, сверху вниз."""
    height, width, _ = rgb.shape
    brightest = rgb.max(axis=2)
    colourful = (brightest - rgb.min(axis=2)) * brightest
    ys = np.linspace(0, height, rows + 1).astype(int)
    xs = np.linspace(0, width, cols + 1).astype(int)
    count = cols * rows
    mean = np.zeros((count, 3), np.float32)
    vivid = np.zeros((count, 3), np.float32)
    weight = np.zeros(count, np.float32)
    light = np.zeros(count, np.float32)
    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            area = (slice(ys[row], ys[row + 1]), slice(xs[col], xs[col + 1]))
            block, block_weight = rgb[area], colourful[area]
            mean[index] = block.mean(axis=(0, 1))
            total = float(block_weight.sum())
            if total > 1e-6:
                vivid[index] = (block * block_weight[..., None]).sum(axis=(0, 1)) / total
            else:
                vivid[index] = mean[index]
            weight[index] = block_weight.mean()
            light[index] = brightest[area].mean()
    return ZoneStats(mean, vivid, weight, light)


def dominant_color(stats: ZoneStats, prefer_bin: int | None = None) -> tuple[np.ndarray, int]:
    """Преобладающий цвет по участкам и сектор цветового круга, в который он попал."""
    hues = np.array([colorsys.rgb_to_hsv(*(float(x) for x in color))[0] for color in stats.vivid])
    bins = (hues * _HUE_BINS).astype(int) % _HUE_BINS
    score = np.bincount(bins, weights=stats.weight, minlength=_HUE_BINS)
    # Соседние сектора — почти тот же цвет, считаем их вместе.
    spread = score + 0.5 * (np.roll(score, 1) + np.roll(score, -1))
    best = int(spread.argmax())
    if prefer_bin is not None and spread[prefer_bin] * _SWITCH_MARGIN >= spread[best]:
        best = prefer_bin
    distance = np.abs(bins - best)
    distance = np.minimum(distance, _HUE_BINS - distance)
    member_weight = np.where(distance <= 1, stats.weight, 0.0)
    grey_weight = _GREY_WEIGHT * stats.light
    total = float(member_weight.sum() + grey_weight.sum())
    if total < 1e-9:  # полностью чёрный кадр
        return stats.mean.mean(axis=0).astype(np.float32), best
    color = (
        (stats.vivid * member_weight[:, None]).sum(axis=0) + (stats.mean * grey_weight[:, None]).sum(axis=0)
    ) / total
    return color.astype(np.float32), best


def colour_confidence(stats: ZoneStats) -> float:
    """Насколько цвету сцены можно верить, 0..1. На это число умножается насыщенность ламп.

    На тёмном сером экране цветных точек доли процента (подсветка текста, значки),
    и их случайный оттенок нельзя выдавать за цвет сцены — лампы уходили в жёлтый.
    Верим в двух случаях: цветного света на экране много, или вся картинка окрашена
    в один тон (ночное небо: темно, но честно синее).
    """
    plain = (stats.mean * stats.light[:, None]).sum(axis=0) / max(float(stats.light.sum()), 1e-9)
    peak = float(plain.max())
    plain_saturation = (peak - float(plain.min())) / peak if peak > 1e-4 else 0.0
    return max(
        _smoothstep(_COLOUR_WEAK, _COLOUR_CLEAR, float(stats.weight.mean())),
        _smoothstep(0.15, 0.40, plain_saturation),
    )


def _merge(parts: list[ZoneStats]) -> ZoneStats:
    return ZoneStats(
        np.concatenate([p.mean for p in parts]),
        np.concatenate([p.vivid for p in parts]),
        np.concatenate([p.weight for p in parts]),
        np.concatenate([p.light for p in parts]),
    )


PARTS = 3  # на сколько частей слева направо делится картинка для режима «по очереди»


def split_into_parts(centres: np.ndarray, left: float, right: float) -> np.ndarray:
    """К какой из PARTS вертикальных полос рабочего стола относится участок с центром centres."""
    share = (centres - left) / max(1.0, right - left)
    return np.clip((share * PARTS).astype(int), 0, PARTS - 1)


def _subset(stats: ZoneStats, mask: np.ndarray) -> ZoneStats:
    return ZoneStats(stats.mean[mask], stats.vivid[mask], stats.weight[mask], stats.light[mask])


@dataclass
class ScreenFrame:
    color: np.ndarray      # цвет для ламп по этому кадру, RGB 0..1 (сглаживает его ColorFader)
    light: float           # общая яркость картинки 0..1
    monitors: list[dict]   # [{"key": "0,0,2560,1440", "colors": [[r, g, b], …]}] — цвета участков 0..255
    # (цвет, яркость, доверие к цвету) левой, средней и правой частей картинки
    parts: list[tuple[np.ndarray, float, float]]
    confidence: float = 1.0  # доверие к общему цвету, 0..1


class ScreenSampler(threading.Thread):
    """Снимает выбранные мониторы и отдаёт кадры в on_frame."""

    def __init__(
        self,
        display_keys: list[str],
        on_frame: Callable[[ScreenFrame], None],
        fps: int = 20,
    ) -> None:
        super().__init__(name="screen-sampler", daemon=True)
        self._display_keys = list(display_keys)
        self._on_frame = on_frame
        self._period = 1.0 / fps
        self._stop_event = threading.Event()
        self._hue_bin: int | None = None
        self._part_bins: list[int | None] = [None] * PARTS
        self._captured: list[str] = []  # что снимаем сейчас — чтобы писать в журнал только изменения

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        # Захват идёт короткими сеансами: каждые несколько секунд список мониторов
        # перечитывается заново. Так подсветка переживает сон компьютера, отключение
        # монитора и смену разрешения — раньше после этого она снимала не тот экран.
        while not self._stop_event.is_set():
            try:
                self._session()
            except Exception:
                log.exception("screen capture failed, retrying")
                self._stop_event.wait(1.0)

    def _session(self) -> None:
        with _open_capture() as sct:
            chosen = choose_monitors(sct.monitors[1:], self._display_keys)
            regions = [
                (monitor_key(mon), mon, grid_for(mon["width"], mon["height"]), max(1, mon["width"] // _SAMPLE_WIDTH))
                for mon in chosen
            ]
            captured = [key for key, *_ in regions]
            if captured != self._captured:
                self._captured = captured
                log.info("capturing monitors: %s", "; ".join(captured))

            # Заранее раскладываем участки всех мониторов по трём полосам рабочего стола.
            centres = []
            for _index, mon, (cols, rows), _step in regions:
                row = mon["left"] + (np.arange(cols) + 0.5) / cols * mon["width"]
                centres.append(np.tile(row, rows))
            part_of_zone = split_into_parts(
                np.concatenate(centres),
                min(mon["left"] for _i, mon, _g, _s in regions),
                max(mon["left"] + mon["width"] for _i, mon, _g, _s in regions),
            )

            hue_bin, part_bins = self._hue_bin, self._part_bins
            rescan_at = time.monotonic() + _RESCAN_SECONDS
            while not self._stop_event.is_set() and time.monotonic() < rescan_at:
                started = time.perf_counter()
                parts: list[ZoneStats] = []
                previews: list[dict] = []
                for index, mon, (cols, rows), step in regions:
                    shot = sct.grab(mon)
                    frame = np.frombuffer(shot.raw, np.uint8).reshape(shot.height, shot.width, 4)
                    rgb = frame[::step, ::step, 2::-1].astype(np.float32) * (1.0 / 255.0)  # BGRA → RGB
                    stats = zone_stats(rgb, cols, rows)
                    parts.append(stats)
                    previews.append({"key": index, "colors": np.rint(stats.mean * 255).astype(int).tolist()})

                merged = _merge(parts)
                color, hue_bin = dominant_color(merged, hue_bin)
                self._hue_bin = hue_bin  # текущий оттенок переживает перечитывание мониторов
                light = float(merged.light.mean())
                confidence = colour_confidence(merged)
                parts = []
                for part in range(PARTS):
                    mask = part_of_zone == part
                    if not mask.any():       # узкий одиночный монитор: полоса пуста — берём общий цвет
                        parts.append((color, light, confidence))
                        continue
                    piece = _subset(merged, mask)
                    part_color, part_bins[part] = dominant_color(piece, part_bins[part])
                    parts.append((part_color, float(piece.light.mean()), colour_confidence(piece)))
                self._on_frame(ScreenFrame(color, light, previews, parts, confidence))

                self._stop_event.wait(max(0.0, self._period - (time.perf_counter() - started)))

"""Цвет экрана для режима подсветки под картинку.

Кадр уменьшается, и каждая его точка голосует за свой оттенок: чем она ярче и
насыщеннее, тем весомее голос. Итоговый цвет ламп — преобладающий оттенок, а не
среднее по экрану: среднее от красного и синего даёт грязно-фиолетовый, которого
на картинке нет. Те же точки и голоса отдаются интерфейсу: на схеме видно, что
снимается и какой оттенок победил.
"""

from __future__ import annotations

import base64
import colorsys
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import mss
import numpy as np

log = logging.getLogger(__name__)

# Сколько точек по ширине брать с кадра: для выбора цвета больше не нужно.
_SAMPLE_WIDTH = 96
# Как часто перечитывать список мониторов, секунды.
_RESCAN_SECONDS = 5.0
# На сколько секторов делится цветовой круг при поиске преобладающего оттенка.
_HUE_BINS = 12
# Вес «серой массы» картинки: белый документ с одной цветной иконкой остаётся белым.
# Умножается на яркость точки, поэтому тёмная сцена со слабым оттенком (ночное небо)
# сохраняет свой цвет, а не превращается в белый.
_GREY_WEIGHT = 0.025
# Лента светит линейно, а цвета на экране закодированы с гаммой. Без пересчёта экранный
# зелёный с небольшой долей синего (0, 230, 128) уходит на ленту как «синий вполсилы»
# и выглядит бирюзовым, хотя на мониторе синего там всего ~20% по свету.
_LED_GAMMA = 2.4
# Средняя «цветная энергия» картинки, ниже и выше которой цвету сцены не верим / верим.
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
    """Мониторы с положением на рабочем столе — для интерфейса."""
    with _open_capture() as sct:
        result = []
        for mon in sct.monitors[1:]:  # нулевой элемент — весь рабочий стол целиком
            result.append(
                {
                    "key": monitor_key(mon),
                    "left": mon["left"],
                    "top": mon["top"],
                    "width": mon["width"],
                    "height": mon["height"],
                    # Основной монитор в Windows всегда начинается в точке (0, 0).
                    "primary": mon["left"] == 0 and mon["top"] == 0,
                }
            )
        return result


@dataclass
class PixelStats:
    """Точки кадра (после уменьшения), по которым выбирается цвет ламп."""

    rgb: np.ndarray     # N×3, цвет точки, 0..1
    weight: np.ndarray  # N, сколько «цветного света» даёт точка: цветность × яркость × площадь
    light: np.ndarray   # N, яркость точки (самый сильный канал)
    area: np.ndarray    # N, сколько настоящих пикселей экрана стоит за точкой
    bins: np.ndarray    # N, сектор цветового круга, в который попал оттенок точки


def pixel_stats(rgb: np.ndarray, area: float = 1.0) -> PixelStats:
    """Разобрать кадр H×W×3 (значения 0..1) на точки. area — площадь экрана за одной точкой:
    мониторы уменьшаются с разным шагом, а голос в выборе цвета должен зависеть от настоящей
    площади на экране, а не от числа оставшихся точек."""
    points = rgb.reshape(-1, 3).astype(np.float32)
    red, green, blue = points[:, 0], points[:, 1], points[:, 2]
    light = points.max(axis=1)
    spread = light - points.min(axis=1)
    safe = np.where(spread > 0, spread, 1.0)
    sixth = np.where(
        light == red, ((green - blue) / safe) % 6.0,
        np.where(light == green, (blue - red) / safe + 2.0, (red - green) / safe + 4.0),
    )  # оттенок в шестых долях круга; у серых точек он не важен — их вес равен нулю
    bins = (sixth / 6.0 * _HUE_BINS).astype(int) % _HUE_BINS
    areas = np.full(len(points), float(area), np.float32)
    # Вес — цветность × яркость: яркая насыщенная точка говорит о цвете сцены больше, чем тусклая.
    return PixelStats(points, spread * light * areas, light, areas, bins)


def dominant_color(stats: PixelStats, prefer_bin: int | None = None) -> tuple[np.ndarray, int]:
    """Преобладающий цвет картинки и сектор цветового круга, в который он попал.

    Каждая точка голосует за свой оттенок. Раньше голосовали участки экрана, и участок, в котором
    было два цвета, отдавал голос за их смесь: синяя сцена с красными объектами по всему экрану
    давала на лампах фиолетовый, которого на картинке нет.
    """
    score = np.bincount(stats.bins, weights=stats.weight, minlength=_HUE_BINS)
    # Соседние сектора — почти тот же цвет, считаем их вместе.
    spread = score + 0.5 * (np.roll(score, 1) + np.roll(score, -1))
    best = int(spread.argmax())
    if prefer_bin is not None and spread[prefer_bin] * _SWITCH_MARGIN >= spread[best]:
        best = prefer_bin
    distance = np.abs(stats.bins - best)
    distance = np.minimum(distance, _HUE_BINS - distance)
    member_weight = np.where(distance <= 1, stats.weight, 0.0)
    grey_weight = _GREY_WEIGHT * stats.light * stats.area
    total = float(member_weight.sum() + grey_weight.sum())
    if total < 1e-9:  # полностью чёрный кадр
        return np.zeros(3, np.float32), best
    color = (stats.rgb * (member_weight + grey_weight)[:, None]).sum(axis=0) / total
    return color.astype(np.float32), best


def scene_light(stats: PixelStats) -> float:
    """Средняя яркость картинки, 0..1."""
    return float((stats.light * stats.area).sum() / max(float(stats.area.sum()), 1e-9))


def colour_confidence(stats: PixelStats) -> float:
    """Насколько цвету сцены можно верить, 0..1. На это число умножается насыщенность ламп.

    На тёмном сером экране цветных точек доли процента (подсветка текста, значки),
    и их случайный оттенок нельзя выдавать за цвет сцены — лампы уходили в жёлтый.
    Верим в двух случаях: цветного света на экране много, или вся картинка окрашена
    в один тон (ночное небо: темно, но честно синее).
    """
    # Обычный средний цвет картинки, без перевеса ярких точек: иначе горстка яркого жёлтого
    # текста на тёмно-сером фоне сама выглядела бы как «вся картинка жёлтая».
    plain = (stats.rgb * stats.area[:, None]).sum(axis=0) / max(float(stats.area.sum()), 1e-9)
    peak = float(plain.max())
    plain_saturation = (peak - float(plain.min())) / peak if peak > 1e-4 else 0.0
    colourful = float(stats.weight.sum() / max(float(stats.area.sum()), 1e-9))
    return max(
        _smoothstep(_COLOUR_WEAK, _COLOUR_CLEAR, colourful),
        _smoothstep(0.15, 0.40, plain_saturation),
    )


# ---- то же самое для схемы в интерфейсе: какие точки и чьи голоса дали цвет ламп ----

# Точки схемы берутся через одну: для картинки шириной в пару сотен пикселей больше не нужно.
_VIEW_STEP = 2
# Насколько заметна на схеме точка, не повлиявшая на цвет (0..255), и при какой «цветности»
# повлиявшая точка показывается в полную силу.
_VIEW_DIM = 70
_VIEW_FULL = 0.25


def vote_shares(stats: PixelStats) -> list[float]:
    """Доля голосов за каждый сектор цветового круга, в сумме 1 (или одни нули, если цвета нет)."""
    score = np.bincount(stats.bins, weights=stats.weight, minlength=_HUE_BINS)
    total = float(score.sum())
    return [round(float(x) / total, 4) if total > 1e-9 else 0.0 for x in score]


def point_alpha(stats: PixelStats, winners) -> np.ndarray:
    """Заметность каждой точки на схеме, 0..255: чем больше точка дала цвету ламп, тем она ярче.

    winners — победивший сектор: одно число на всю картинку или по числу на каждую точку
    (в режиме «по очереди» у каждой трети экрана свой победитель).
    """
    distance = np.abs(stats.bins - winners)
    distance = np.minimum(distance, _HUE_BINS - distance)
    colourful = stats.weight / np.maximum(stats.area, 1e-9)
    strength = np.sqrt(np.clip(colourful / _VIEW_FULL, 0.0, 1.0)) * (distance <= 1)
    return (_VIEW_DIM + (255 - _VIEW_DIM) * strength).astype(np.uint8)


def points_view(key: str, rgb: np.ndarray, alpha: np.ndarray) -> dict:
    """Картинка монитора для схемы: точки RGBA подряд по строкам, в base64 — так её дешевле
    всего передать в интерфейс и сразу положить на холст."""
    height, width, _ = rgb.shape
    rgba = np.dstack([np.rint(rgb * 255).astype(np.uint8), alpha.reshape(height, width)])[::_VIEW_STEP, ::_VIEW_STEP]
    return {
        "key": key,
        "width": rgba.shape[1],
        "height": rgba.shape[0],
        "pixels": base64.b64encode(np.ascontiguousarray(rgba).tobytes()).decode("ascii"),
    }


def _merge(parts: list[PixelStats]) -> PixelStats:
    return PixelStats(
        np.concatenate([p.rgb for p in parts]),
        np.concatenate([p.weight for p in parts]),
        np.concatenate([p.light for p in parts]),
        np.concatenate([p.area for p in parts]),
        np.concatenate([p.bins for p in parts]),
    )


PARTS = 3  # на сколько частей слева направо делится картинка для режима «по очереди»


def split_into_parts(centres: np.ndarray, left: float, right: float) -> np.ndarray:
    """К какой из PARTS вертикальных полос рабочего стола относится участок с центром centres."""
    share = (centres - left) / max(1.0, right - left)
    return np.clip((share * PARTS).astype(int), 0, PARTS - 1)


def _subset(stats: PixelStats, mask: np.ndarray) -> PixelStats:
    return PixelStats(stats.rgb[mask], stats.weight[mask], stats.light[mask], stats.area[mask], stats.bins[mask])


@dataclass
class ScreenFrame:
    color: np.ndarray      # цвет для ламп по этому кадру, RGB 0..1 (сглаживает его ColorFader)
    light: float           # общая яркость картинки 0..1
    views: list[dict]      # картинки мониторов для схемы в интерфейсе, см. points_view
    # (цвет, яркость, доверие к цвету) левой, средней и правой частей картинки
    parts: list[tuple[np.ndarray, float, float]]
    confidence: float = 1.0  # доверие к общему цвету, 0..1
    # Голоса по секторам цветового круга и победивший сектор: сначала вся картинка, потом три части.
    votes: list[list[float]] = field(default_factory=list)
    winners: list[int] = field(default_factory=list)


class ScreenSampler(threading.Thread):
    """Снимает выбранные мониторы и отдаёт кадры в on_frame."""

    def __init__(
        self,
        display_keys: list[str],
        on_frame: Callable[[ScreenFrame], None],
        fps: int = 20,
        split_view: Callable[[], bool] = lambda: False,
    ) -> None:
        super().__init__(name="screen-sampler", daemon=True)
        self._display_keys = list(display_keys)
        self._on_frame = on_frame
        # Для схемы: подсвечивать точки победителя всей картинки или каждой трети своего («по очереди»).
        self._split_view = split_view
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
            regions = [(monitor_key(mon), mon, max(1, mon["width"] // _SAMPLE_WIDTH)) for mon in chosen]
            captured = [key for key, _mon, _step in regions]
            if captured != self._captured:
                self._captured = captured
                log.info("capturing monitors: %s", "; ".join(captured))

            # Рабочий стол делится на три полосы слева направо — по ним считаются «три части».
            desk_left = min(mon["left"] for _key, mon, _step in regions)
            desk_right = max(mon["left"] + mon["width"] for _key, mon, _step in regions)

            hue_bin, part_bins = self._hue_bin, self._part_bins
            rescan_at = time.monotonic() + _RESCAN_SECONDS
            while not self._stop_event.is_set() and time.monotonic() < rescan_at:
                started = time.perf_counter()
                pieces: list[PixelStats] = []
                part_of_point: list[np.ndarray] = []
                pictures: list[tuple[str, np.ndarray]] = []
                for key, mon, step in regions:
                    shot = sct.grab(mon)
                    frame = np.frombuffer(shot.raw, np.uint8).reshape(shot.height, shot.width, 4)
                    rgb = frame[::step, ::step, 2::-1].astype(np.float32) * (1.0 / 255.0)  # BGRA → RGB
                    pieces.append(pixel_stats(rgb, area=step * step))
                    # К какой полосе рабочего стола относится каждая точка: по её столбцу.
                    centres = mon["left"] + (np.arange(rgb.shape[1]) + 0.5) * step
                    part_of_point.append(np.tile(split_into_parts(centres, desk_left, desk_right), rgb.shape[0]))
                    pictures.append((key, rgb))

                merged = _merge(pieces)
                part_index = np.concatenate(part_of_point)
                color, hue_bin = dominant_color(merged, hue_bin)
                self._hue_bin = hue_bin  # текущий оттенок переживает перечитывание мониторов
                light = scene_light(merged)
                confidence = colour_confidence(merged)
                parts = []
                votes, winners = [vote_shares(merged)], [hue_bin]
                for part in range(PARTS):
                    mask = part_index == part
                    if not mask.any():       # на всякий случай: пустая полоса берёт общий цвет
                        parts.append((color, light, confidence))
                        votes.append(votes[0])
                        winners.append(hue_bin)
                        continue
                    piece = _subset(merged, mask)
                    part_color, part_bins[part] = dominant_color(piece, part_bins[part])
                    parts.append((part_color, scene_light(piece), colour_confidence(piece)))
                    votes.append(vote_shares(piece))
                    winners.append(part_bins[part])

                # Схема для интерфейса: те же точки, ярче — те, что дали цвет лампам.
                wanted = np.array(winners[1:])[part_index] if self._split_view() else hue_bin
                alpha = point_alpha(merged, wanted)
                views, offset = [], 0
                for key, rgb in pictures:
                    count = rgb.shape[0] * rgb.shape[1]
                    views.append(points_view(key, rgb, alpha[offset:offset + count]))
                    offset += count
                self._on_frame(ScreenFrame(color, light, views, parts, confidence, votes, winners))

                self._stop_event.wait(max(0.0, self._period - (time.perf_counter() - started)))

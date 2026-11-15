"""Положение солнца: когда снаружи темно, во сколько сегодня рассвет и закат.

Считается по формулам NOAA, без сети и без запросов о местоположении:
координаты берутся из настроек (город из списка или введённые вручную).
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timedelta, timezone

# Высота центра солнца в момент видимого восхода/захода: рефракция плюс радиус диска.
_HORIZON = -0.833

# (город, широта, долгота, смещение от UTC в часах). Внутри одного пояса первым идёт
# город, который подставляется по умолчанию при этом часовом поясе системы.
CITIES: list[tuple[str, float, float, float]] = [
    ("Калининград", 54.71, 20.51, 2),
    ("Киев", 50.45, 30.52, 2),
    ("Рига", 56.95, 24.11, 2),
    ("Вильнюс", 54.69, 25.28, 2),
    ("Таллин", 59.44, 24.75, 2),
    ("Кишинёв", 47.01, 28.86, 2),
    ("Тель-Авив", 32.08, 34.78, 2),
    ("Москва", 55.76, 37.62, 3),
    ("Санкт-Петербург", 59.94, 30.31, 3),
    ("Нижний Новгород", 56.33, 44.00, 3),
    ("Казань", 55.79, 49.12, 3),
    ("Ростов-на-Дону", 47.23, 39.72, 3),
    ("Краснодар", 45.04, 38.98, 3),
    ("Сочи", 43.59, 39.73, 3),
    ("Воронеж", 51.67, 39.18, 3),
    ("Волгоград", 48.71, 44.51, 3),
    ("Архангельск", 64.54, 40.54, 3),
    ("Мурманск", 68.97, 33.08, 3),
    ("Минск", 53.90, 27.56, 3),
    ("Стамбул", 41.01, 28.98, 3),
    ("Самара", 53.20, 50.15, 4),
    ("Саратов", 51.53, 46.03, 4),
    ("Ижевск", 56.85, 53.20, 4),
    ("Ульяновск", 54.31, 48.40, 4),
    ("Тбилиси", 41.72, 44.79, 4),
    ("Ереван", 40.18, 44.51, 4),
    ("Баку", 40.41, 49.87, 4),
    ("Дубай", 25.20, 55.27, 4),
    ("Екатеринбург", 56.84, 60.61, 5),
    ("Челябинск", 55.16, 61.40, 5),
    ("Пермь", 58.01, 56.25, 5),
    ("Уфа", 54.74, 55.97, 5),
    ("Тюмень", 57.15, 65.53, 5),
    ("Алматы", 43.24, 76.89, 5),
    ("Астана", 51.17, 71.43, 5),
    ("Ташкент", 41.31, 69.24, 5),
    ("Омск", 54.99, 73.37, 6),
    ("Бишкек", 42.87, 74.59, 6),
    ("Новосибирск", 55.03, 82.92, 7),
    ("Красноярск", 56.01, 92.87, 7),
    ("Томск", 56.48, 84.95, 7),
    ("Барнаул", 53.35, 83.78, 7),
    ("Кемерово", 55.35, 86.09, 7),
    ("Бангкок", 13.76, 100.50, 7),
    ("Иркутск", 52.29, 104.30, 8),
    ("Улан-Удэ", 51.83, 107.58, 8),
    ("Чита", 52.03, 113.50, 9),
    ("Якутск", 62.03, 129.73, 9),
    ("Благовещенск", 50.29, 127.53, 9),
    ("Владивосток", 43.12, 131.89, 10),
    ("Хабаровск", 48.48, 135.08, 10),
    ("Южно-Сахалинск", 46.96, 142.73, 11),
    ("Магадан", 59.56, 150.80, 11),
    ("Петропавловск-Камчатский", 53.04, 158.65, 12),
    ("Варшава", 52.23, 21.01, 1),
    ("Берлин", 52.52, 13.40, 1),
    ("Лондон", 51.51, -0.13, 0),
    ("Нью-Йорк", 40.71, -74.01, -5),
    ("Лос-Анджелес", 34.05, -118.24, -8),
]


def default_location() -> tuple[str, float, float]:
    """Город для первого запуска: первый в списке с тем же поясом, что у системы."""
    offset = -time.timezone / 3600  # зимнее (стандартное) смещение системы
    for name, lat, lon, city_offset in CITIES:
        if city_offset == offset:
            return name, lat, lon
    return "Москва", 55.76, 37.62


def elevation(when: datetime, lat: float, lon: float) -> float:
    """Высота солнца над горизонтом в градусах. when — момент с часовым поясом."""
    utc = when.astimezone(timezone.utc)
    day_minutes = utc.hour * 60 + utc.minute + utc.second / 60
    julian_day = utc.toordinal() + 1721424.5 + day_minutes / 1440
    t = (julian_day - 2451545.0) / 36525.0  # юлианские столетия от эпохи J2000

    mean_long = math.radians((280.46646 + t * (36000.76983 + 0.0003032 * t)) % 360)
    anomaly = math.radians(357.52911 + t * (35999.05029 - 0.0001537 * t))
    eccentricity = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    centre = (
        math.sin(anomaly) * (1.914602 - t * (0.004817 + 0.000014 * t))
        + math.sin(2 * anomaly) * (0.019993 - 0.000101 * t)
        + math.sin(3 * anomaly) * 0.000289
    )
    omega = math.radians(125.04 - 1934.136 * t)
    apparent_long = math.radians(math.degrees(mean_long) + centre - 0.00569 - 0.00478 * math.sin(omega))
    obliquity = math.radians(
        23 + (26 + (21.448 - t * (46.815 + t * (0.00059 - 0.001813 * t))) / 60) / 60 + 0.00256 * math.cos(omega)
    )
    declination = math.asin(math.sin(obliquity) * math.sin(apparent_long))

    y = math.tan(obliquity / 2) ** 2
    equation_of_time = 4 * math.degrees(  # минуты: насколько истинный полдень расходится со средним
        y * math.sin(2 * mean_long)
        - 2 * eccentricity * math.sin(anomaly)
        + 4 * eccentricity * y * math.sin(anomaly) * math.cos(2 * mean_long)
        - 0.5 * y * y * math.sin(4 * mean_long)
        - 1.25 * eccentricity * eccentricity * math.sin(2 * anomaly)
    )
    solar_minutes = (day_minutes + equation_of_time + 4 * lon) % 1440
    hour_angle = math.radians(solar_minutes / 4 - 180)
    latitude = math.radians(lat)
    cos_zenith = math.sin(latitude) * math.sin(declination) + math.cos(latitude) * math.cos(declination) * math.cos(hour_angle)
    return 90 - math.degrees(math.acos(max(-1.0, min(1.0, cos_zenith))))


def sun_is_down(when: datetime, lat: float, lon: float) -> bool:
    return elevation(when, lat, lon) < _HORIZON


def lamps_wanted(when: datetime, lat: float, lon: float, margin_minutes: int = 0) -> bool:
    """Должны ли лампы гореть: от «заката минус запас» до «рассвета плюс запас».

    Проверяем солнце на «запас» раньше и позже текущего момента: если хотя бы там оно
    под горизонтом — уже (или ещё) сумерки. Так полярная ночь и полярный день
    получаются сами собой, без особых случаев.
    """
    margin = timedelta(minutes=max(0, margin_minutes))
    return sun_is_down(when - margin, lat, lon) or sun_is_down(when + margin, lat, lon)


def _crossing(before: datetime, after: datetime, lat: float, lon: float) -> datetime:
    """Момент между before и after, когда солнце пересекает горизонт (с точностью до секунд)."""
    down_before = sun_is_down(before, lat, lon)
    while (after - before).total_seconds() > 1:
        middle = before + (after - before) / 2
        if sun_is_down(middle, lat, lon) == down_before:
            before = middle
        else:
            after = middle
    return after


def sun_events(day: datetime, lat: float, lon: float) -> tuple[datetime | None, datetime | None]:
    """Рассвет и закат в сутках, начинающихся с момента day (обычно — местная полночь).

    None — события в эти сутки нет (полярные день или ночь).
    """
    sunrise = sunset = None
    step = timedelta(minutes=5)
    moment = day
    was_down = sun_is_down(moment, lat, lon)
    for _ in range(288):
        following = moment + step
        is_down = sun_is_down(following, lat, lon)
        if is_down != was_down:
            crossing = _crossing(moment, following, lat, lon)
            if was_down and sunrise is None:
                sunrise = crossing
            elif not was_down and sunset is None:
                sunset = crossing
        moment, was_down = following, is_down
    return sunrise, sunset

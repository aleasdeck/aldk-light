"""Расписание включения и выключения ламп."""

from __future__ import annotations

from datetime import datetime, timedelta


def parse_hhmm(text: str) -> tuple[int, int] | None:
    """«08:30» → (8, 30); None, если строка не похожа на время."""
    try:
        hours, minutes = str(text).strip().split(":")
        hours, minutes = int(hours), int(minutes)
    except ValueError:
        return None
    if 0 <= hours <= 23 and 0 <= minutes <= 59:
        return hours, minutes
    return None


def desired_power(now: datetime, on_at: str, off_at: str, days: list[int]) -> bool | None:
    """Должны ли лампы сейчас гореть по расписанию.

    Смотрим, какое событие — включение или выключение — случилось последним.
    days — дни недели (0 = понедельник), в которые события срабатывают.
    None — расписание ничего не говорит (нет дней, времена совпадают или некорректны).
    """
    on_time, off_time = parse_hhmm(on_at), parse_hhmm(off_at)
    if on_time is None or off_time is None or on_time == off_time or not days:
        return None
    latest: tuple[datetime, bool] | None = None
    for back in range(8):  # неделя назад с запасом на сегодняшний день
        day = now - timedelta(days=back)
        if day.weekday() not in days:
            continue
        for (hours, minutes), turns_on in ((on_time, True), (off_time, False)):
            moment = day.replace(hour=hours, minute=minutes, second=0, microsecond=0)
            if moment <= now and (latest is None or moment > latest[0]):
                latest = (moment, turns_on)
    return None if latest is None else latest[1]

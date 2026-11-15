import unittest
from datetime import datetime

import numpy as np

from aldk_light import protocol
from aldk_light.schedule import desired_power, parse_hhmm
from aldk_light.screen import ZoneStats, colour_confidence, dominant_color, grid_for, zone_stats


class ProtocolTest(unittest.TestCase):
    # Пакеты питания, яркости и цвета проверены на лампах.
    def test_power(self):
        self.assertEqual(protocol.power(True).hex(), "bc01010155")
        self.assertEqual(protocol.power(False).hex(), "bc01010055")

    def test_brightness(self):
        self.assertEqual(protocol.brightness(1000).hex(), "bc050603e80000000055")
        self.assertEqual(protocol.brightness(100).hex(), "bc050600640000000055")

    def test_brightness_is_clamped(self):
        self.assertEqual(protocol.brightness(5000), protocol.brightness(1000))
        self.assertEqual(protocol.brightness(-3), protocol.brightness(0))

    def test_hue_sat(self):
        self.assertEqual(protocol.hue_sat(120, 1000).hex(), "bc0406007803e8000055")
        self.assertEqual(protocol.hue_sat(280, 1000).hex(), "bc0406011803e8000055")
        self.assertEqual(protocol.hue_sat(0, 0).hex(), "bc040600000000000055")

    def test_hue_wraps(self):
        self.assertEqual(protocol.hue_sat(360, 500), protocol.hue_sat(0, 500))

    def test_effect(self):
        self.assertEqual(protocol.effect(0x37).hex(), "bc0602003755")
        self.assertEqual(protocol.effect_reversed(False).hex(), "bc07010055")
        self.assertEqual(protocol.effect_reversed(True).hex(), "bc07010155")

    def test_effect_speed(self):
        self.assertEqual(protocol.effect_speed(50).hex(), "bc08013255")
        self.assertEqual(protocol.effect_speed(0), protocol.effect_speed(1))
        self.assertEqual(protocol.effect_speed(500), protocol.effect_speed(100))

    def test_effect_ids_unique(self):
        self.assertEqual(len(protocol.EFFECT_IDS), len(protocol.EFFECTS))


def solid(color, height=30, width=40):
    return np.full((height, width, 3), color, np.float32)


class ZoneTest(unittest.TestCase):
    def test_grid_follows_orientation(self):
        self.assertEqual(grid_for(2560, 1440), (4, 3))
        self.assertEqual(grid_for(1440, 2560), (3, 4))

    def test_zones_go_left_to_right_top_to_bottom(self):
        frame = solid((0.0, 0.0, 0.0))
        frame[:15, 20:, :] = (1.0, 0.0, 0.0)   # правый верхний угол красный
        stats = zone_stats(frame, 2, 2)
        self.assertTrue(np.allclose(stats.mean[1], (1, 0, 0)))
        for other in (0, 2, 3):
            self.assertTrue(np.allclose(stats.mean[other], 0))
        self.assertGreater(stats.weight[1], 0.9)
        self.assertEqual(float(stats.weight[0]), 0.0)

    def test_dominant_is_not_a_mix(self):
        # Слева красное, справа синее, красного больше: лампы должны быть красными, а не фиолетовыми.
        frame = solid((0.0, 0.0, 1.0))
        frame[:, :24, :] = (1.0, 0.0, 0.0)
        color, _ = dominant_color(zone_stats(frame, 4, 3))
        self.assertGreater(color[0], 0.8)
        self.assertLess(color[2], 0.2)

    def test_grey_screen_with_small_icon_stays_grey(self):
        frame = solid((0.9, 0.9, 0.9))
        frame[:2, :2, :] = (1.0, 0.0, 0.0)
        color, _ = dominant_color(zone_stats(frame, 4, 3))
        self.assertLess(float(color.max() - color.min()), 0.15)

    def test_current_hue_is_kept_when_rival_is_only_slightly_stronger(self):
        def stats(red_weight, blue_weight):
            return ZoneStats(
                mean=np.array([[1, 0, 0], [0, 0, 1]], np.float32),
                vivid=np.array([[1, 0, 0], [0, 0, 1]], np.float32),
                weight=np.array([red_weight, blue_weight], np.float32),
                light=np.array([1, 1], np.float32),
            )

        _, red_bin = dominant_color(stats(1.0, 0.5))
        _, kept = dominant_color(stats(1.0, 1.1), prefer_bin=red_bin)
        self.assertEqual(kept, red_bin)
        _, switched = dominant_color(stats(1.0, 2.0), prefer_bin=red_bin)
        self.assertNotEqual(switched, red_bin)

    def test_screen_green_is_not_sent_as_turquoise(self):
        import colorsys

        from aldk_light.screen import lamp_hue_sat

        # Зелёный с экрана пользователя: на мониторе это зелёный, а «в лоб» оттенок выходит 153° — бирюза.
        screen_green = (20 / 255, 130 / 255, 80 / 255)
        naive = colorsys.rgb_to_hsv(*screen_green)[0] * 360
        hue, sat = lamp_hue_sat(screen_green)
        self.assertGreater(naive, 150)
        self.assertLess(hue * 360, 142)
        self.assertGreater(hue * 360, 120)
        self.assertGreater(sat, 0.9)
        # Убавленный синий сдвигает ещё ближе к чистому зелёному.
        self.assertLess(lamp_hue_sat(screen_green, blue=60)[0], hue)

    def test_lamp_colour_keeps_pure_and_grey_colours(self):
        from aldk_light.screen import lamp_hue_sat

        self.assertAlmostEqual(lamp_hue_sat((1, 0, 0))[0], 0.0)
        self.assertAlmostEqual(lamp_hue_sat((0, 1, 0))[0], 1 / 3)
        self.assertAlmostEqual(lamp_hue_sat((0, 0, 1))[0], 2 / 3)
        self.assertEqual(lamp_hue_sat((0.5, 0.5, 0.5))[1], 0.0)
        self.assertEqual(lamp_hue_sat((0, 0, 0))[1], 0.0)
        self.assertEqual(lamp_hue_sat((0.9, 0.2, 0.2), saturation=250)[1], 1.0)

    def test_dark_scenes_are_dimmed_and_bright_ones_are_not(self):
        from aldk_light.screen import scene_brightness

        dark, bright = 0.12, 0.8
        self.assertLess(scene_brightness(dark, 0, 40), scene_brightness(dark, 0, 0) * 0.6)
        self.assertLess(scene_brightness(dark, 0, 100), scene_brightness(dark, 0, 40))
        self.assertEqual(scene_brightness(bright, 0, 100), scene_brightness(bright, 0, 0))
        self.assertEqual(scene_brightness(0.0, 0, 40), 0.0)
        self.assertAlmostEqual(scene_brightness(0.0, 10, 40), 0.01)   # минимум «на глаз»: 10 → 1% мощности

    def test_full_dimming_makes_the_lamp_as_dark_as_the_picture(self):
        from aldk_light.screen import scene_brightness

        # Тёмная тема пользователя: яркость экрана 0.119. Раньше на лампы уходило 12% мощности
        # и он не видел разницы с обычной яркостью; теперь — около 1%.
        self.assertLess(scene_brightness(0.119, 0, 100), 0.015)
        self.assertLess(scene_brightness(0.119, 4, 38), 0.07)
        # Сцена средней яркости при этом не проваливается в темноту.
        self.assertGreater(scene_brightness(0.5, 0, 60), 0.4)

    def test_brightness_never_goes_backwards(self):
        from aldk_light.screen import scene_brightness

        for dimming in (0, 40, 100):
            values = [scene_brightness(i / 200, 5, dimming) for i in range(201)]
            self.assertEqual(values, sorted(values))

    def test_dark_tinted_scene_keeps_its_tint(self):
        # Тёмно-синяя ночная сцена: раньше серая масса перевешивала и лампы белели.
        color, _ = dominant_color(zone_stats(solid((0.02, 0.03, 0.10)), 4, 3))
        self.assertGreater(color[2], color[0] * 3)

    def test_dark_grey_screen_with_a_few_coloured_pixels_stays_neutral(self):
        from aldk_light.screen import lamp_hue_sat

        # Как тёмная тема у пользователя: фон (27, 27, 28), около 1% точек — яркий жёлтый текст.
        # Раньше такие точки красили лампы в насыщенный жёлтый.
        frame = solid((27 / 255, 27 / 255, 28 / 255), height=60, width=100)
        rng = np.random.default_rng(1)
        ys, xs = rng.integers(0, 60, 60), rng.integers(0, 100, 60)
        frame[ys, xs, :] = (0.9, 0.8, 0.2)
        stats = zone_stats(frame, 4, 3)
        color, _ = dominant_color(stats)
        _, sat = lamp_hue_sat(color, saturation=200)
        self.assertGreater(sat, 0.9)                              # без проверки доверия — тот самый жёлтый
        self.assertLess(sat * colour_confidence(stats), 0.1)      # с ней лампы остаются нейтральными

    def test_colourful_screen_is_still_trusted(self):
        from aldk_light.screen import lamp_hue_sat

        # Четверть экрана занята насыщенным цветом — это уже цвет сцены, а не шум.
        frame = solid((0.12, 0.12, 0.12), height=60, width=100)
        frame[:, :25, :] = (0.1, 0.7, 0.2)
        stats = zone_stats(frame, 4, 3)
        color, _ = dominant_color(stats)
        hue, sat = lamp_hue_sat(color)
        self.assertEqual(colour_confidence(stats), 1.0)
        self.assertGreater(sat, 0.8)
        self.assertAlmostEqual(hue * 360, 120, delta=15)

    def test_dark_but_evenly_tinted_scene_is_trusted(self):
        # Ночное небо: света мало, но вся картинка синяя — это цвет сцены.
        self.assertEqual(colour_confidence(zone_stats(solid((0.02, 0.03, 0.10)), 4, 3)), 1.0)

    def test_plain_grey_and_black_screens_have_no_colour(self):
        self.assertEqual(colour_confidence(zone_stats(solid((0.5, 0.5, 0.5)), 4, 3)), 0.0)
        self.assertEqual(colour_confidence(zone_stats(solid((0.0, 0.0, 0.0)), 4, 3)), 0.0)

    def test_monitor_choice_survives_renumbering(self):
        from aldk_light.screen import choose_monitors, monitor_key

        main = {"left": 0, "top": 0, "width": 2560, "height": 1440}
        side = {"left": 2560, "top": -560, "width": 1440, "height": 2560}
        picked = [monitor_key(main)]
        self.assertEqual(picked, ["0,0,2560,1440"])
        # Тот самый случай: после сна Windows выдала мониторы в другом порядке.
        self.assertEqual(choose_monitors([main, side], picked), [main])
        self.assertEqual(choose_monitors([side, main], picked), [main])

    def test_monitor_choice_falls_back_to_all(self):
        from aldk_light.screen import choose_monitors

        main = {"left": 0, "top": 0, "width": 2560, "height": 1440}
        side = {"left": 2560, "top": -560, "width": 1440, "height": 2560}
        self.assertEqual(choose_monitors([main, side], []), [main, side])
        self.assertEqual(choose_monitors([main, side], ["9,9,800,600"]), [main, side])   # выбранный монитор отключили

    def test_display_keys_from_the_interface_are_validated(self):
        from aldk_light.engine import _display_keys

        self.assertEqual(_display_keys(["0,0,2560,1440", "2560,-560,1440,2560"]), ["0,0,2560,1440", "2560,-560,1440,2560"])
        self.assertEqual(_display_keys(["0,0,2560,1440", "0,0,2560,1440", "<script>", 5, "1,2,3"]), ["0,0,2560,1440"])

    def test_desktop_splits_into_three_parts(self):
        from aldk_light.screen import split_into_parts

        # Как на столе у пользователя: горизонтальный 2560 слева, вертикальный 1440 справа.
        landscape = 0 + (np.arange(4) + 0.5) / 4 * 2560
        portrait = 2560 + (np.arange(3) + 0.5) / 3 * 1440
        parts = split_into_parts(np.concatenate([landscape, portrait]), 0, 4000)
        self.assertEqual(parts.tolist(), [0, 0, 1, 1, 2, 2, 2])

    def test_single_point_never_leaves_range(self):
        from aldk_light.screen import split_into_parts

        self.assertEqual(split_into_parts(np.array([0.0, 4000.0, 9999.0]), 0, 4000).tolist(), [0, 2, 2])

    def test_black_frame(self):
        stats = zone_stats(solid((0.0, 0.0, 0.0)), 4, 3)
        color, _ = dominant_color(stats)
        self.assertTrue(np.allclose(color, 0.0))
        self.assertEqual(float(stats.light.max()), 0.0)


class FadeTest(unittest.TestCase):
    def test_same_result_at_any_update_rate(self):
        from aldk_light.fade import ease

        one_step = ease(0.0, 1.0, 0.3, 0.5)
        many_steps = 0.0
        for _ in range(30):
            many_steps = ease(many_steps, 1.0, 0.01, 0.5)
        self.assertAlmostEqual(one_step, many_steps, places=6)

    def test_reaches_target_and_never_overshoots(self):
        from aldk_light.fade import ease

        value = 0.0
        for _ in range(300):
            value = ease(value, 1.0, 1 / 30, 0.3)
            self.assertLessEqual(value, 1.0)
        self.assertGreater(value, 0.999)

    def test_slider_range(self):
        from aldk_light.fade import smoothing_for, time_constant

        self.assertAlmostEqual(time_constant(0), 0.03)
        self.assertAlmostEqual(time_constant(95), 3.0)
        self.assertEqual(time_constant(500), time_constant(95))
        self.assertLess(time_constant(30), time_constant(60))
        for position in (0, 20, 61, 95):
            self.assertEqual(smoothing_for(time_constant(position)), position)

    def test_old_settings_keep_their_transition_time(self):
        import json
        import os
        import tempfile

        from aldk_light.fade import time_constant
        from aldk_light.settings import Settings

        with tempfile.TemporaryDirectory() as folder:
            os.makedirs(os.path.join(folder, "AldkLight"))
            path = os.path.join(folder, "AldkLight", "settings.json")
            with open(path, "w", encoding="utf-8") as file:
                json.dump({"smoothing": 95, "saturation": 200}, file)   # старый формат: 95 = 1.23 с
            old = os.environ.get("APPDATA")
            os.environ["APPDATA"] = folder
            try:
                loaded = Settings.load()
                loaded.save()
                again = Settings.load()
            finally:
                if old is None:
                    del os.environ["APPDATA"]
                else:
                    os.environ["APPDATA"] = old
        self.assertAlmostEqual(time_constant(loaded.smoothing), 1.23, delta=0.05)
        self.assertEqual(loaded.saturation, 200)
        self.assertEqual(again.smoothing, loaded.smoothing)   # повторная загрузка не пересчитывает ещё раз

    def test_two_stage_fade_starts_gently(self):
        import threading
        import time

        from aldk_light.fade import ColorFader

        seen, done = [], threading.Event()

        def collect(color, light):
            seen.append(light)
            if len(seen) >= 40:
                done.set()

        fader = ColorFader(lambda: 40, collect, rate=100)
        fader.start()
        try:
            fader.set_target(np.zeros(3, np.float32), 0.0)
            time.sleep(0.05)
            seen.clear()
            fader.set_target(np.zeros(3, np.float32), 1.0)
            self.assertTrue(done.wait(3.0))
        finally:
            fader.stop()
        steps = [b - a for a, b in zip(seen, seen[1:])]
        # Плавный разгон: первый шаг намного меньше самого большого, а не самый большой.
        self.assertLess(steps[0], max(steps) * 0.5)

    def test_fader_moves_gradually_towards_target(self):
        import threading
        import time

        from aldk_light.fade import ColorFader

        seen, done = [], threading.Event()

        def collect(color, light):
            seen.append((float(color[0]), light))
            if len(seen) >= 12:
                done.set()

        fader = ColorFader(lambda: 60, collect, rate=100)
        fader.start()
        try:
            fader.set_target(np.array([0.0, 0.0, 0.0], np.float32), 0.0)
            time.sleep(0.05)
            fader.set_target(np.array([1.0, 0.0, 0.0], np.float32), 1.0)
            self.assertTrue(done.wait(3.0))
        finally:
            fader.stop()
        reds = [red for red, _ in seen]
        self.assertEqual(reds, sorted(reds))           # только вверх, без скачков назад
        self.assertLess(reds[-1], 1.0)                 # за 0.1 с при плавности 0.5 с до цели не дошёл
        self.assertGreater(reds[-1], reds[0])


class SunTest(unittest.TestCase):
    MOSCOW = (55.7558, 37.6173)
    MSK = None

    @classmethod
    def setUpClass(cls):
        from datetime import timedelta, timezone

        cls.MSK = timezone(timedelta(hours=3))

    def assertClock(self, moment, expected, minutes=3):
        from datetime import timedelta

        hours, mins = map(int, expected.split(":"))
        target = moment.replace(hour=hours, minute=mins, second=0, microsecond=0)
        self.assertLess(abs(moment - target), timedelta(minutes=minutes), f"{moment:%H:%M} вместо {expected}")

    def test_known_sunrise_and_sunset(self):
        from aldk_light.sun import sun_events

        # Справочные значения для Москвы: летнее и зимнее солнцестояние.
        sunrise, sunset = sun_events(datetime(2026, 6, 21, tzinfo=self.MSK), *self.MOSCOW)
        self.assertClock(sunrise, "03:44")
        self.assertClock(sunset, "21:18")
        sunrise, sunset = sun_events(datetime(2026, 12, 21, tzinfo=self.MSK), *self.MOSCOW)
        self.assertClock(sunrise, "08:57")
        self.assertClock(sunset, "15:58")

    def test_polar_night_and_polar_day(self):
        from aldk_light.sun import lamps_wanted, sun_events

        murmansk = (68.97, 33.08)
        winter, summer = datetime(2026, 12, 21, tzinfo=self.MSK), datetime(2026, 6, 21, tzinfo=self.MSK)
        self.assertEqual(sun_events(winter, *murmansk), (None, None))
        self.assertEqual(sun_events(summer, *murmansk), (None, None))
        for hour in (0, 6, 12, 18):
            self.assertTrue(lamps_wanted(winter.replace(hour=hour), *murmansk))    # ночь круглые сутки
            self.assertFalse(lamps_wanted(summer.replace(hour=hour), *murmansk))   # солнце не заходит

    def test_lamps_follow_sunset_and_sunrise_with_margin(self):
        from aldk_light.sun import lamps_wanted

        day = datetime(2026, 12, 21, tzinfo=self.MSK)   # рассвет 08:57, закат 15:58
        at = lambda h, m: day.replace(hour=h, minute=m)
        self.assertTrue(lamps_wanted(at(3, 0), *self.MOSCOW))
        self.assertFalse(lamps_wanted(at(12, 0), *self.MOSCOW))
        self.assertTrue(lamps_wanted(at(20, 0), *self.MOSCOW))
        # Без запаса лампы включаются ровно на закате, с запасом 30 минут — на полчаса раньше.
        self.assertFalse(lamps_wanted(at(15, 40), *self.MOSCOW, 0))
        self.assertTrue(lamps_wanted(at(15, 40), *self.MOSCOW, 30))
        self.assertTrue(lamps_wanted(at(16, 5), *self.MOSCOW, 0))
        # Утром с запасом лампы горят ещё полчаса после рассвета.
        self.assertFalse(lamps_wanted(at(9, 15), *self.MOSCOW, 0))
        self.assertTrue(lamps_wanted(at(9, 15), *self.MOSCOW, 30))
        self.assertFalse(lamps_wanted(at(9, 40), *self.MOSCOW, 30))

    def test_result_does_not_depend_on_the_clock_zone(self):
        from datetime import timezone

        from aldk_light.sun import elevation

        local = datetime(2026, 12, 21, 12, 0, tzinfo=self.MSK)
        self.assertAlmostEqual(elevation(local, *self.MOSCOW), elevation(local.astimezone(timezone.utc), *self.MOSCOW))

    def test_default_city_exists(self):
        from aldk_light.sun import CITIES, default_location

        name, lat, lon = default_location()
        self.assertIn(name, [city[0] for city in CITIES])
        self.assertEqual(len({city[0] for city in CITIES}), len(CITIES))

    def test_coordinates_survive_a_settings_round_trip(self):
        import json
        import os
        import tempfile

        from aldk_light.settings import Settings

        with tempfile.TemporaryDirectory() as folder:
            os.makedirs(os.path.join(folder, "AldkLight"))
            with open(os.path.join(folder, "AldkLight", "settings.json"), "w", encoding="utf-8") as file:
                # Широта записана целым числом — так JSON сохраняет «55.0».
                json.dump({"version": 2, "schedule_sun": True, "schedule_lat": 55, "schedule_lon": 37.62, "schedule_city": ""}, file)
            old = os.environ.get("APPDATA")
            os.environ["APPDATA"] = folder
            try:
                loaded = Settings.load()
            finally:
                if old is None:
                    del os.environ["APPDATA"]
                else:
                    os.environ["APPDATA"] = old
        self.assertTrue(loaded.schedule_sun)
        self.assertEqual((loaded.schedule_lat, loaded.schedule_lon, loaded.schedule_city), (55.0, 37.62, ""))


class ScheduleTest(unittest.TestCase):
    ALL = [0, 1, 2, 3, 4, 5, 6]

    def test_parse(self):
        self.assertEqual(parse_hhmm("08:05"), (8, 5))
        for bad in ("", "25:00", "8", "ab:cd", "10:60"):
            self.assertIsNone(parse_hhmm(bad))

    def test_daytime_window(self):
        day = datetime(2026, 11, 16)  # понедельник
        self.assertTrue(desired_power(day.replace(hour=12), "09:00", "23:30", self.ALL))
        self.assertFalse(desired_power(day.replace(hour=23, minute=45), "09:00", "23:30", self.ALL))
        self.assertFalse(desired_power(day.replace(hour=3), "09:00", "23:30", self.ALL))

    def test_window_over_midnight(self):
        day = datetime(2026, 11, 16)
        self.assertTrue(desired_power(day.replace(hour=1), "18:00", "02:00", self.ALL))
        self.assertFalse(desired_power(day.replace(hour=10), "18:00", "02:00", self.ALL))

    def test_only_selected_days_fire(self):
        # Расписание только по понедельникам: во вторник днём действует понедельничное выключение.
        tuesday_noon = datetime(2026, 11, 17, 12, 0)
        self.assertFalse(desired_power(tuesday_noon, "09:00", "23:30", [0]))
        monday_noon = datetime(2026, 11, 16, 12, 0)
        self.assertTrue(desired_power(monday_noon, "09:00", "23:30", [0]))

    def test_nothing_to_say(self):
        now = datetime(2026, 11, 16, 12, 0)
        self.assertIsNone(desired_power(now, "09:00", "09:00", self.ALL))
        self.assertIsNone(desired_power(now, "09:00", "23:30", []))
        self.assertIsNone(desired_power(now, "oops", "23:30", self.ALL))


class LevelMeterTest(unittest.TestCase):
    def setUp(self):
        from aldk_light.audio import LevelMeter

        self.rate = 48000
        self.meter = LevelMeter(self.rate)
        t = np.arange(1024, dtype=np.float32) / self.rate
        self.bass = (0.8 * np.sin(2 * np.pi * 90 * t)).astype(np.float32)
        self.treble = (0.8 * np.sin(2 * np.pi * 6000 * t)).astype(np.float32)
        self.silence = np.zeros(1024, np.float32)

    def test_bass_hit_after_silence_is_loud_and_a_beat(self):
        for _ in range(20):
            self.meter.feed(self.silence, 1.0)
        level, beat = self.meter.feed(self.bass, 1.0)
        self.assertGreater(level, 0.9)
        self.assertTrue(beat)

    def test_treble_is_ignored(self):
        self.meter.feed(self.bass, 1.0)  # задаёт опорную громкость
        for _ in range(30):
            level, _ = self.meter.feed(self.treble, 1.0)
        self.assertLess(level, 0.05)

    def test_level_falls_in_silence(self):
        self.meter.feed(self.bass, 1.0)
        for _ in range(30):
            level, beat = self.meter.feed(self.silence, 1.0)
        self.assertLess(level, 0.01)
        self.assertFalse(beat)

    def test_short_chunk_is_accepted(self):
        level, _ = self.meter.feed(self.bass[:300], 1.0)
        self.assertGreaterEqual(level, 0.0)


if __name__ == "__main__":
    unittest.main()

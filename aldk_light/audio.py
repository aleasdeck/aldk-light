"""Уровень звука компьютера (то, что играет в колонках) для режима цветомузыки."""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import numpy as np
import pyaudiowpatch as pyaudio

log = logging.getLogger(__name__)

_CHUNK = 1024
_BASS_HZ = (30.0, 250.0)


class LevelMeter:
    """Превращает куски звука в уровень 0..1 и признак удара (бита)."""

    def __init__(self, rate: int) -> None:
        self._window = np.hanning(_CHUNK).astype(np.float32)
        freqs = np.fft.rfftfreq(_CHUNK, 1.0 / rate)
        self._bass = (freqs >= _BASS_HZ[0]) & (freqs <= _BASS_HZ[1])
        self._peak = 1e-4      # автоподстройка под громкость
        self._average = 0.0    # медленное среднее — относительно него ищем удары
        self._level = 0.0

    def feed(self, mono: np.ndarray, sensitivity: float) -> tuple[float, bool]:
        if len(mono) < _CHUNK:
            mono = np.pad(mono, (0, _CHUNK - len(mono)))
        spectrum = np.abs(np.fft.rfft(mono[:_CHUNK] * self._window))
        energy = float(spectrum[self._bass].mean())

        self._peak = max(self._peak * 0.9985, energy, 1e-4)
        raw = min(1.0, (energy / self._peak) ** 1.5 * sensitivity)
        beat = raw > 0.35 and raw > self._average * 1.6
        self._average += (raw - self._average) * 0.08
        # Мгновенный подъём, плавный спад.
        self._level = raw if raw > self._level else self._level * 0.82
        return self._level, beat

    def decay(self) -> float:
        self._level *= 0.7
        return self._level


class AudioSampler:
    """Слушает звук вывода по умолчанию и вызывает on_level(уровень 0..1, удар)."""

    def __init__(self, get_sensitivity: Callable[[], int], on_level: Callable[[float, bool], None]) -> None:
        self._get_sensitivity = get_sensitivity
        self._on_level = on_level
        self._pa: pyaudio.PyAudio | None = None
        self._stream = None
        self._meter: LevelMeter | None = None
        self._channels = 2
        self._last_chunk = 0.0
        self._stop_event = threading.Event()
        self._watchdog: threading.Thread | None = None

    def start(self) -> None:
        self._pa = pyaudio.PyAudio()
        device = self._pa.get_default_wasapi_loopback()
        rate = int(device["defaultSampleRate"])
        self._channels = int(device["maxInputChannels"])
        self._meter = LevelMeter(rate)
        self._last_chunk = time.monotonic()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self._channels,
            rate=rate,
            frames_per_buffer=_CHUNK,
            input=True,
            input_device_index=device["index"],
            stream_callback=self._on_audio,
        )
        self._watchdog = threading.Thread(target=self._watch_silence, name="audio-silence", daemon=True)
        self._watchdog.start()
        log.info("audio loopback: %s, %d Hz, %d ch", device["name"], rate, self._channels)

    def stop(self) -> None:
        self._stop_event.set()
        try:
            if self._stream is not None:
                self._stream.stop_stream()
                self._stream.close()
        except Exception:
            log.exception("audio stream close failed")
        finally:
            self._stream = None
            if self._pa is not None:
                self._pa.terminate()
                self._pa = None
        # После возврата из stop() уровень больше никто не присылает.
        if self._watchdog is not None:
            self._watchdog.join(timeout=1.0)
            self._watchdog = None

    def _on_audio(self, data, _frames, _time_info, _status):
        try:
            samples = np.frombuffer(data, np.int16).astype(np.float32) * (1.0 / 32768.0)
            mono = samples.reshape(-1, self._channels).mean(axis=1)
            self._last_chunk = time.monotonic()
            level, beat = self._meter.feed(mono, self._get_sensitivity() / 100.0)
            self._on_level(level, beat)
        except Exception:
            log.exception("audio callback failed")
        return None, pyaudio.paContinue

    def _watch_silence(self) -> None:
        # Когда ничего не играет, Windows не присылает звук вовсе — гасим уровень сами.
        while not self._stop_event.wait(0.06):
            if self._stop_event.is_set():
                return
            if self._meter is not None and time.monotonic() - self._last_chunk > 0.25:
                self._on_level(self._meter.decay(), False)

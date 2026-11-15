"""Связь с контроллером ламп по Bluetooth LE.

Работает в собственном потоке со своим циклом asyncio, чтобы не мешать интерфейсу.
Команды не копятся в очереди: по каждому каналу (питание, яркость, цвет, эффект)
хранится только последнее значение, поэтому лампы не отстают от источника,
даже если он шлёт обновления быстрее, чем их принимает контроллер.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Callable

from bleak import BleakClient, BleakScanner

from . import protocol

log = logging.getLogger(__name__)

# Порядок отправки каналов в одном цикле.
_ORDER = ("power", "bri", "hs", "effect", "dir", "speed")

# Пауза между пакетами: 60 пакетов/с — столько контроллер принимал на замере.
# Подсветка шлёт цвет и яркость 30 раз в секунду, это как раз 60 пакетов.
_PACKET_INTERVAL = 1 / 60
# Пакеты идут без подтверждения. Замер на лампах: подтверждённая запись занимает ~41 мс
# (три интервала соединения по 13.75 мс), и если подтверждать каждый пакет, до ламп доходит
# около 20 обновлений в секунду вместо 50, причём неровно — переход цвета идёт ступеньками.
# Редкое подтверждение оставлено как страховка: оно не даёт стеку Bluetooth копить хвост.
_ACK_EVERY = 150

StatusCallback = Callable[[str, str], None]


class LampLink:
    """Потокобезопасное управление лампами с автоматическим переподключением."""

    def __init__(
        self, device_name: str, on_status: StatusCallback | None = None, known_address: str = ""
    ) -> None:
        self._device_name = device_name
        self._known_address = known_address.upper()
        self._on_status = on_status or (lambda state, detail: None)
        self._lock = threading.Lock()
        self._state: dict[str, bytes] = {}    # всё, что должно быть на лампе
        self._pending: dict[str, bytes] = {}  # что ещё не отправлено
        self._flush_wanted = False
        self._flush_done = threading.Event()
        self._last_packet: bytes | None = None
        self._connected = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._wake: asyncio.Event | None = None
        self._task: asyncio.Task | None = None
        self._thread: threading.Thread | None = None
        self._stopping = False
        self.sent_packets = 0                 # сколько пакетов реально ушло — для замеров

    # ---- управление из любого потока -------------------------------------

    def set_power(self, on: bool) -> None:
        self._put("power", protocol.power(on))

    def set_brightness(self, level: int) -> None:
        self._put("bri", protocol.brightness(level))

    def set_color(self, hue: int, sat: int) -> None:
        self._put("hs", protocol.hue_sat(hue, sat), drop=("effect", "dir", "speed"))

    def set_effect(self, effect_id: int, is_reversed: bool, speed: int) -> None:
        self._put("effect", protocol.effect(effect_id), drop=("hs",))
        self._put("dir", protocol.effect_reversed(is_reversed))
        self._put("speed", protocol.effect_speed(speed))

    def flush(self, timeout: float) -> bool:
        """Дождаться, пока всё накопленное дойдёт до контроллера. False — не успели или нет связи."""
        if not self._connected:
            return False
        self._flush_done.clear()
        self._flush_wanted = True
        self._kick()
        return self._flush_done.wait(timeout)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="lamp-ble", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopping = True
        loop, task = self._loop, self._task
        if loop is not None and task is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    # ---- внутреннее -------------------------------------------------------

    def _put(self, key: str, packet: bytes, drop: tuple[str, ...] = ()) -> None:
        with self._lock:
            for other in drop:
                self._state.pop(other, None)
                self._pending.pop(other, None)
            if self._state.get(key) == packet:
                return
            self._state[key] = packet
            self._pending[key] = packet
        self._kick()

    def _kick(self) -> None:
        loop, wake = self._loop, self._wake
        if loop is not None and wake is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(wake.set)
            except RuntimeError:
                pass  # цикл закрылся между проверкой и вызовом

    def _status(self, state: str, detail: str = "") -> None:
        try:
            self._on_status(state, detail)
        except Exception:  # получатель статуса не должен ронять связь
            log.exception("status callback failed")

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._wake = asyncio.Event()
        self._task = loop.create_task(self._main())
        try:
            loop.run_until_complete(self._task)
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("BLE thread crashed")
        finally:
            self._connected = False
            loop.close()

    def _matches(self, device, adv) -> bool:
        # Имя приходит не в каждом рекламном пакете, а адрес — в каждом.
        if self._known_address and device.address.upper() == self._known_address:
            return True
        name = adv.local_name or device.name or ""
        return name == self._device_name

    async def _main(self) -> None:
        assert self._wake is not None
        while not self._stopping:
            try:
                self._status("searching")
                device = await BleakScanner.find_device_by_filter(self._matches, timeout=8.0)
                if device is None:
                    continue
                self._status("connecting")
                lost = asyncio.Event()

                def on_disconnect(_client, lost=lost) -> None:
                    lost.set()
                    self._wake.set()

                async with BleakClient(device, disconnected_callback=on_disconnect, timeout=15.0) as client:
                    self._known_address = device.address.upper()
                    with self._lock:
                        self._pending = dict(self._state)  # привести лампы к текущему состоянию
                    self._connected = True
                    self._status("connected", device.address)
                    self._wake.set()
                    await self._pump(client, lost)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("BLE error: %s: %s", type(exc).__name__, exc)
                self._status("error", str(exc) or type(exc).__name__)
                await asyncio.sleep(2.0)
            finally:
                self._connected = False

    async def _pump(self, client: BleakClient, lost: asyncio.Event) -> None:
        assert self._wake is not None
        while True:
            await self._wake.wait()
            self._wake.clear()
            if lost.is_set():
                return
            with self._lock:
                batch, self._pending = self._pending, {}
            for key in _ORDER:
                packet = batch.get(key)
                if packet is None:
                    continue
                self.sent_packets += 1
                acked = self.sent_packets % _ACK_EVERY == 0
                await client.write_gatt_char(protocol.WRITE_UUID, packet, response=acked)
                self._last_packet = packet
                await asyncio.sleep(_PACKET_INTERVAL)
            if self._flush_wanted:
                with self._lock:
                    idle = not self._pending
                if idle:
                    # Повторяем последний пакет с подтверждением: когда контроллер ответил на него,
                    # всё отправленное раньше тоже гарантированно дошло.
                    if self._last_packet is not None:
                        await client.write_gatt_char(protocol.WRITE_UUID, self._last_packet, response=True)
                    self._flush_wanted = False
                    self._flush_done.set()

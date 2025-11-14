# slot_manager.py

# -*- coding: utf-8 -*-

"""
Упрощённый SlotManager: только free/oldest по LRU, без hot/cold.

- get_slot(): сначала свободный (ещё не использовался), иначе самый старый по времени.
- Для big: если есть restore_key — делаем restore на выбранный слот.
- Сохранение всегда после завершения запроса.
"""

import time
import asyncio
import logging
from typing import List, Tuple, Dict, Optional

from config import BACKENDS

log = logging.getLogger(__name__)

GSlot = Tuple[int, int]  # (backend_id, local_slot_id)


class SlotManager:
    def __init__(self):
        self.backends = []
        total_slots = 0

        for be_id, conf in enumerate(BACKENDS):
            n_slots = int(conf["n_slots"])
            self.backends.append({"id": be_id, "client": None, "n_slots": n_slots})
            total_slots += n_slots

        self._all_slots: List[GSlot] = [
            (be_id, s)
            for be_id, be in enumerate(self.backends)
            for s in range(be["n_slots"])
        ]

        self._last_used: Dict[GSlot, float] = {g: 0.0 for g in self._all_slots}
        self._locks: Dict[GSlot, asyncio.Lock] = {
            g: asyncio.Lock() for g in self._all_slots
        }

        log.info(
            "slot_manager n_backends=%d total_slots=%d",
            len(self.backends),
            total_slots,
        )

    def set_clients(self, clients: List):
        for i, client in enumerate(clients):
            self.backends[i]["client"] = client

    def _is_free(self, g: GSlot) -> bool:
        return self._last_used.get(g, 0.0) == 0.0

    def _get_free_or_oldest(self) -> Tuple[GSlot, asyncio.Lock]:
        free = [g for g in self._all_slots if self._is_free(g)]
        if free:
            g = free[0]
            return g, self._locks[g]

        g = sorted(self._all_slots, key=lambda x: self._last_used.get(x, 0.0))[0]
        return g, self._locks[g]

    async def acquire_for_request(
        self,
        restore_key: Optional[str] = None,
    ) -> Tuple[GSlot, asyncio.Lock, Optional[bool]]:
        g, lock = self._get_free_or_oldest()
        await lock.acquire()

        restored: Optional[bool] = None
        if restore_key:
            client = self.backends[g[0]]["client"]
            restored = await client.restore_slot(g[1], restore_key)
            log.info(
                "restore_before_chat g=%s key=%s ok=%s",
                g,
                (restore_key[:16] if restore_key else None),
                restored,
            )

        return g, lock, restored

    async def save_after(self, g: GSlot, key: str) -> bool:
        client = self.backends[g[0]]["client"]
        ok = await client.save_slot(g[1], key)
        self._last_used[g] = time.time()
        return ok

    def release(self, g: GSlot):
        if self._locks[g].locked():
            self._locks[g].release()

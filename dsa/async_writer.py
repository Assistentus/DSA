"""Background writer for DSA state files.

Guarantees (see CHANGELOG.md):

* F1: a batch is never dropped.  If the queue is full, ``put_update`` waits
  ``put_timeout`` seconds and then writes the batch synchronously.
* F2: rows in flight are tracked per parameter, so ``step`` never reads a
  row the writer has not finished writing.
* F5: indices are sorted before writing (sequential IO).
* F6: ``shutdown`` is a state machine (``RUNNING -> DRAINING -> STOPPED``,
  or ``FAILED``) and is retryable: a transient failure (stuck disk, slow
  msync) does not turn a second call into a false success.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from enum import Enum
from typing import Dict, Optional, Set

import numpy as np

logger = logging.getLogger("DSA_AsyncWriter")


class WriterState(str, Enum):
    RUNNING = "RUNNING"
    DRAINING = "DRAINING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


class AsyncDiskWriter:
    """Double-buffered asynchronous writer over ``np.memmap`` files."""

    def __init__(self, memmaps: Dict[str, np.memmap], max_queue_size: int = 150,
                 put_timeout: float = 5.0):
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be > 0")
        if put_timeout < 0:
            raise ValueError("put_timeout must be >= 0")
        self.memmaps = dict(memmaps)
        self.queue: "queue.Queue" = queue.Queue(maxsize=max_queue_size)
        self._stop_signal = object()
        self._lock = threading.Lock()          # protects memmaps and _inflight
        self._condition = threading.Condition()  # protects the state machine
        self._put_timeout = float(put_timeout)
        self._inflight: Dict[str, Set[int]] = {}
        self._write_errors = []
        self._flush_errors = []
        self._sync_fallbacks = 0
        self._state = WriterState.RUNNING
        self._active_puts = 0
        self._stop_sent = False
        self._fatal_error: Optional[BaseException] = None

        self.worker_thread = threading.Thread(
            target=self._process_queue, name="DSA_AsyncWriter", daemon=True)
        self.worker_thread.start()
        logger.info("💿 AsyncDiskWriter: фоновый поток инициализирован.")

    # ───────────────────────────── state ─────────────────────────────

    @property
    def state(self) -> WriterState:
        with self._condition:
            return self._state

    @property
    def _stopped(self) -> bool:
        """Backward-compatible view: True only after a clean shutdown."""
        return self.state is WriterState.STOPPED

    # ─────────────────────── F2: rows in flight ───────────────────────

    def intersects_inflight(self, name: str, indices) -> bool:
        """Whether any of ``indices`` is still being written."""
        with self._lock:
            inflight = self._inflight.get(name)
            if not inflight:
                return False
            if isinstance(indices, np.ndarray):
                return bool(inflight.intersection(indices.tolist()))
            return any(int(i) in inflight for i in indices)

    def wait_all_flushed(self, timeout: Optional[float] = None):
        """Wait until the queue is empty and no rows are in flight.

        Raises if the worker died, if a write failed during the session, or
        if ``timeout`` expires.  A failed write is never hidden from the
        training loop.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self._write_errors:
                raise RuntimeError(
                    "AsyncDiskWriter: за сессию были ошибки записи; состояние "
                    f"на диске неполно. Первая ошибка: {self._write_errors[0]}")
            if not self.worker_thread.is_alive() and self.queue.unfinished_tasks:
                raise RuntimeError(
                    "AsyncDiskWriter: фоновый поток завершился, но очередь "
                    f"содержит {self.queue.unfinished_tasks} незавершённых "
                    "батчей — данные не записаны")
            if self.queue.unfinished_tasks == 0:
                break
            if deadline is not None and time.monotonic() >= deadline:
                raise RuntimeError(
                    f"AsyncDiskWriter: очередь не опустела за {timeout:.1f} с "
                    f"(осталось батчей: {self.queue.unfinished_tasks}) — "
                    f"данные НЕ дописаны")
            time.sleep(0.005)

        with self._lock:
            leftover = {n: sorted(s) for n, s in self._inflight.items() if s}
        if leftover:
            raise RuntimeError(
                f"AsyncDiskWriter: очередь пуста, но строки остались в полёте: "
                f"{leftover}")

    def _wait_queue_drained(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            if not self.worker_thread.is_alive() and self.queue.unfinished_tasks:
                return False
            if self.queue.unfinished_tasks == 0:
                return True
            if time.monotonic() >= deadline:
                return self.queue.unfinished_tasks == 0
            time.sleep(0.005)

    # ─────────────────────────── writing ───────────────────────────

    def _apply_write(self, name: str, indices, w_np, m_np, v_np, t_np=None):
        """Synchronous write of one batch (F5: sorted indices)."""
        indices = np.asarray(indices)
        order = np.argsort(indices)
        indices = indices[order]
        w_np, m_np, v_np = w_np[order], m_np[order], v_np[order]
        t_sorted = None if t_np is None else np.asarray(t_np)[order]
        with self._lock:
            self.memmaps[f"{name}_w"][indices] = w_np
            self.memmaps[f"{name}_m"][indices] = m_np
            self.memmaps[f"{name}_v"][indices] = v_np
            if t_sorted is not None:
                t_mm = self.memmaps.get(f"{name}_t")
                if t_mm is None:
                    raise RuntimeError(
                        f"передан счётчик шага для '{name}', но memmap "
                        f"'{name}_t' не открыт")
                t_mm[indices] = t_sorted

    def _process_queue(self):
        while True:
            batch_data = self.queue.get()
            if batch_data is self._stop_signal:
                try:
                    self._final_flush()
                except Exception as exc:  # pragma: no cover - defensive
                    self._flush_errors.append(f"финальный flush: {exc!r}")
                finally:
                    self.queue.task_done()
                break

            name, indices, w_np, m_np, v_np, t_np = batch_data
            try:
                self._apply_write(name, indices, w_np, m_np, v_np, t_np)
            except Exception as exc:  # F6: recorded, re-raised on shutdown
                msg = f"батч '{name}' ({len(indices)} строк) НЕ записан: {exc!r}"
                self._write_errors.append(msg)
                logger.error("❌ IO Error: " + msg)
            finally:
                with self._lock:
                    inflight = self._inflight.get(name)
                    if inflight is not None:
                        inflight.difference_update(int(i) for i in indices)
                self.queue.task_done()

    def _final_flush(self):
        logger.info("💾 AsyncDiskWriter: финализация данных на диске...")
        with self._lock:
            for key, mm in self.memmaps.items():
                try:
                    mm.flush()
                except Exception as exc:
                    msg = f"flush {key} не выполнен: {exc!r}"
                    self._flush_errors.append(msg)
                    logger.error("❌ " + msg)

    def put_update(self, name: str, indices_pt, w_pt, m_pt, v_pt, t_pt=None):
        """Queue one batch; never drops it (F1)."""
        with self._condition:
            if self._state is not WriterState.RUNNING:
                raise RuntimeError(
                    f"AsyncDiskWriter в состоянии {self._state.value}: запись "
                    f"невозможна (shutdown уже начат или завершён)")
            self._active_puts += 1

        try:
            idx_np = np.asarray(indices_pt.detach().cpu().numpy())
            w_np = w_pt.detach().cpu().numpy()
            m_np = m_pt.detach().cpu().numpy()
            v_np = v_pt.detach().cpu().numpy()
            t_np = None if t_pt is None else np.asarray(
                t_pt.detach().cpu().numpy(), dtype=np.int32)

            idx_set = {int(i) for i in idx_np.tolist()}
            with self._lock:
                self._inflight.setdefault(name, set()).update(idx_set)
            try:
                self.queue.put((name, idx_np, w_np, m_np, v_np, t_np),
                               timeout=self._put_timeout)
            except queue.Full:
                # F1: disk is behind — write synchronously, never drop.
                self._sync_fallbacks += 1
                logger.warning(
                    "⚠️ Очередь записи полна дольше %.1f с ('%s', %d строк): "
                    "синхронная запись напрямую.",
                    self._put_timeout, name, len(idx_np))
                try:
                    self._apply_write(name, idx_np, w_np, m_np, v_np, t_np)
                finally:
                    with self._lock:
                        self._inflight.setdefault(name, set()).difference_update(idx_set)
            except BaseException:
                with self._lock:
                    self._inflight.setdefault(name, set()).difference_update(idx_set)
                raise
        finally:
            with self._condition:
                self._active_puts -= 1
                self._condition.notify_all()

    # ─────────────────────────── shutdown ───────────────────────────

    def shutdown(self, timeout: float = 300.0, *, raise_on_error: bool = True) -> bool:
        """Drain, flush and stop; retryable after a transient failure.

        Returns ``True`` on a clean stop and ``False`` when
        ``raise_on_error=False`` and the writer could not finish.  A writer
        that previously failed and whose thread is still alive is retried,
        not reported as stopped (the 0.2.0 bug).
        """
        if timeout is None:
            raise TypeError("timeout must be a number of seconds")
        if timeout < 0:
            raise ValueError("timeout must be >= 0")

        with self._condition:
            if self._state is WriterState.STOPPED:
                return True
            if (self._state is WriterState.FAILED
                    and self._fatal_error is not None
                    and not self.worker_thread.is_alive()):
                if raise_on_error:
                    raise self._fatal_error
                return False
            self._state = WriterState.DRAINING
            self._condition.notify_all()

        err: Optional[BaseException] = None
        deadline = None if timeout is None else time.monotonic() + timeout

        # 1. Wait for put_update calls that started before DRAINING.
        while True:
            with self._condition:
                if self._active_puts == 0:
                    break
            if deadline is not None and time.monotonic() >= deadline:
                err = RuntimeError(
                    f"AsyncDiskWriter: {self._active_puts} вызов(ов) put_update "
                    f"не завершились за {timeout:.1f} с")
                break
            time.sleep(0.005)

        # 2. Drain queued batches (bounded: a stuck disk is an error, not a hang).
        if err is None and not self._wait_queue_drained(timeout):
            if self._write_errors:
                err = RuntimeError(
                    "AsyncDiskWriter: невыполненные записи за сессию: "
                    f"{len(self._write_errors)}; пример: {self._write_errors[0]}")
            else:
                err = RuntimeError(
                    f"AsyncDiskWriter: за {timeout:.1f} с очередь не опустела "
                    f"(осталось батчей: {self.queue.unfinished_tasks}) — "
                    f"данные НЕ дописаны")

        # 3. Stop the worker (FIFO: stop is enqueued after all data).
        if err is None and self.worker_thread.is_alive() and not self._stop_sent:
            self._stop_sent = True
            try:
                self.queue.put_nowait(self._stop_signal)
            except queue.Full:
                err = RuntimeError(
                    "AsyncDiskWriter: очередь заполнилась между дреном и "
                    "сигналом остановки")

        # 4. Join (msync may be slow).
        if err is None and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=timeout)
        if err is None and self.worker_thread.is_alive():
            err = RuntimeError(
                f"AsyncDiskWriter: фоновый поток не завершился за "
                f"{timeout if timeout is not None else '∞'} с — данные НЕ "
                f"гарантированно записаны на диск")

        if err is None:
            problems = []
            if self._flush_errors:
                problems.append(f"ошибки финального flush: {len(self._flush_errors)}")
            if self._write_errors:
                problems.append(f"невыполненные записи за сессию: {len(self._write_errors)}")
            if problems:
                first = (self._flush_errors + self._write_errors)[0]
                err = RuntimeError(
                    "AsyncDiskWriter: " + "; ".join(problems) +
                    f"; пример: {first}")

        if err is not None:
            with self._condition:
                self._state = WriterState.FAILED
                self._fatal_error = err
            if raise_on_error:
                raise err
            return False

        with self._condition:
            self._state = WriterState.STOPPED
            self._fatal_error = None
        logger.info("✅ AsyncDiskWriter: все данные сохранены на диск.")
        return True

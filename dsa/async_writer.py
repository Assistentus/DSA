import threading
import queue
import time
import numpy as np
import logging
from typing import Dict, Set

logger = logging.getLogger("DSA_AsyncWriter")


class AsyncDiskWriter:
    """
    Фоновый воркер для асинхронной записи обновлений на диск.
    Реализует Double Buffering: GPU вычисляет батч N+1, пока CPU пишет батч N.

    Гарантии (исправления F1 и F2, см. CHANGELOG.md):

    - F1: записи никогда не выбрасываются. Если очередь полна, ``put_update``
      блокируется на ``put_timeout`` секунд; если за это время место не освободилось,
      батч пишется синхронно напрямую в memmap (под локом). Потеря данных исключена.
    - F2: писатель под локом ведёт множество строк «в полёте» (в очереди либо
      в записи) по каждому имени параметра — только целочисленные индексы.
      ``intersects_inflight`` даёт оптимизатору дешёвую проверку пересечения,
      ``wait_all_flushed`` — полную очистку очереди.
    """

    def __init__(self, memmaps: Dict[str, np.memmap], max_queue_size: int = 150,
                 put_timeout: float = 5.0):
        self.memmaps = dict(memmaps)
        self.queue = queue.Queue(maxsize=max_queue_size)
        self._stop_signal = object()
        self._lock = threading.Lock()
        self._put_timeout = float(put_timeout)
        # F2: строки в полёте, по имени параметра; защищено self._lock
        self._inflight: Dict[str, Set[int]] = {}
        # F1: число синхронных fallback-записей мимо очереди (диагностика)
        self._sync_fallbacks = 0

        self.worker_thread = threading.Thread(
            target=self._process_queue, name="DSA_AsyncWriter", daemon=True)
        self.worker_thread.start()
        logger.info("💿 AsyncDiskWriter: Фоновый поток инициализирован.")

    # ─────────────────────────── F2: строки в полёте ───────────────────────────

    def intersects_inflight(self, name: str, indices) -> bool:
        """Есть ли среди `indices` строки, запись которых ещё не завершена."""
        with self._lock:
            inflight = self._inflight.get(name)
            if not inflight:
                return False
            if isinstance(indices, np.ndarray):
                return bool(inflight.intersection(indices.tolist()))
            return any(int(i) in inflight for i in indices)

    def wait_all_flushed(self, timeout: float = None):
        """Дождаться полной очистки очереди и обнуления множества строк в полёте.

        `timeout=None` — ждать без ограничений (обычный путь в обучении);
        при числовом timeout невыполнение дрена за отведённое время — ошибка.
        """
        if timeout is None:
            self.queue.join()
        else:
            if not self._wait_queue_drained(timeout):
                raise RuntimeError(
                    f"AsyncDiskWriter: очередь не опустела за {timeout:.1f} с "
                    f"(осталось батчей: {self.queue.unfinished_tasks}) — "
                    f"данные НЕ дописаны")
        with self._lock:
            leftover = {n: sorted(s) for n, s in self._inflight.items() if s}
        if leftover:
            raise RuntimeError(
                f"AsyncDiskWriter: очередь пуста, но строки остались в полёте: {leftover}")

    def _wait_queue_drained(self, timeout: float) -> bool:
        """Ограниченное по времени ожидание доработы очереди (без join-зависаний)."""
        deadline = time.monotonic() + timeout
        while True:
            if self.queue.unfinished_tasks == 0:
                return True
            if time.monotonic() >= deadline:
                return self.queue.unfinished_tasks == 0
            time.sleep(0.01)

    # ─────────────────────────── запись ───────────────────────────

    def _apply_write(self, name: str, indices, w_np, m_np, v_np, t_np=None):
        """Синхронная запись одного батча в memmap."""
        with self._lock:
            self.memmaps[f"{name}_w"][indices] = w_np
            self.memmaps[f"{name}_m"][indices] = m_np
            self.memmaps[f"{name}_v"][indices] = v_np
            if t_np is not None:
                t_mm = self.memmaps.get(f"{name}_t")
                if t_mm is None:
                    raise RuntimeError(
                        f"передан счётчик шага для '{name}', но memmap '{name}_t' не открыт")
                t_mm[indices] = t_np

    def _process_queue(self):
        """Цикл записи: работает параллельно с основным обучением."""
        while True:
            batch_data = self.queue.get()

            if batch_data is self._stop_signal:
                # Перед выходом фиксируем все данные на физическом носителе
                self._final_flush()
                self.queue.task_done()
                break

            name, indices, w_np, m_np, v_np, t_np = batch_data
            try:
                self._apply_write(name, indices, w_np, m_np, v_np, t_np)
            except Exception as e:
                logger.error(f"❌ IO Error: {e}")
            finally:
                # F2: строки этого батча больше не в полёте — до task_done,
                # чтобы после queue.join() множество гарантированно было пустым
                with self._lock:
                    inflight = self._inflight.get(name)
                    if inflight is not None:
                        inflight.difference_update(int(i) for i in indices)
                self.queue.task_done()

    def _final_flush(self):
        """Принудительная запись на диск перед выключением."""
        logger.info("💾 AsyncDiskWriter: Финализация данных на диске...")
        with self._lock:
            for mm in self.memmaps.values():
                try:
                    mm.flush()
                except:
                    pass

    def put_update(self, name: str, indices_pt, w_pt, m_pt, v_pt, t_pt=None):
        """
        Подготовка данных для записи. Выполняется в потоке GPU.

        F1: данные не теряются никогда. При полной очереди ждём место
        `put_timeout` секунд; не дождались — пишем синхронно напрямую.
        """
        # Переводим в Numpy на CPU. Это освобождает GPU тензоры немедленно.
        idx_np = np.asarray(indices_pt.detach().cpu().numpy())
        w_np = w_pt.detach().cpu().numpy()
        m_np = m_pt.detach().cpu().numpy()
        v_np = v_pt.detach().cpu().numpy()
        t_np = None if t_pt is None else np.asarray(
            t_pt.detach().cpu().numpy(), dtype=np.int32)

        idx_set = set(int(i) for i in idx_np.tolist())
        # F2: помечаем строки «в полёте» до постановки в очередь — тогда между
        # проверкой пересечения в step() и постановкой не возникает окна гонки
        with self._lock:
            self._inflight.setdefault(name, set()).update(idx_set)
        try:
            self.queue.put((name, idx_np, w_np, m_np, v_np, t_np),
                           timeout=self._put_timeout)
        except queue.Full:
            # F1: диск не успевает — пишем сами, синхронно, данные не теряем.
            self._sync_fallbacks += 1
            logger.warning(
                f"⚠️ Очередь записи полна дольше {self._put_timeout:.1f} с "
                f"('{name}', {len(idx_np)} строк): синхронная запись напрямую.")
            try:
                self._apply_write(name, idx_np, w_np, m_np, v_np, t_np)
            finally:
                # строки записаны синхронно — из полёта их снимаем
                with self._lock:
                    self._inflight.setdefault(name, set()).difference_update(idx_set)

    def shutdown(self, timeout: float = 5.0):
        """
        Завершение работы:
        1. Дописывает все оставшиеся батчи из очереди на диск.
        2. Выполняет flush().
        3. Завершает поток с таймаутом (защита от зависания msync в Linux/Kaggle).
        """
        logger.info("⏳ AsyncDiskWriter: Завершение операций и сброс на диск...")

        # 1. Отправляем сигнал остановки в очередь
        self.queue.put(self._stop_signal)

        # 2. Ждем завершения фонового потока не более timeout секунд
        self.worker_thread.join(timeout=timeout)

        if self.worker_thread.is_alive():
            logger.warning("⚠️ AsyncDiskWriter: Время ожидания msync истекло, но все данные из очереди записаны.")
        else:
            logger.info("✅ AsyncDiskWriter: Все данные успешно сохранены на диск.")

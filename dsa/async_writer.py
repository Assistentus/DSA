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

    Гарантии (исправления по разбору дефектов, см. CHANGELOG.md):

    - F1: записи никогда не выбрасываются. Если очередь полна, ``put_update``
      блокируется на ``put_timeout`` секунд; если за это время место не освободилось,
      батч пишется синхронно напрямую в memmap (под локом). Потеря данных исключена.
    - F2: писатель под локом ведёт множество строк «в полёте» (в очереди либо
      в записи) по каждому имени параметра — только целочисленные индексы.
      ``intersects_inflight`` даёт оптимизатору дешёвую проверку пересечения,
      ``wait_all_flushed`` — полную очистку очереди.
    - F5: индексы сортируются перед записью — локальность IO.
    - F6: ``shutdown`` дренирует очередь до конца, ждёт поток с достаточным
      таймаутом и при невозможности дописать данные честно бросает ошибку.
      Голых ``except: pass`` нет; все ошибки записи фиксируются и всплывают.
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
        # F6: честный учёт ошибок вместо молчаливых потерь
        self._write_errors = []
        self._flush_errors = []
        self._sync_fallbacks = 0
        self._stopped = False

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
        """Синхронная запись одного батча в memmap (F5: с сортировкой индексов)."""
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
                        f"передан счётчик шага для '{name}', но memmap '{name}_t' не открыт")
                t_mm[indices] = t_sorted

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
            except Exception as e:  # F6: ошибка фиксируется и всплывёт на shutdown
                msg = f"батч '{name}' ({len(indices)} строк) НЕ записан: {e!r}"
                self._write_errors.append(msg)
                logger.error("❌ IO Error: " + msg)
            finally:
                # F2: строки этого батча больше не в полёте — до task_done,
                # чтобы после queue.join() множество гарантированно было пустым
                with self._lock:
                    inflight = self._inflight.get(name)
                    if inflight is not None:
                        inflight.difference_update(int(i) for i in indices)
                self.queue.task_done()

    def _final_flush(self):
        """Принудительная запись на диск перед выключением (без голых except)."""
        logger.info("💾 AsyncDiskWriter: Финализация данных на диске...")
        with self._lock:
            for key, mm in self.memmaps.items():
                try:
                    mm.flush()
                except Exception as e:
                    msg = f"flush {key} не выполнен: {e!r}"
                    self._flush_errors.append(msg)
                    logger.error("❌ " + msg)

    def put_update(self, name: str, indices_pt, w_pt, m_pt, v_pt, t_pt=None):
        """
        Подготовка данных для записи. Выполняется в потоке GPU.

        F1: данные не теряются никогда. При полной очереди ждём место
        `put_timeout` секунд; не дождались — пишем синхронно напрямую.
        """
        if self._stopped:
            raise RuntimeError("AsyncDiskWriter уже остановлен: запись невозможна")

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

    def shutdown(self, timeout: float = 300.0):
        """
        Честное завершение работы (F6):
        1. Дописывает все оставшиеся батчи из очереди на диск.
        2. Выполняет flush() всех memmap.
        3. Ждёт завершения потока с достаточным таймаутом.

        Если за `timeout` данные дописать не удалось, либо за сессию были
        ошибки записи/flush — бросает RuntimeError. «Всё записано» сообщается
        только когда это правда.
        """
        if self._stopped:
            return
        self._stopped = True
        logger.info("⏳ AsyncDiskWriter: Дренаж очереди и финальный сброс на диск...")

        # 1. Всё, что уже в очереди, будет записано (дрен тоже ограничен
        #    таймаутом: застрявший диск — честная ошибка, а не вечное ожидание)
        if not self._wait_queue_drained(timeout):
            raise RuntimeError(
                f"AsyncDiskWriter: за {timeout:.1f} с очередь не опустела "
                f"(осталось батчей: {self.queue.unfinished_tasks}) — "
                f"данные НЕ дописаны на диск")
        # 2. Сигнал остановки: воркер сделает финальный flush и выйдет
        #    (после дрена очередь пуста, put не блокируется)
        self.queue.put(self._stop_signal)
        # 3. Ждём завершения фонового потока (финальный msync может быть долгим)
        self.worker_thread.join(timeout=timeout)

        if self.worker_thread.is_alive():
            raise RuntimeError(
                f"AsyncDiskWriter: фоновый поток не завершился за {timeout:.1f} с — "
                f"данные НЕ гарантированно записаны на диск")

        problems = []
        if self._flush_errors:
            problems.append(f"ошибки финального flush: {len(self._flush_errors)}")
        if self._write_errors:
            problems.append(f"невыполненные записи за сессию: {len(self._write_errors)}")
        if problems:
            first = (self._flush_errors + self._write_errors)[0]
            raise RuntimeError(
                "AsyncDiskWriter: " + "; ".join(problems) + f"; пример: {first}")
        logger.info("✅ AsyncDiskWriter: Все данные успешно сохранены на диск.")

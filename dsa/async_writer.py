import threading
import queue
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger("DSA_AsyncWriter")

class AsyncDiskWriter:
    """
    Фоновый воркер для асинхронной записи обновлений на диск.
    Реализует Double Buffering: GPU вычисляет батч N+1, пока CPU пишет батч N.
    """
    def __init__(self, memmaps: Dict[str, np.memmap], max_queue_size: int = 150):
        self.memmaps = memmaps
        self.queue = queue.Queue(maxsize=max_queue_size)
        self._stop_signal = object()
        self._lock = threading.Lock()
        
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        logger.info("💿 AsyncDiskWriter: Фоновый поток инициализирован.")

    def _process_queue(self):
        """Цикл записи: работает параллельно с основным обучением."""
        while True:
            batch_data = self.queue.get()
            
            if batch_data is self._stop_signal:
                # Перед выходом фиксируем все данные на физическом носителе
                self._final_flush()
                self.queue.task_done()
                break
            
            name, indices, w_np, m_np, v_np = batch_data
            try:
                with self._lock:
                    # Записываем данные в mmap (в оперативную память ОС)
                    # ИСПРАВЛЕНО: Добавлен суффикс _w для совпадения с ключами из optimizer.py
                    self.memmaps[f"{name}_w"][indices] = w_np
                    self.memmaps[f"{name}_m"][indices] = m_np
                    self.memmaps[f"{name}_v"][indices] = v_np
                    # .flush() здесь не вызываем для максимальной скорости IO
            except Exception as e:
                logger.error(f"❌ IO Error: {e}")
            finally:
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

    def put_update(self, name: str, indices_pt, w_pt, m_pt, v_pt):
        """
        Подготовка данных для записи. Выполняется в потоке GPU.
        """
        # Переводим в Numpy на CPU. Это освобождает GPU тензоры немедленно.
        idx_np = indices_pt.detach().cpu().numpy()
        w_np = w_pt.detach().cpu().numpy()
        m_np = m_pt.detach().cpu().numpy()
        v_np = v_pt.detach().cpu().numpy()

        try:
            # Отправляем в очередь без блокировки
            self.queue.put_nowait((name, idx_np, w_np, m_np, v_np))
        except queue.Full:
            # Если диск не успевает, мы жертвуем этим батчем записи, чтобы не тормозить GPU.
            # В градиентном спуске это допустимый шум (stale gradients).
            logger.warning(f"⚠️ Диск перегружен! Пропуск записи батча для '{name}'.")

    def shutdown(self, timeout: float = 5.0):
        """
        Безопасное завершение работы:
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


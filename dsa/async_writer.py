import threading
import queue
import logging

logger = logging.getLogger("DSA_AsyncWriter")

class AsyncDiskWriter:
    """
    Фоновый воркер для асинхронной записи тензоров в np.memmap.
    Освобождает GPU/CPU от блокировок дисковыми I/O операциями.
    """
    def __init__(self, max_queue_size: int = 150):
        self.queue = queue.Queue(maxsize=max_queue_size)
        self._stop_signal = object()  # Уникальный Sentinel для гарантии остановки
        self._lock = threading.Lock() # Мьютекс для защиты дисковых транзакций
        
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        logger.info("💿 AsyncDiskWriter started in background.")

    def _process_queue(self):
        while True:
            batch_data = self.queue.get()
            
            # Обработка сигнала завершения
            if batch_data is self._stop_signal:
                self.queue.task_done()
                break
            
            name, indices, w_np, m_np, v_np, memmaps = batch_data
            try:
                # Атомарная и безопасная запись на диск
                with self._lock:
                    memmaps[name][indices] = w_np
                    memmaps[f"{name}_m"][indices] = m_np
                    memmaps[f"{name}_v"][indices] = v_np
                    
                    # Принудительный сброс буферов OS на железо (SSD)
                    memmaps[name].flush()
                    memmaps[f"{name}_m"].flush()
                    memmaps[f"{name}_v"].flush()
            except Exception as e:
                logger.error(f"❌ IO Error during write-back: {e}")
            finally:
                self.queue.task_done()

    def put_update(self, name: str, indices_pt, w_pt, m_pt, v_pt, memmaps_dict: dict):
        """
        Отрывает тензоры от графа вычислений, переводит в numpy и отправляет в очередь.
        """
        indices_np = indices_pt.detach().cpu().numpy()
        w_np = w_pt.detach().cpu().numpy()
        m_np = m_pt.detach().cpu().numpy()
        v_np = v_pt.detach().cpu().numpy()

        try:
            # put_nowait позволяет GPU сразу идти дальше, если диск захлебнулся
            self.queue.put_nowait((name, indices_np, w_np, m_np, v_np, memmaps_dict))
        except queue.Full:
            logger.warning(f"⚠️ IO Queue full! Dropping write batch for '{name}'. SSD is too slow.")

    def shutdown(self):
        """Гарантирует дописывание очереди перед выходом из программы."""
        logger.info("⏳ AsyncDiskWriter: waiting for IO queue to flush...")
        self.queue.put(self._stop_signal)
        self.queue.join()
        self.worker_thread.join()
        logger.info("✅ AsyncDiskWriter: Shutdown complete. All data saved.")

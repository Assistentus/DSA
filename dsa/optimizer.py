import os
import json
import math
import logging
from datetime import datetime

import torch
import numpy as np
from torch.optim.optimizer import Optimizer
from typing import Dict, Tuple, Optional, Union

from .async_writer import AsyncDiskWriter

logger = logging.getLogger("DSA")

# Версия формата дискового состояния (meta-файлы). Несовпадение — явная ошибка.
STATE_FORMAT_VERSION = 1


class DiskSparseRiemannianAdam(Optimizer):
    """
    Disk-based Sparse Riemannian Adam (DSA).

    Специализированный оптимизатор для обучения сверхбольших графовых эмбеддингов.
    - Хранит состояния (m, v) и веса (w) на NVMe через np.memmap.
    - Масштабирует использование памяти от размера батча O(B), а не графа O(N).
    - Реализует Riemannian Gradient Scaling и проекцию (Retraction) для шара Пуанкаре.
    - Использует фоновую асинхронную запись для устранения простоев GPU.

    Гарантии корректности (см. CHANGELOG.md): записи никогда не выбрасываются
    при полной очереди (F1); step() не читает строки, находящиеся в полёте
    у писателя, без дрена очереди (F2); повторное открытие каталога состояния
    управляется явным `resume` и meta-файлами (F3).
    """

    def __init__(self, params: Union[list, dict], lr=1e-3, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0, k=1.0, disk_dir="./disk_cache",
                 max_queue_size=150, resume: bool = False,
                 put_timeout: float = 5.0):

        if isinstance(params, dict):
            self.param_names = list(params.keys())
            params_list = list(params.values())
        else:
            params_list = list(params)
            self.param_names = [f"param_{i}" for i in range(len(params_list))]

        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps,
                        weight_decay=weight_decay, k=k)

        super().__init__(params_list, defaults)

        self.disk_dir = os.path.abspath(disk_dir)
        os.makedirs(self.disk_dir, exist_ok=True)

        self.state_files = {}
        self._paths = {}

        # ── Инициализация/валидация хранилища на диске (F3) ──────────────────
        for name, param in zip(self.param_names, self.param_groups[0]['params']):
            shape = tuple(param.shape)
            paths = {
                'w': os.path.join(self.disk_dir, f"{name}_w.dat"),
                'm': os.path.join(self.disk_dir, f"{name}_m.dat"),
                'v': os.path.join(self.disk_dir, f"{name}_v.dat"),
            }
            meta_path = os.path.join(self.disk_dir, f"{name}_meta.json")
            self._init_param_storage(name, shape, paths, meta_path, param, resume)
            self._paths[name] = paths

            self.state_files[name] = {
                'w': np.memmap(paths['w'], dtype='float32', mode='r+', shape=shape),
                'm': np.memmap(paths['m'], dtype='float32', mode='r+', shape=shape),
                'v': np.memmap(paths['v'], dtype='float32', mode='r+', shape=shape),
            }

        # ── Запуск фонового писателя ─────────────────────────────────────────
        self.writer = AsyncDiskWriter(
            memmaps={f"{n}_{s}": self.state_files[n][s]
                     for n in self.param_names for s in ['w', 'm', 'v']},
            max_queue_size=max_queue_size,
            put_timeout=put_timeout,
        )

    # ─────────────────────── дисковое состояние (F3) ───────────────────────

    def _write_meta(self, meta_path: str, name: str, shape, paths: dict):
        meta = {
            'format_version': STATE_FORMAT_VERSION,
            'param_name': name,
            'shape': list(shape),
            'dtype': 'float32',
            'files': {s: os.path.basename(p) for s, p in paths.items()},
            'updated': datetime.now().isoformat(timespec='seconds'),
        }
        tmp = meta_path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, meta_path)

    def _load_meta(self, meta_path: str, name: str) -> dict:
        if not os.path.exists(meta_path):
            raise RuntimeError(
                f"для параметра '{name}' в {self.disk_dir} есть файлы состояния, но нет "
                f"meta-файла {os.path.basename(meta_path)}: происхождение состояния "
                f"неизвестно. Удалите каталог или восстановите meta.")
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        except (OSError, ValueError) as e:
            raise RuntimeError(f"meta-файл {meta_path} не читается: {e!r}") from e
        if int(meta.get('format_version', -1)) != STATE_FORMAT_VERSION:
            raise RuntimeError(
                f"meta-файл {meta_path}: версия формата {meta.get('format_version')!r} "
                f"не поддерживается (ожидается {STATE_FORMAT_VERSION})")
        return meta

    def _verify_state_file_sizes(self, name, shape, paths):
        """Сверка фактических размеров файлов состояния с формой из meta (F3).

        np.memmap(dtype float32, mode='r+', shape) молча продлевает короткий
        файл нулями до запрошенного размера (и молча отображает только начало
        слишком длинного), поэтому обрезанный при сбое диска/копирования файл
        без этой проверки открылся бы штатно: потерянные строки читались бы
        как нули и тихо смешивались со старыми.
        """
        row_bytes = int(np.prod(shape)) * 4      # float32: w, m, v
        expected = {'w': row_bytes, 'm': row_bytes, 'v': row_bytes}
        for suffix, exp in expected.items():
            actual = os.path.getsize(paths[suffix])
            if actual != exp:
                raise RuntimeError(
                    f"файл состояния DSA {paths[suffix]} повреждён: размер "
                    f"{actual} байт, по форме {list(shape)} ожидается {exp} "
                    f"(состояние обрезано — сбой диска/копирования — либо файл "
                    f"чужой). resume=True невозможен: восстановите файл из "
                    f"резервной копии/чекпоинта или переоткройте с "
                    f"resume=False (полная переинициализация).")

    def _write_fresh_state(self, name, shape, paths, param):
        """Полная (пере)инициализация: w из params, m=v=0, свежий meta."""
        for suffix in ('w', 'm', 'v'):
            mm = np.memmap(paths[suffix], dtype='float32', mode='w+', shape=shape)
            if suffix == 'w':
                mm[:] = param.detach().cpu().numpy().astype('float32')
            else:
                mm[:] = 0.0
            mm.flush()
            del mm

    def _init_param_storage(self, name, shape, paths, meta_path, param, resume: bool):
        exists = {s: os.path.exists(p) for s, p in paths.items()}

        if not any(exists.values()):
            # Новый каталог — честная инициализация с нуля
            self._write_fresh_state(name, shape, paths, param)
            self._write_meta(meta_path, name, shape, paths)
            return

        # На диске что-то есть — валидируем строго (F3: никакого тихого переиспользования)
        if not all(exists.values()):
            missing = [s for s, e in exists.items() if not e]
            raise RuntimeError(
                f"частичное состояние DSA для '{name}' в {self.disk_dir}: отсутствуют "
                f"{missing}. Состояние ненадёжно — удалите каталог целиком или "
                f"восстановите все файлы ({', '.join(paths.values())}).")

        meta = self._load_meta(meta_path, name)
        disk_shape = tuple(meta.get('shape', []))
        if disk_shape != tuple(shape):
            raise RuntimeError(
                f"несовпадение формы состояния DSA для '{name}': на диске "
                f"{list(disk_shape)}, в параметрах {list(shape)}. Удалите "
                f"{self.disk_dir} или укажите новый disk_dir.")
        if meta.get('dtype') != 'float32':
            raise RuntimeError(
                f"meta-файл {meta_path}: неожиданный тип данных "
                f"(dtype={meta.get('dtype')!r}), ожидался float32.")

        if resume:
            # Явное продолжение: существующее состояние используется как есть,
            # w из params НЕ перезаписывает диск.
            # До открытия memmap сверяем фактические размеры файлов: np.memmap
            # в режиме r+ молча продлевает обрезанный файл нулями (сбой диска
            # или копирования), и потерянные строки тихо смешались бы со
            # старыми как нулевые (начало шара) — без ошибки и без лога.
            self._verify_state_file_sizes(name, shape, paths)
            logger.info(
                f"DSA '{name}': продолжаем существующее состояние {list(shape)} "
                f"из {self.disk_dir} (resume=True)")
        else:
            logger.warning(
                f"DSA '{name}': в {self.disk_dir} найдено состояние {list(shape)}; "
                f"resume=False → полная переинициализация: w берётся из params, "
                f"m=v=0.")
            self._write_fresh_state(name, shape, paths, param)
            self._write_meta(meta_path, name, shape, paths)

    @torch.no_grad()
    def step(self, updates: Dict[str, Tuple[torch.Tensor, torch.Tensor]], 
             current_k: Optional[Union[float, torch.Tensor]] = None):
        """
        updates: словарь {'param_name': (indices_tensor, grads_tensor)}
        """
        # Глобальный счетчик шагов хранится в первом параметре для совместимости
        param_base = self.param_groups[0]['params'][0]
        state = self.state[param_base]
        if len(state) == 0:
            state['step'] = 0
        state['step'] += 1
        
        group = self.param_groups[0]
        lr = group['lr']
        beta1, beta2 = group['beta1'], group['beta2']
        eps = group['eps']
        weight_decay = group['weight_decay']
        
        bc1 = 1.0 - beta1 ** state['step']
        bc2 = 1.0 - beta2 ** state['step']
        
        k = current_k.item() if isinstance(current_k, torch.Tensor) else (current_k if current_k is not None else group['k'])

        for name, (indices, grads) in updates.items():
            if name not in self.state_files or indices.numel() == 0:
                continue
            
            # 1. Агрегация градиентов для дублирующихся индексов в батче
            if indices.numel() != torch.unique(indices).numel():
                unique_idx, inv = torch.unique(indices, return_inverse=True)
                agg_grads = torch.zeros((len(unique_idx), grads.shape[1]), device=grads.device, dtype=grads.dtype)
                agg_grads.index_add_(0, inv, grads)
                active_indices, active_grads = unique_idx, agg_grads
            else:
                active_indices, active_grads = indices, grads

            idx_np = active_indices.cpu().numpy()
            states = self.state_files[name]

            # F2: не читать строки, которые писатель ещё не дописал.
            # Пересечение пусто → читаем сразу (асинхронность сохраняется);
            # непусто → ждём полной очистки очереди и только потом читаем.
            if self.writer.intersects_inflight(name, idx_np):
                self.writer.wait_all_flushed()
            
            # 2. Чтение из mmap в оперативную память (только нужные строки)
            w = torch.from_numpy(states['w'][idx_np].copy())
            m = torch.from_numpy(states['m'][idx_np].copy())
            v = torch.from_numpy(states['v'][idx_np].copy())
            g_euc = active_grads.cpu().float()

            # 3. Riemannian Gradient Scaling (Poincaré Ball)
            if k > 1e-9:
                w_sq_norm = torch.sum(w * w, dim=-1, keepdim=True)
                max_radius_sq = (1.0 - 1e-5) / k
                w_sq_norm = torch.clamp(w_sq_norm, max=max_radius_sq)
                rescale_factor = ((1.0 - k * w_sq_norm) ** 2) / 4.0
                g_riem = g_euc * rescale_factor
            else:
                g_riem = g_euc

            # 4. Adam Step
            if weight_decay > 0:
                w.mul_(1.0 - lr * weight_decay)

            m.mul_(beta1).add_(g_riem, alpha=1.0 - beta1)
            v.mul_(beta2).addcmul_(g_riem, g_riem, value=1.0 - beta2)

            m_hat = m / bc1
            v_hat = v / bc2
            step_size = lr * m_hat / (torch.sqrt(v_hat) + eps)
            w.sub_(step_size)

            # 5. Retraction (удержание на многообразии)
            if k > 1e-9:
                w_norm = torch.norm(w, p=2, dim=-1, keepdim=True)
                max_radius = math.sqrt((1.0 - 1e-5) / k)
                w = torch.where(w_norm > max_radius, w / w_norm * max_radius, w)

            # 6. Передача на асинхронную фоновую запись
            self.writer.put_update(name, active_indices, w, m, v)

    def shutdown(self):
        """Вызывать в конце обучения для финализации записи на диск."""
        if hasattr(self, 'writer'):
            self.writer.shutdown()
            
    def zero_grad(self, set_to_none: bool = False):
        """Заглушка для совместимости, так как мы работаем с явными градиентами."""
        pass

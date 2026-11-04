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

# R3: отладочный режим: неизвестный ключ updates — исключение вместо warning.
# Включается переменной окружения DSA_DEBUG=1 либо присвоением
# dsa.optimizer.DEBUG = True.
DEBUG = os.environ.get("DSA_DEBUG", "0") not in ("", "0")


def _project_rows_to_ball(w: torch.Tensor, k: float) -> torch.Tensor:
    """Проекция строк в шар Пуанкаре радиуса sqrt((1-1e-5)/k) (F7 и ретракция)."""
    max_radius = math.sqrt((1.0 - 1e-5) / k)
    norm = torch.norm(w, p=2, dim=-1, keepdim=True)
    return torch.where(norm > max_radius, w / norm * max_radius, w)


class DiskSparseRiemannianAdam(Optimizer):
    """
    Disk-based Sparse Riemannian Adam (DSA).

    Специализированный оптимизатор для обучения сверхбольших графовых эмбеддингов.
    - Хранит состояния (m, v), веса (w) и пер-строчные счётчики шагов (t) на NVMe
      через np.memmap.
    - Масштабирует использование памяти от размера батча O(B), а не графа O(N).
    - Реализует Riemannian Gradient Scaling и проекцию (Retraction) для шара Пуанкаре.
    - Использует фоновую асинхронную запись для устранения простоев GPU.

    Гарантии корректности (см. CHANGELOG.md): записи никогда не выбрасываются
    при полной очереди (F1); step() не читает строки, находящиеся в полёте
    у писателя, без дрена очереди (F2); повторное открытие каталога состояния
    управляется явным `resume` и meta-файлами (F3);
    bias-коррекция Adam — по пер-строчному счётчику шага, а не по глобальному t
    (F4); индексы сортируются перед IO (F5); чтение весов проецируется в шар (F7);
    чекпоинты — save_checkpoint/load_checkpoint, state_dict() честно бросает
    NotImplementedError (F8). Дополнительно: не-конечные строки из битого файла
    обрабатываются на чтении (R1), переинициализация помечается маркером в meta
    (R2), неизвестные ключи updates дают warning со счётчиком (R3).
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
        # R1/R3: счётчики диагностики (не-конечные строки, неизвестные ключи)
        self._nan_rows_replaced = 0
        self._unknown_key_warnings = 0

        # ── Инициализация/валидация хранилища на диске (F3) ──────────────────
        for name, param in zip(self.param_names, self.param_groups[0]['params']):
            shape = tuple(param.shape)
            paths = {
                'w': os.path.join(self.disk_dir, f"{name}_w.dat"),
                'm': os.path.join(self.disk_dir, f"{name}_m.dat"),
                'v': os.path.join(self.disk_dir, f"{name}_v.dat"),
                't': os.path.join(self.disk_dir, f"{name}_t.dat"),
            }
            meta_path = os.path.join(self.disk_dir, f"{name}_meta.json")
            self._init_param_storage(name, shape, paths, meta_path, param, resume)
            self._paths[name] = paths

            self.state_files[name] = {
                'w': np.memmap(paths['w'], dtype='float32', mode='r+', shape=shape),
                'm': np.memmap(paths['m'], dtype='float32', mode='r+', shape=shape),
                'v': np.memmap(paths['v'], dtype='float32', mode='r+', shape=shape),
                # F4: пер-строчные счётчики шага, int32, N×4 байта
                't': np.memmap(paths['t'], dtype='int32', mode='r+', shape=(shape[0],)),
            }

        # ── Запуск фонового писателя ─────────────────────────────────────────
        self.writer = AsyncDiskWriter(
            memmaps={f"{n}_{s}": self.state_files[n][s]
                     for n in self.param_names for s in ['w', 'm', 'v', 't']},
            max_queue_size=max_queue_size,
            put_timeout=put_timeout,
        )

    # ─────────────────────── дисковое состояние (F3) ───────────────────────

    def _write_meta(self, meta_path: str, name: str, shape, paths: dict,
                    reinit_in_progress: bool = False):
        meta = {
            'format_version': STATE_FORMAT_VERSION,
            'param_name': name,
            'shape': list(shape),
            'dtype': 'float32',
            'counter_dtype': 'int32',
            'files': {s: os.path.basename(p) for s, p in paths.items()},
            'updated': datetime.now().isoformat(timespec='seconds'),
        }
        if reinit_in_progress:
            # R2: маркер незавершённой переинициализации
            meta['reinit_in_progress'] = True
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
        как нули и тихо смешивались со старыми. Та же проверка выполняется
        в load_checkpoint (_verify_checkpoint_file); здесь — штатный путь
        открытия с resume=True.
        """
        row_bytes = int(np.prod(shape)) * 4      # float32: w, m, v
        expected = {'w': row_bytes, 'm': row_bytes, 'v': row_bytes,
                    't': int(shape[0]) * 4}      # int32: пер-строчные счётчики
        for suffix, exp in expected.items():
            actual = os.path.getsize(paths[suffix])
            if actual != exp:
                raise RuntimeError(
                    f"файл состояния DSA {paths[suffix]} повреждён: размер "
                    f"{actual} байт, по форме {list(shape)} ожидается {exp} "
                    f"(состояние обрезано — сбой диска/копирования — либо файл "
                    f"чужой). resume=True невозможен: восстановите файл из "
                    f"резервной копии/чекпоинта или переоткройтесь с "
                    f"resume=False (полная переинициализация).")

    def _write_fresh_state(self, name, shape, paths, param):
        """Полная (пере)инициализация: w из params, m=v=0, счётчики 0, свежий meta."""
        for suffix in ('w', 'm', 'v'):
            mm = np.memmap(paths[suffix], dtype='float32', mode='w+', shape=shape)
            if suffix == 'w':
                mm[:] = param.detach().cpu().numpy().astype('float32')
            else:
                mm[:] = 0.0
            mm.flush()
            del mm
        t = np.memmap(paths['t'], dtype='int32', mode='w+', shape=(shape[0],))
        t[:] = 0
        t.flush()
        del t

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
        if meta.get('reinit_in_progress'):
            # R2: процесс умер посреди переинициализации — часть файлов уже
            # перезаписана, часть осталась от старого состояния. Принимать
            # такое молча нельзя: resume=True — явная ошибка, resume=False —
            # штатное восстановление полной переинициализацией.
            if resume:
                raise RuntimeError(
                    f"meta-файл {meta_path}: состояние помечено "
                    f"'reinit_in_progress' — предыдущая переинициализация не "
                    f"завершилась (сбой процесса между записью файлов и meta), "
                    f"файлы могут быть смешанными. resume=True невозможен: "
                    f"удалите {self.disk_dir} целиком или переоткройте с "
                    f"resume=False — полная переинициализация пересоздаст "
                    f"файлы и снимет маркер.")
            logger.warning(
                f"DSA '{name}': предыдущая переинициализация не завершилась "
                f"(маркер 'reinit_in_progress'); resume=False → полная "
                f"переинициализация, маркер будет снят.")
        disk_shape = tuple(meta.get('shape', []))
        if disk_shape != tuple(shape):
            raise RuntimeError(
                f"несовпадение формы состояния DSA для '{name}': на диске "
                f"{list(disk_shape)}, в параметрах {list(shape)}. Удалите "
                f"{self.disk_dir} или укажите новый disk_dir.")
        if meta.get('dtype') != 'float32' or meta.get('counter_dtype') != 'int32':
            raise RuntimeError(
                f"meta-файл {meta_path}: неожиданные типы данных "
                f"(w/m/v={meta.get('dtype')!r}, t={meta.get('counter_dtype')!r}), "
                f"ожидались float32/int32.")

        if resume:
            # Явное продолжение: существующее состояние используется как есть,
            # w из params НЕ перезаписывает диск, счётчики шагов сохраняются.
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
                f"m=v=0, счётчики шагов сброшены в 0.")
            # R2: маркер ставится ДО перезаписи файлов и снимается финальной
            # записью meta — смерть процесса посреди переинициализации не
            # оставит смешанное состояние, выглядящее как валидное
            self._write_meta(meta_path, name, shape, paths,
                             reinit_in_progress=True)
            self._write_fresh_state(name, shape, paths, param)
            self._write_meta(meta_path, name, shape, paths)

    # ─────────────────────────────── шаг ───────────────────────────────

    @torch.no_grad()
    def step(self, updates: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
             current_k: Optional[Union[float, torch.Tensor]] = None):
        """
        updates: словарь {'param_name': (indices_tensor, grads_tensor)}
        """
        # Глобальный счетчик шагов хранится в первом параметре для совместимости
        # (в bias-коррекции он больше не участвует — она пер-строчная, F4)
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

        k = current_k.item() if isinstance(current_k, torch.Tensor) else (
            current_k if current_k is not None else group['k'])

        for name, (indices, grads) in updates.items():
            if name not in self.state_files:
                # R3: молча отброшенное обновление — потерянные данные без
                # сообщения. Warning со счётчиком; в отладочном режиме —
                # исключение (счётчик не растёт: warning не выдавался).
                msg = (f"step(updates): неизвестный ключ '{name}' — обновление "
                       f"пропущено (известные ключи: {sorted(self.state_files)}); "
                       f"вероятна опечатка в имени параметра")
                if DEBUG:
                    raise KeyError(msg)
                self._unknown_key_warnings += 1
                logger.warning(msg)
                continue
            if indices.numel() == 0:
                continue

            # 1. Агрегация градиентов для дублирующихся индексов в батче
            if indices.numel() != torch.unique(indices).numel():
                unique_idx, inv = torch.unique(indices, return_inverse=True)
                agg_grads = torch.zeros((len(unique_idx), grads.shape[1]),
                                        device=grads.device, dtype=grads.dtype)
                agg_grads.index_add_(0, inv, grads)
                active_indices, active_grads = unique_idx, agg_grads
            else:
                active_indices, active_grads = indices, grads

            # F5: сортировка индексов — локальность и чтения, и записи
            if active_indices.numel() > 1:
                order = torch.argsort(active_indices)
                active_indices = active_indices[order]
                active_grads = active_grads[order]

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
            # F4: пер-строчный счётчик шага; +1 = этот шаг
            t_row = torch.from_numpy(
                states['t'][idx_np].copy().astype('int64')) + 1
            g_euc = active_grads.cpu().float()

            # R1: не-конечные строки из битого файла не проходят молча:
            # torch.where(norm > max) при NaN даёт False, и NaN расползся бы
            # в m/v и далее в обучение. Строка обнуляется (как у свежей строки),
            # проекция в шар ниже удерживает геометрию.
            bad = torch.nonzero(
                ~(torch.isfinite(w).all(dim=-1) & torch.isfinite(m).all(dim=-1)
                  & torch.isfinite(v).all(dim=-1))).flatten()
            if bad.numel() > 0:
                self._nan_rows_replaced += int(bad.numel())
                logger.warning(
                    f"DSA '{name}': {int(bad.numel())} строк с NaN/inf в файлах "
                    f"состояния (строки диска {active_indices[bad][:10].tolist()}); "
                    f"обнулены и спроецированы в шар")
                w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
                m = torch.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
                v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

            # F7: защита чтения — битый файл не должен расходить геометрию:
            # прочитанные веса сразу проецируются в шар
            if k > 1e-9:
                w = _project_rows_to_ball(w, k)

            # 3. Riemannian Gradient Scaling (Poincaré Ball)
            if k > 1e-9:
                w_sq_norm = torch.sum(w * w, dim=-1, keepdim=True)
                max_radius_sq = (1.0 - 1e-5) / k
                w_sq_norm = torch.clamp(w_sq_norm, max=max_radius_sq)
                rescale_factor = ((1.0 - k * w_sq_norm) ** 2) / 4.0
                g_riem = g_euc * rescale_factor
            else:
                g_riem = g_euc

            # F4: bias-коррекция по счётчику СТРОКИ, а не по глобальному шагу.
            # Первый шаг свежей строки даёт lr*sign(g): m_hat=g, v_hat=g^2.
            bc1 = (1.0 - torch.pow(torch.full_like(t_row, beta1, dtype=torch.float64),
                                   t_row.double())).to(torch.float32).unsqueeze(-1)
            bc2 = (1.0 - torch.pow(torch.full_like(t_row, beta2, dtype=torch.float64),
                                   t_row.double())).to(torch.float32).unsqueeze(-1)

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
                w = _project_rows_to_ball(w, k)

            # 6. Передача на асинхронную фоновую запись (счётчик — вместе
            # с w/m/v, одной очередью, чтобы состояние осталось согласованным)
            self.writer.put_update(name, active_indices, w, m, v,
                                   t_row.to(torch.int32))

    # ─────────────────────── чекпоинты (F8) ───────────────────────

    def save_checkpoint(self, path: str) -> str:
        """
        Явный чекпоинт: дождаться записи всей очереди, flush всех файлов
        состояния (w/m/v/t) и атомарно записать манифест путей в `path`.
        Возвращает абсолютный путь манифеста.
        """
        path = os.path.abspath(path)
        self.writer.wait_all_flushed()
        param_base = self.param_groups[0]['params'][0]
        with self.writer._lock:
            manifest = {
                'format': 'DSA_checkpoint',
                'version': 1,
                'saved_at': datetime.now().isoformat(timespec='seconds'),
                'global_step': int(self.state[param_base].get('step', 0)),
                'params': {},
            }
            for name in self.param_names:
                files = {}
                for suffix, mm in self.state_files[name].items():
                    mm.flush()
                    files[suffix] = self._paths[name][suffix]
                manifest['params'][name] = {
                    'shape': list(self.state_files[name]['w'].shape),
                    'files': files,
                }
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            tmp = path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        return path

    def load_checkpoint(self, path: str) -> dict:
        """
        Восстановление из манифеста save_checkpoint. Файлы чекпоинта должны
        лежать по тем же путям, на которых открыт оптимизатор (обычный сценарий:
        создать DiskSparseRiemannianAdam(..., resume=True) на том же disk_dir и
        вызвать load_checkpoint(манифест) для проверки целостности и переоткрытия
        memmap). Чекпоинт из другого каталога не подхватывается молча — это
        явная ошибка: скопируйте файлы в disk_dir и откройтесь с resume=True.
        Проверяются существование файлов, их размеры и формы.
        """
        path = os.path.abspath(path)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
        except (OSError, ValueError) as e:
            raise RuntimeError(f"чекпоинт {path} не читается: {e!r}") from e
        if manifest.get('format') != 'DSA_checkpoint':
            raise RuntimeError(f"{path} не является чекпоинтом DSA "
                               f"(format={manifest.get('format')!r})")

        self.writer.wait_all_flushed()
        new_files = {}
        with self.writer._lock:
            for name, param in zip(self.param_names, self.param_groups[0]['params']):
                info = manifest['params'].get(name)
                if info is None:
                    raise RuntimeError(f"чекпоинт {path} не содержит параметр '{name}'")
                shape = tuple(param.shape)
                if tuple(info['shape']) != shape:
                    raise RuntimeError(
                        f"несовпадение формы для '{name}': чекпоинт "
                        f"{list(info['shape'])}, оптимизатор {list(shape)}")
                for suffix in ('w', 'm', 'v'):
                    expected = int(np.prod(shape)) * 4  # float32
                    self._verify_checkpoint_file(name, suffix, info, expected)
                self._verify_checkpoint_file(name, 't', info, shape[0] * 4)  # int32

                new_files[name] = {
                    s: np.memmap(self._paths[name][s],
                                 dtype='float32' if s != 't' else 'int32',
                                 mode='r+',
                                 shape=shape if s != 't' else (shape[0],))
                    for s in ('w', 'm', 'v', 't')
                }
            self.state_files = new_files
            self.writer.memmaps = {f"{n}_{s}": mm
                                   for n, mm2 in new_files.items()
                                   for s, mm in mm2.items()}
        return manifest

    def _verify_checkpoint_file(self, name, suffix, info, expected_bytes: int):
        p = info['files'].get(suffix)
        mine = os.path.abspath(self._paths[name][suffix])
        if p is None:
            raise RuntimeError(f"чекпоинт не содержит файл '{suffix}' для '{name}'")
        if os.path.abspath(p) != mine:
            raise RuntimeError(
                f"чекпоинт ссылается на {p}, а оптимизатор открыт на {mine}: "
                f"скопируйте файлы чекпоинта в disk_dir и создайте оптимизатор "
                f"с resume=True")
        if not os.path.exists(p):
            raise RuntimeError(f"файл чекпоинта отсутствует: {p}")
        actual = os.path.getsize(p)
        if actual != expected_bytes:
            raise RuntimeError(
                f"файл {p}: размер {actual} байт, ожидалось {expected_bytes} "
                f"— состояние повреждено")

    # F8: молчаливо неверного state_dict не отдаём — только честная ошибка
    def state_dict(self):
        raise NotImplementedError(
            "DSA хранит состояние на диске (файлы <name>_w/_m/_v/_t.dat в "
            f"{self.disk_dir}); сериализовать его через state_dict() нельзя. "
            "Используйте save_checkpoint()/load_checkpoint() либо shutdown() "
            "и повторное открытие с resume=True.")

    def load_state_dict(self, state_dict):
        raise NotImplementedError(
            "DSA восстанавливается с диска: создайте оптимизатор с resume=True "
            "на том же disk_dir (и при необходимости load_checkpoint()). "
            "load_state_dict() не поддерживается.")

    def shutdown(self):
        """Вызывать в конце обучения для финализации записи на диск."""
        if hasattr(self, 'writer'):
            self.writer.shutdown()

    def zero_grad(self, set_to_none: bool = False):
        """Заглушка для совместимости, так как мы работаем с явными градиентами."""
        pass

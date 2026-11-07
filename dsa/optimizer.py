"""Disk-based sparse Riemannian Adam (DSA).

The optimizer keeps ``w`` (weights), ``m`` (first moment), ``v`` (second
moment) and a per-row step counter ``t`` in ``np.memmap`` files, so only the
rows touched by a batch are read into RAM/VRAM.

Hyperbolic mode (``k > 0``) implements the Riemannian Adam of Becigneul &
Ganea (ICLR 2019) on the Poincare ball of curvature ``-k``:

* the Euclidean gradient is converted to a Riemannian one at the current
  point (``lambda(x)^-2 g``);
* Adam moments live in the tangent space at the current point;
* the step is an exponential map, not a Euclidean subtraction plus clip;
* after the step the moments are parallel-transported to the new tangent
  space.

``k = 0`` is plain Euclidean Adam.  See ``README.md`` and ``CHANGELOG.md``
for the exact semantics and the 0.3.0 migration notes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.optim.optimizer import Optimizer

from .async_writer import AsyncDiskWriter
from .manifold import (
    BALL_EPS,
    expmap,
    parallel_transport,
    project_to_ball,
    project_to_ball_np,
    riemannian_gradient,
)

logger = logging.getLogger("DSA")

# Version of the per-parameter and global metadata formats.
STATE_FORMAT_VERSION = 2
LEGACY_STATE_FORMAT_VERSIONS = (1,)
ALGORITHM = "riemannian_adam"

CHECKPOINT_FORMAT = "DSA_checkpoint"
CHECKPOINT_VERSION = 2
CHECKPOINT_MANIFEST_NAME = "checkpoint.json"
GLOBAL_META_NAME = "dsa_global_meta.json"

# Number of rows processed at once by whole-table operations.
CHUNK_ROWS = 65536

# R3: debug mode: an unknown key in ``updates`` raises instead of warning.
DEBUG = os.environ.get("DSA_DEBUG", "0") not in ("", "0")

WEIGHT_DECAY_MODES = ("l2", "decoupled")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _validate_k(k: Any, where: str = "k") -> float:
    """Return a finite non-negative curvature; reject NaN/inf/negative early."""
    try:
        value = float(k.item() if isinstance(k, torch.Tensor) else k)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{where} must be a real number, got {k!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{where} must be finite, got {value!r}")
    if value < 0.0:
        raise ValueError(f"{where} must be >= 0, got {value!r}")
    return value


def _validate_mode(mode: str) -> str:
    if mode not in WEIGHT_DECAY_MODES:
        raise ValueError(
            f"weight_decay_mode must be one of {WEIGHT_DECAY_MODES}, got {mode!r}")
    return mode


def _project_rows_to_ball(w: torch.Tensor, k: float) -> torch.Tensor:
    """Backward-compatible alias for :func:`manifold.project_to_ball`."""
    return project_to_ball(w, k) if k > 0.0 else w


class DiskSparseRiemannianAdam(Optimizer):
    """Out-of-core sparse Riemannian Adam.

    Parameters
    ----------
    params:
        ``{name: tensor}`` or a list of tensors.  Tensors are used only to
        initialise the on-disk state; by default the optimizer does not keep
        references to them (``retain_params=False``), which is what makes the
        O(batch) memory claim true.  Use :meth:`read_rows` to read weights.
    k:
        Curvature magnitude: ``k > 0`` is the Poincare ball of curvature
        ``-k``, ``k = 0`` is Euclidean Adam.  A state directory stores its
        curvature; on ``resume=True`` the stored value wins over the
        constructor argument (a warning is logged if they differ).
    resume:
        ``None`` (default) refuses to touch an existing state directory and
        asks for an explicit choice; ``True`` continues it; ``False``
        reinitialises it from ``params`` (with a warning).  A state directory
        created by 0.1.0/0.2.0 is migrated on ``resume=True``: weights are
        preserved, moments and per-row counters are reset because the old
        math is not compatible with tangent-space moments.
    weight_decay_mode:
        ``"l2"`` (default) adds ``weight_decay * w`` to the Euclidean
        gradient before the Riemannian conversion, i.e. it is the
        Riemannian gradient of ``f + wd/2 * ||w||^2`` (classic Adam L2
        semantics).  ``"decoupled"`` adds ``wd * w`` to the descent
        direction in the tangent space, reproducing AdamW exactly when
        ``k = 0``.
    retain_params:
        If ``False`` (default), source tensors are replaced by zero-length
        CPU placeholders after initialisation, so the optimizer cannot keep
        a full embedding table alive in VRAM.  Set to ``True`` only for
        compatibility with code that inspects ``param_groups[...]['params']``.
    """

    def __init__(self, params: Union[list, dict], lr: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
                 weight_decay: float = 0.0, k: float = 1.0,
                 disk_dir: str = "./disk_cache", max_queue_size: int = 150,
                 resume: Optional[bool] = None, put_timeout: float = 5.0,
                 weight_decay_mode: str = "l2",
                 retain_params: bool = False,
                 init_max_norm_fraction: float = 0.5):

        if isinstance(params, dict):
            self.param_names = list(params.keys())
            params_list = list(params.values())
        else:
            params_list = list(params)
            self.param_names = [f"param_{i}" for i in range(len(params_list))]

        if not params_list:
            raise ValueError("params must not be empty")
        if len(set(self.param_names)) != len(self.param_names):
            raise ValueError("parameter names must be unique")
        for name in self.param_names:
            if not isinstance(name, str) or not name:
                raise ValueError("parameter names must be non-empty strings")
            if os.sep in name or (os.altsep is not None and os.altsep in name):
                raise ValueError(
                    f"parameter name {name!r} must not contain path separators")
        for name, param in zip(self.param_names, params_list):
            if not isinstance(param, torch.Tensor):
                raise TypeError(f"parameter '{name}' is not a torch.Tensor")
            if param.dim() != 2:
                raise ValueError(
                    f"parameter '{name}' must be 2-D (rows, dim), got shape "
                    f"{tuple(param.shape)}")
            if not param.is_floating_point():
                raise TypeError(f"parameter '{name}' must be floating point")

        if lr < 0:
            raise ValueError(f"lr must be >= 0, got {lr!r}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"beta1 must be in [0, 1), got {beta1!r}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"beta2 must be in [0, 1), got {beta2!r}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps!r}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay!r}")
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be > 0")
        if put_timeout < 0:
            raise ValueError("put_timeout must be >= 0")
        if not 0.0 < init_max_norm_fraction <= 1.0:
            raise ValueError(
                "init_max_norm_fraction must be in (0, 1], got "
                f"{init_max_norm_fraction!r}")
        self.init_max_norm_fraction = float(init_max_norm_fraction)

        requested_k = _validate_k(k)
        mode = _validate_mode(weight_decay_mode)

        self.disk_dir = os.path.abspath(os.fspath(disk_dir))
        os.makedirs(self.disk_dir, exist_ok=True)

        self._paths: Dict[str, Dict[str, str]] = {}
        meta_paths: Dict[str, str] = {}
        for name, param in zip(self.param_names, params_list):
            shape = tuple(int(s) for s in param.shape)
            self._paths[name] = {
                s: os.path.join(self.disk_dir, f"{name}_{s}.dat")
                for s in ("w", "m", "v", "t")
            }
            meta_paths[name] = os.path.join(self.disk_dir, f"{name}_meta.json")

        global_meta_path = os.path.join(self.disk_dir, GLOBAL_META_NAME)
        state_present = os.path.exists(global_meta_path) or any(
            os.path.exists(p) for paths in self._paths.values()
            for p in paths.values()) or any(
            os.path.exists(p) for p in meta_paths.values())

        if resume is None:
            if state_present:
                raise RuntimeError(
                    f"в {self.disk_dir} уже есть состояние DSA. Явно выберите "
                    f"режим: resume=True — продолжить, resume=False — "
                    f"переинициализировать из params (старое состояние будет "
                    f"перезаписано).")
            resume = False
        if not isinstance(resume, bool):
            raise TypeError("resume must be True, False or None")
        if resume and not state_present:
            raise RuntimeError(
                f"resume=True, но в {self.disk_dir} нет состояния DSA: "
                f"проверьте disk_dir или создайте состояние с resume=False.")

        stored_k = self._peek_stored_k(meta_paths) if resume else None
        effective_k = stored_k if stored_k is not None else requested_k
        if stored_k is not None and abs(stored_k - requested_k) > 0.0:
            logger.warning(
                "DSA: в состоянии сохранена кривизна k=%s; аргумент "
                "конструктора k=%s проигнорирован при resume=True. Для смены "
                "кривизны вызовите set_curvature() явно.",
                stored_k, requested_k)

        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps,
                        weight_decay=weight_decay, k=effective_k,
                        weight_decay_mode=mode)
        super().__init__(params_list, defaults)

        self._param_shapes: Dict[str, Tuple[int, int]] = {
            name: tuple(int(s) for s in param.shape)
            for name, param in zip(self.param_names, params_list)
        }
        self._placeholders: List[torch.Tensor] = []
        self._nan_rows_replaced = 0
        self._unknown_key_warnings = 0
        self._init_rows_projected = 0
        self._resume_rows_projected = 0
        self._global_step = 0

        self.state_files: Dict[str, Dict[str, np.memmap]] = {}

        for name, param in zip(self.param_names, params_list):
            shape = self._param_shapes[name]
            self._init_param_storage(
                name, shape, self._paths[name], meta_paths[name], param,
                resume=resume)

        self.state_files = self._open_memmaps()

        if resume:
            global_meta = self._read_global_meta_optional()
            if global_meta is not None:
                self._global_step = int(global_meta.get("global_step", 0))
            self._validate_ball_on_open()

        # C3: drop references to the (potentially huge) source tensors.
        if not retain_params:
            self._release_source_params()

        self.writer = AsyncDiskWriter(
            memmaps={f"{n}_{s}": self.state_files[n][s]
                     for n in self.param_names for s in ("w", "m", "v", "t")},
            max_queue_size=max_queue_size,
            put_timeout=put_timeout,
        )
        self._write_global_meta()

    # ───────────────────────── metadata helpers ─────────────────────────

    def _current_k(self) -> float:
        return float(self.param_groups[0]["k"])

    def _meta_path(self, name: str) -> str:
        return os.path.join(self.disk_dir, f"{name}_meta.json")

    def _legacy_tag(self, name: str):
        meta = self._read_meta_optional(self._meta_path(name))
        return None if meta is None else meta.get("legacy_migrated_from")

    def _read_meta_optional(self, meta_path: str) -> Optional[dict]:
        if not os.path.exists(meta_path):
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"meta-файл {meta_path} не читается: {exc!r}") from exc

    def _peek_stored_k(self, meta_paths: Dict[str, str]) -> Optional[float]:
        global_meta = self._read_global_meta_optional()
        if global_meta is not None and "k" in global_meta:
            return _validate_k(global_meta["k"], where="сохранённый k")
        for name in self.param_names:
            meta = self._read_meta_optional(meta_paths[name])
            if meta is not None and "k" in meta:
                return _validate_k(meta["k"], where=f"сохранённый k ({name})")
        return None

    def _read_global_meta_optional(self) -> Optional[dict]:
        path = os.path.join(self.disk_dir, GLOBAL_META_NAME)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"глобальный meta-файл {path} не читается: {exc!r}") from exc
        version = int(meta.get("format_version", -1))
        if version != STATE_FORMAT_VERSION:
            raise RuntimeError(
                f"глобальный meta-файл {path}: версия формата {version!r} "
                f"не поддерживается (ожидается {STATE_FORMAT_VERSION})")
        if "global_step" in meta:
            try:
                step = int(meta["global_step"])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"глобальный meta-файл {path}: global_step="
                    f"{meta['global_step']!r} не является целым") from exc
            if step < 0:
                raise RuntimeError(
                    f"глобальный meta-файл {path}: global_step={step} < 0")
        return meta

    def _write_global_meta(self) -> str:
        path = os.path.join(self.disk_dir, GLOBAL_META_NAME)
        meta = {
            "format_version": STATE_FORMAT_VERSION,
            "algorithm": ALGORITHM,
            "global_step": int(self._global_step),
            "k": float(self._current_k()),
            "weight_decay_mode": self.param_groups[0]["weight_decay_mode"],
            "param_names": list(self.param_names),
            "updated": _utc_now(),
        }
        self._atomic_json_write(path, meta)
        return path

    @staticmethod
    def _atomic_json_write(path: str, payload: dict) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def _write_meta(self, meta_path: str, name: str, shape,
                    paths: Dict[str, str], *,
                    reinit_in_progress: bool = False,
                    legacy_from: Optional[Union[int, str]] = None,
                    ball_validated: bool = True) -> None:
        meta = {
            "format_version": STATE_FORMAT_VERSION,
            "algorithm": ALGORITHM,
            "param_name": name,
            "shape": list(shape),
            "dtype": "float32",
            "counter_dtype": "int32",
            "k": float(self._current_k()),
            "weight_decay_mode": self.param_groups[0]["weight_decay_mode"],
            "files": {s: os.path.basename(p) for s, p in paths.items()},
            "ball_validated": bool(ball_validated),
            "updated": _utc_now(),
        }
        if reinit_in_progress:
            meta["reinit_in_progress"] = True
        if legacy_from is not None:
            meta["legacy_migrated_from"] = legacy_from
        self._atomic_json_write(meta_path, meta)

    def _check_meta_compat(self, meta: dict, name: str, shape, meta_path: str,
                           paths: Dict[str, str]) -> int:
        version = int(meta.get("format_version", -1))
        if version != STATE_FORMAT_VERSION and version not in LEGACY_STATE_FORMAT_VERSIONS:
            raise RuntimeError(
                f"meta-файл {meta_path}: версия формата {version!r} не "
                f"поддерживается (поддерживаются "
                f"{list(LEGACY_STATE_FORMAT_VERSIONS) + [STATE_FORMAT_VERSION]})")
        disk_shape = tuple(meta.get("shape", []))
        if disk_shape != tuple(shape):
            raise RuntimeError(
                f"несовпадение формы состояния DSA для '{name}': на диске "
                f"{list(disk_shape)}, в параметрах {list(shape)}. Удалите "
                f"{self.disk_dir} или укажите новый disk_dir.")
        if meta.get("dtype") != "float32" or meta.get("counter_dtype") != "int32":
            raise RuntimeError(
                f"meta-файл {meta_path}: неожиданные типы данных "
                f"(w/m/v={meta.get('dtype')!r}, t={meta.get('counter_dtype')!r}), "
                f"ожидались float32/int32.")
        algorithm = meta.get("algorithm")
        if algorithm is not None and algorithm != ALGORITHM:
            raise RuntimeError(
                f"meta-файл {meta_path}: алгоритм {algorithm!r} не совпадает "
                f"с {ALGORITHM!r}; состояние создано другой версией DSA.")
        if "k" in meta:
            disk_k = _validate_k(meta["k"], where=f"сохранённый k ({name})")
            if abs(disk_k - self._current_k()) > 0.0:
                raise RuntimeError(
                    f"meta-файл {meta_path}: кривизна k={disk_k} не совпадает "
                    f"с k={self._current_k()}; смешивать геометрии нельзя.")
        files = meta.get("files") or {}
        for suffix, path in paths.items():
            expected = os.path.basename(path)
            if suffix in files and files[suffix] != expected:
                raise RuntimeError(
                    f"meta-файл {meta_path}: файл '{suffix}' называется "
                    f"{files[suffix]!r}, ожидался {expected!r}.")
        return version

    # ───────────────────────── state initialisation ─────────────────────────

    def _verify_state_file_sizes(self, name, shape, paths, *, include_t: bool = True):
        """Check real file sizes before memmap opens them (r+ zero-extends)."""
        row_bytes = int(np.prod(shape)) * 4
        expected = {"w": row_bytes, "m": row_bytes, "v": row_bytes}
        if include_t:
            expected["t"] = int(shape[0]) * 4
        for suffix, exp in expected.items():
            if not os.path.exists(paths[suffix]):
                raise RuntimeError(
                    f"частичное состояние DSA для '{name}': отсутствует файл "
                    f"{paths[suffix]}")
            actual = os.path.getsize(paths[suffix])
            if actual != exp:
                raise RuntimeError(
                    f"файл состояния DSA {paths[suffix]} повреждён: размер "
                    f"{actual} байт, по форме {list(shape)} ожидается {exp} "
                    f"(состояние обрезано — сбой диска/копирования — либо файл "
                    f"чужой). resume=True невозможен: восстановите файл из "
                    f"резервной копии/чекпоинта или переоткройтесь с "
                    f"resume=False (полная переинициализация).")

    def _project_param_to_ball(self, name: str, arr: np.ndarray) -> np.ndarray:
        k = self._current_k()
        if k <= 0.0:
            return arr
        if arr.shape[0] == 0:
            return arr
        radius = math.sqrt((1.0 - BALL_EPS) / k)
        max_norm = float(np.linalg.norm(arr.astype(np.float64), axis=1).max())
        if max_norm <= 0.0:
            return arr
        limit = self.init_max_norm_fraction * radius
        if max_norm > limit and self.init_max_norm_fraction < 1.0:
            scale = limit / max_norm
            self._init_rows_projected += arr.shape[0]
            logger.warning(
                "DSA '%s': начальные веса выходят за %.0f%% радиуса шара "
                "(max |w| = %.6g, радиус %.6g); весь тензор умножен на %.6g, "
                "чтобы обучение не началось на границе (λ² ≈ 0). Передайте "
                "init_max_norm_fraction=1.0, если нужна только проекция.",
                name, 100.0 * self.init_max_norm_fraction, max_norm, radius,
                scale)
            arr = (arr.astype(np.float64) * scale).astype(arr.dtype, copy=False)
        # Even with fraction=1.0 a row may be outside; project individually.
        norm = np.linalg.norm(arr.astype(np.float64), axis=1)
        bad = norm > radius
        if bad.any():
            self._init_rows_projected += int(np.count_nonzero(bad))
            logger.warning(
                "DSA '%s': %d строк начальных весов лежат вне шара "
                "(радиус %.6g) и спроецированы.", name, int(bad.sum()), radius)
            arr = arr.copy()
            arr[bad] = project_to_ball_np(arr[bad], k)
        return arr

    def _write_fresh_state(self, name, shape, paths, param):
        arr = param.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
        if not np.isfinite(arr).all():
            bad_rows = np.nonzero(~np.isfinite(arr).all(axis=1))[0][:10].tolist()
            raise ValueError(
                f"начальные веса '{name}' содержат NaN/inf (строки {bad_rows}); "
                f"исправьте инициализацию.")
        arr = self._project_param_to_ball(name, arr)

        for suffix in ("w", "m", "v"):
            mm = np.memmap(paths[suffix], dtype="float32", mode="w+", shape=shape)
            if suffix == "w":
                mm[:] = arr
            else:
                mm[:] = 0.0
            mm.flush()
            del mm
        t = np.memmap(paths["t"], dtype="int32", mode="w+", shape=(shape[0],))
        t[:] = 0
        t.flush()
        del t

    def _reset_moments(self, name, shape, paths):
        for suffix in ("m", "v"):
            mm = np.memmap(paths[suffix], dtype="float32", mode="r+", shape=shape)
            mm[:] = 0.0
            mm.flush()
            del mm
        t = np.memmap(paths["t"], dtype="int32", mode="r+", shape=(shape[0],))
        t[:] = 0
        t.flush()
        del t

    def _init_param_storage(self, name, shape, paths, meta_path, param,
                            *, resume: bool) -> None:
        exists = {s: os.path.exists(p) for s, p in paths.items()}
        meta = self._read_meta_optional(meta_path)
        any_data = any(exists.values())

        if not resume:
            # resume=False is the documented reinitialisation path: it must
            # work even for a partial/legacy/truncated directory (C1), but a
            # trustworthy shape mismatch is still an explicit error.  Curvature
            # and algorithm may differ: the whole state is being replaced.
            if any_data or meta is not None:
                logger.warning(
                    "DSA '%s': в %s найдено состояние; resume=False → полная "
                    "переинициализация: w берётся из params, m=v=0, счётчики "
                    "шагов сброшены в 0.", name, self.disk_dir)
            if meta is not None:
                disk_shape = tuple(meta.get("shape", []))
                if disk_shape != tuple(shape):
                    raise RuntimeError(
                        f"несовпадение формы состояния DSA для '{name}': на "
                        f"диске {list(disk_shape)}, в параметрах {list(shape)}. "
                        f"Удалите {self.disk_dir} или укажите новый disk_dir.")
            if any_data or meta is not None:
                # R2: mark the transition BEFORE touching the data files.
                # _write_fresh_state rewrites w/m/v/t one by one; a crash in
                # the middle would leave a mixed state (some files new, some
                # old) that the next resume=True cannot detect by size checks
                # alone, because every file keeps its correct size.  With the
                # marker, resume=True refuses and resume=False is the
                # documented recovery (same rule as import_checkpoint).
                self._write_meta(meta_path, name, shape, paths,
                                 reinit_in_progress=True)
            self._write_fresh_state(name, shape, paths, param)
            self._write_meta(meta_path, name, shape, paths)
            return

        # resume=True
        if not any_data and meta is None:
            raise RuntimeError(
                f"resume=True, но состояния для '{name}' в {self.disk_dir} нет")
        if meta is None:
            # 0.1.0: only w/m/v, no meta, no per-row counters.
            if exists["w"] and exists["m"] and exists["v"] and not exists["t"]:
                self._verify_state_file_sizes(name, shape, paths, include_t=False)
                logger.warning(
                    "DSA '%s': найдено состояние 0.1.0 (без meta и без "
                    "счётчиков t). Веса сохранены, моменты m/v и счётчики "
                    "сброшены: старая математика несовместима с моментами в "
                    "касательном пространстве.", name)
                # Create the missing counter file first, then zero m/v/t.
                t = np.memmap(paths["t"], dtype="int32", mode="w+",
                              shape=(shape[0],))
                t[:] = 0
                t.flush()
                del t
                self._reset_moments(name, shape, paths)
                self._write_meta(meta_path, name, shape, paths,
                                 legacy_from="0.1.0", ball_validated=False)
                return
            missing = [s for s, e in exists.items() if not e]
            raise RuntimeError(
                f"для параметра '{name}' в {self.disk_dir} есть файлы "
                f"состояния, но нет meta-файла {os.path.basename(meta_path)}: "
                f"происхождение состояния неизвестно (отсутствуют {missing}). "
                f"Удалите каталог или восстановите meta.")

        version = self._check_meta_compat(meta, name, shape, meta_path, paths)
        if meta.get("reinit_in_progress"):
            raise RuntimeError(
                f"meta-файл {meta_path}: состояние помечено "
                f"'reinit_in_progress' — предыдущая переинициализация не "
                f"завершилась (сбой процесса между записью файлов и meta), "
                f"файлы могут быть смешанными. resume=True невозможен: "
                f"удалите {self.disk_dir} целиком или переоткройте с "
                f"resume=False — полная переинициализация пересоздаст файлы "
                f"и снимет маркер.")

        self._verify_state_file_sizes(name, shape, paths)
        if version == 1:
            logger.warning(
                "DSA '%s': состояние 0.2.0 (format_version=1). Веса сохранены, "
                "моменты m/v и счётчики t сброшены: 0.3.0 хранит моменты в "
                "касательном пространстве и переносит их параллельно, старые "
                "значения к новой математике неприменимы.", name)
            self._reset_moments(name, shape, paths)
            self._write_meta(meta_path, name, shape, paths,
                             legacy_from=1, ball_validated=False)
        else:
            logger.info(
                "DSA '%s': продолжаем состояние %s из %s (resume=True)",
                name, list(shape), self.disk_dir)

    def _open_memmaps(self) -> Dict[str, Dict[str, np.memmap]]:
        files: Dict[str, Dict[str, np.memmap]] = {}
        for name in self.param_names:
            shape = self._param_shapes[name]
            files[name] = {
                "w": np.memmap(self._paths[name]["w"], dtype="float32",
                               mode="r+", shape=shape),
                "m": np.memmap(self._paths[name]["m"], dtype="float32",
                               mode="r+", shape=shape),
                "v": np.memmap(self._paths[name]["v"], dtype="float32",
                               mode="r+", shape=shape),
                "t": np.memmap(self._paths[name]["t"], dtype="int32",
                               mode="r+", shape=(shape[0],)),
            }
        return files

    @staticmethod
    def _close_memmaps(files: Dict[str, Dict[str, np.memmap]]) -> None:
        for per_param in files.values():
            for mm in per_param.values():
                try:
                    mm.flush()
                except Exception:  # pragma: no cover - best effort on broken mmap
                    pass
                try:
                    mm._mmap.close()
                except Exception:  # pragma: no cover
                    pass

    def _release_source_params(self) -> None:
        """Replace source tensors with zero-length CPU placeholders (C3)."""
        placeholders = [torch.empty(0, dtype=torch.float32) for _ in self.param_names]
        self.param_groups[0]["params"] = placeholders
        self._placeholders = placeholders
        # No optimizer state is keyed by the source tensors (DSA keeps it on
        # disk), but clear the mapping defensively.
        self.state.clear()

    def _validate_ball_on_open(self) -> None:
        """On resume, project finite rows that fall outside the stored ball."""
        k = self._current_k()
        if k <= 0.0:
            return
        radius = math.sqrt((1.0 - BALL_EPS) / k)
        projected = 0
        nonfinite = 0
        for name in self.param_names:
            mm = self.state_files[name]["w"]
            n = mm.shape[0]
            for start in range(0, n, CHUNK_ROWS):
                stop = min(start + CHUNK_ROWS, n)
                w = np.array(mm[start:stop], dtype=np.float32, copy=True)
                finite = np.isfinite(w).all(axis=1)
                nonfinite += int(np.count_nonzero(~finite))
                if finite.any():
                    norm = np.linalg.norm(w[finite].astype(np.float64), axis=1)
                    bad = np.nonzero(norm > radius)[0]
                    if bad.size:
                        rows = np.nonzero(finite)[0][bad]
                        w[rows] = project_to_ball_np(w[rows], k)
                        mm[start:stop] = w
                        projected += int(rows.size)
            mm.flush()
            self._write_meta(self._meta_path(name), name,
                             self._param_shapes[name], self._paths[name],
                             legacy_from=self._legacy_tag(name),
                             ball_validated=True)
        self._resume_rows_projected = projected
        if projected:
            logger.warning(
                "DSA: при возобновлении %d строк весов лежали вне шара "
                "k=%s и были спроецированы.", projected, k)
        if nonfinite:
            logger.warning(
                "DSA: при возобновлении найдено %d не-конечных строк; они "
                "будут полностью сброшены при первом обновлении.", nonfinite)

    # ───────────────────────── indices / gradients ─────────────────────────

    def _validate_indices(self, indices: Any, n: int, name: str) -> torch.Tensor:
        if isinstance(indices, torch.Tensor):
            if indices.dim() != 1:
                raise ValueError(
                    f"indices для '{name}' должны быть 1-D, получено "
                    f"{tuple(indices.shape)}")
            if indices.dtype not in (torch.int64, torch.int32, torch.int16):
                raise TypeError(
                    f"indices для '{name}' должны быть целочисленными, "
                    f"получен dtype {indices.dtype}")
            idx = indices.detach().to(device="cpu", dtype=torch.int64)
        else:
            arr = np.asarray(indices)
            if arr.ndim != 1:
                raise ValueError(
                    f"indices для '{name}' должны быть 1-D, получено "
                    f"{arr.shape}")
            if not np.issubdtype(arr.dtype, np.integer):
                raise TypeError(
                    f"indices для '{name}' должны быть целочисленными, "
                    f"получен dtype {arr.dtype}")
            idx = torch.from_numpy(arr.astype(np.int64, copy=False))
        if idx.numel():
            if int(idx.min()) < 0:
                raise IndexError(
                    f"indices для '{name}' содержат отрицательный индекс "
                    f"{int(idx.min())}: отрицательные индексы не поддерживаются")
            if int(idx.max()) >= n:
                raise IndexError(
                    f"indices для '{name}' выходят за границы: максимум "
                    f"{int(idx.max())}, строк {n}")
        return idx

    def _validate_grads(self, grads: torch.Tensor, count: int, dim: int,
                        name: str) -> torch.Tensor:
        if not isinstance(grads, torch.Tensor):
            grads = torch.as_tensor(grads)
        if grads.is_sparse:
            raise TypeError(
                f"градиент '{name}' разреженный (sparse); DSA принимает "
                f"плотные градиенты батча")
        if tuple(grads.shape) != (count, dim):
            raise ValueError(
                f"градиент '{name}' имеет форму {tuple(grads.shape)}, "
                f"ожидалась ({count}, {dim})")
        if not grads.is_floating_point():
            raise TypeError(
                f"градиент '{name}' должен быть вещественным, получен "
                f"dtype {grads.dtype}")
        g = grads.detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(g).all():
            bad = torch.nonzero(~torch.isfinite(g).all(dim=-1)).flatten()[:10]
            raise ValueError(
                f"градиент '{name}' содержит NaN/inf (позиции {bad.tolist()}); "
                f"обучение остановлено, чтобы не испортить моменты.")
        return g

    # ─────────────────────────────── step ───────────────────────────────

    @torch.no_grad()
    def step(self, updates: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
             current_k: Optional[Union[float, torch.Tensor]] = None):
        """Apply one optimizer step.

        ``updates`` maps a parameter name to ``(indices, grads)``.  Indices
        must be 1-D integer tensors/arrays in ``[0, N)``; gradients must have
        shape ``(len(indices), dim)`` and be finite.  Duplicate indices are
        summed.  ``current_k`` changes the curvature for this and subsequent
        steps (all rows are reprojected, because the ball itself changes).
        """
        if current_k is not None:
            k_new = _validate_k(current_k, where="current_k")
            if k_new != self._current_k():
                self.set_curvature(k_new)

        if not isinstance(updates, dict):
            raise TypeError("updates must be a dict {name: (indices, grads)}")

        self._global_step += 1

        group = self.param_groups[0]
        lr = float(group["lr"])
        beta1 = float(group["beta1"])
        beta2 = float(group["beta2"])
        eps = float(group["eps"])
        weight_decay = float(group["weight_decay"])
        mode = group["weight_decay_mode"]
        k = self._current_k()

        for name, pair in updates.items():
            if name not in self.state_files:
                msg = (f"step(updates): неизвестный ключ '{name}' — обновление "
                       f"пропущено (известные ключи: {sorted(self.state_files)}); "
                       f"вероятна опечатка в имени параметра")
                if DEBUG:
                    raise KeyError(msg)
                self._unknown_key_warnings += 1
                logger.warning(msg)
                continue
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise TypeError(
                    f"updates['{name}'] должен быть (indices, grads), "
                    f"получено {type(pair).__name__}")
            indices, grads = pair
            n, dim = self._param_shapes[name]
            idx = self._validate_indices(indices, n, name)
            if idx.numel() == 0:
                continue
            g_euc = self._validate_grads(grads, int(idx.numel()), dim, name).double()

            # 1. Sum duplicate indices (hub nodes) within the batch.
            unique, inverse = torch.unique(idx, return_inverse=True)
            if unique.numel() != idx.numel():
                agg = torch.zeros((unique.numel(), dim), dtype=g_euc.dtype)
                agg.index_add_(0, inverse, g_euc)
                idx, g_euc = unique, agg

            # 2. Sequential IO: sorted, unique indices.
            if idx.numel() > 1 and not bool(torch.all(idx[1:] > idx[:-1])):
                order = torch.argsort(idx)
                idx = idx[order]
                g_euc = g_euc[order]

            idx_np = idx.numpy()
            states = self.state_files[name]

            # 3. Never read rows the writer has not finished writing (F2).
            if self.writer.intersects_inflight(name, idx_np):
                self.writer.wait_all_flushed()

            # Batch math is done in float64.  The chart is numerically
            # ill-conditioned near the boundary; float32 intermediate values
            # lose the cancellation in the gyration formula and can overflow
            # the moments even though the exact transport is an isometry.
            # Only O(batch) rows are involved, so the cost is negligible
            # compared with disk I/O; storage remains float32.
            w = torch.from_numpy(states["w"][idx_np].copy()).double()
            m = torch.from_numpy(states["m"][idx_np].copy()).double()
            v = torch.from_numpy(states["v"][idx_np].copy()).double()
            t_prev = torch.from_numpy(states["t"][idx_np].copy().astype("int64"))

            # 4. A non-finite row cannot be repaired by projection: reset the
            # whole row (weights, moments, counter), so this step is a fresh
            # Adam step (A4/R1).  A negative counter is corruption too: bias
            # correction 1 - beta**t turns negative, sqrt(v/bc2) becomes NaN
            # and the step would die with a misleading "non-finite weights".
            finite = (torch.isfinite(w).all(dim=-1)
                      & torch.isfinite(m).all(dim=-1)
                      & torch.isfinite(v).all(dim=-1)
                      & (v >= 0).all(dim=-1)
                      & (t_prev >= 0))
            bad = ~finite
            if bool(bad.any()):
                count = int(bad.sum())
                self._nan_rows_replaced += count
                logger.warning(
                    "DSA '%s': %d строк с NaN/inf, отрицательной дисперсией "
                    "или битым счётчиком в файлах состояния (строки диска "
                    "%s); строка полностью сброшена (w=m=v=0, t=0).", name,
                    count, idx[bad][:10].tolist())
                w[bad] = 0.0
                m[bad] = 0.0
                v[bad] = 0.0
                t_prev[bad] = 0

            # A zero/negative counter means the row has never had an Adam
            # step; any moments on disk are stale (e.g. a lost counter) and
            # must not leak into the fresh bias correction.
            fresh = t_prev <= 0
            if bool(fresh.any()):
                m[fresh] = 0.0
                v[fresh] = 0.0

            # 5. Defensive projection of read weights (corrupted file, F7/H1).
            if k > 0.0:
                w = project_to_ball(w, k)

            # 6. Riemannian gradient.  "l2" is the Riemannian gradient of
            # f + wd/2 * ||w||^2; in the Euclidean case it is classic Adam L2.
            if weight_decay > 0.0 and mode == "l2":
                g_euc = g_euc + weight_decay * w
            g_riem = riemannian_gradient(g_euc, w, k)

            # 7. Adam moments in the tangent space at w.
            t_new = t_prev + 1
            bc1 = (1.0 - beta1 ** t_new.double()).unsqueeze(-1)
            bc2 = (1.0 - beta2 ** t_new.double()).unsqueeze(-1)

            m.mul_(beta1).add_(g_riem, alpha=1.0 - beta1)
            v.mul_(beta2).addcmul_(g_riem, g_riem, value=1.0 - beta2)
            direction = (m / bc1) / (torch.sqrt(v / bc2) + eps)

            # 8. Decoupled decay is a tangent-space term, so in the Euclidean
            # case it is exactly AdamW and in the hyperbolic case it is part
            # of the same geodesic step.
            if weight_decay > 0.0 and mode == "decoupled":
                direction = direction + weight_decay * w

            # 9. Exponential map + parallel transport of the moments.  The
            # first moment is a tangent vector and is transported directly.
            # The second moment is a diagonal (per-coordinate) accumulator;
            # transporting it directly can rotate it into negative components,
            # so its non-negative square root is transported and squared back
            # (same scale, no NaNs).
            if k > 0.0:
                w_new = expmap(w, -lr * direction, k)
                w_new = project_to_ball(w_new, k)
                m_new = parallel_transport(w, w_new, m, k)
                sqrt_v_new = parallel_transport(w, w_new, torch.sqrt(v), k)
                v_new = sqrt_v_new.square()
            else:
                w_new = w - lr * direction
                m_new, v_new = m, v

            if not torch.isfinite(w_new).all():
                raise RuntimeError(
                    f"DSA '{name}': шаг дал не-конечные веса (k={k}, lr={lr}); "
                    f"уменьшите lr или проверьте градиенты.")

            # 10. Queue the consistent row update (storage is float32).
            w_out = w_new.float()
            m_out = m_new.float()
            v_out = v_new.float()
            if not (torch.isfinite(w_out).all() and torch.isfinite(m_out).all()
                    and torch.isfinite(v_out).all()):
                raise RuntimeError(
                    f"DSA '{name}': состояние момента не представимо во "
                    f"float32 (k={k}); точка слишком близка к границе шара. "
                    f"Уменьшите lr или увеличьте зазор BALL_EPS.")
            self.writer.put_update(name, idx, w_out, m_out, v_out,
                                   torch.clamp(t_new, max=2**31 - 1).to(torch.int32))

    # ───────────────────────── curvature / rows ─────────────────────────

    def set_curvature(self, k: float, *, reset_moments: bool = False) -> None:
        """Change curvature and restore the ball invariant for all rows.

        Changing curvature changes the manifold, so the whole table is
        scanned and out-of-ball rows are projected.  Moments live in tangent
        spaces of the old geometry; by default they are kept (a documented
        approximation for annealing).  Pass ``reset_moments=True`` for a
        mathematically clean restart of the adaptive state.
        """
        k = _validate_k(k)
        self.writer.wait_all_flushed()
        old_k = self._current_k()
        self.param_groups[0]["k"] = k
        self.defaults["k"] = k

        if reset_moments:
            with self.writer._lock:
                for name in self.param_names:
                    self._reset_moments(name, self._param_shapes[name],
                                        self._paths[name])
            logger.warning(
                "DSA: set_curvature(%s -> %s, reset_moments=True): моменты и "
                "счётчики сброшены.", old_k, k)
        else:
            logger.info("DSA: кривизна изменена %s -> %s", old_k, k)

        self.reproject_all()
        self._write_global_meta()

    def reproject_all(self, name: Optional[str] = None) -> int:
        """Project every finite row into the current ball (O(N), honest).

        Non-finite rows are fully reset (``w=m=v=0, t=0``).  Returns the
        number of projected rows.
        """
        self.writer.wait_all_flushed()
        k = self._current_k()
        names = [name] if name is not None else list(self.param_names)
        if name is not None and name not in self.state_files:
            raise KeyError(f"неизвестный параметр '{name}'")
        projected = 0
        with self.writer._lock:
            for pname in names:
                shape = self._param_shapes[pname]
                states = self.state_files[pname]
                for start in range(0, shape[0], CHUNK_ROWS):
                    stop = min(start + CHUNK_ROWS, shape[0])
                    w = np.array(states["w"][start:stop], copy=True)
                    m = np.array(states["m"][start:stop], copy=True)
                    v = np.array(states["v"][start:stop], copy=True)
                    t = np.array(states["t"][start:stop], copy=True)
                    finite = np.isfinite(w).all(axis=1) & \
                        np.isfinite(m).all(axis=1) & \
                        np.isfinite(v).all(axis=1) & (v >= 0).all(axis=1)
                    if (~finite).any():
                        w[~finite] = 0.0
                        m[~finite] = 0.0
                        v[~finite] = 0.0
                        t[~finite] = 0
                    if k > 0.0 and finite.any():
                        norm = np.linalg.norm(w[finite].astype(np.float64), axis=1)
                        radius = math.sqrt((1.0 - BALL_EPS) / k)
                        bad = norm > radius
                        if bad.any():
                            rows = np.nonzero(finite)[0][bad]
                            w[rows] = project_to_ball_np(w[rows], k)
                            projected += int(rows.size)
                    states["w"][start:stop] = w
                    states["m"][start:stop] = m
                    states["v"][start:stop] = v
                    states["t"][start:stop] = t
                for mm in states.values():
                    mm.flush()
                self._write_meta(self._meta_path(pname), pname, shape,
                                 self._paths[pname],
                                 legacy_from=self._legacy_tag(pname),
                                 ball_validated=True)
        if projected:
            logger.warning("DSA: %d строк спроецированы в шар k=%s.",
                           projected, k)
        return projected

    def read_rows(self, indices, name: Optional[str] = None) -> torch.Tensor:
        """Read rows of ``w`` for inference (projected into the ball).

        This is the public alternative to reading ``state_files`` directly.
        """
        if name is None:
            name = self.param_names[0]
        if name not in self.state_files:
            raise KeyError(f"неизвестный параметр '{name}'")
        n, _ = self._param_shapes[name]
        idx = self._validate_indices(indices, n, name)
        if idx.numel() == 0:
            return torch.empty((0, self._param_shapes[name][1]),
                               dtype=torch.float32)
        idx_np = idx.numpy()
        if self.writer.intersects_inflight(name, idx_np):
            self.writer.wait_all_flushed()
        w = torch.from_numpy(self.state_files[name]["w"][idx_np].copy())
        if not torch.isfinite(w).all():
            raise RuntimeError(
                f"DSA '{name}': прочитанные строки содержат NaN/inf; файл "
                f"состояния повреждён (лечится обновлением этих строк).")
        k = self._current_k()
        if k > 0.0:
            w = project_to_ball(w, k)
        return w

    # ───────────────────────── checkpoints (F8/C2) ─────────────────────────

    @staticmethod
    def _copy_and_hash(src: str, dst: str) -> Tuple[int, str]:
        digest = hashlib.sha256()
        size = 0
        with open(src, "rb") as fin, open(dst, "wb") as fout:
            while True:
                chunk = fin.read(1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
                fout.write(chunk)
                size += len(chunk)
            fout.flush()
            os.fsync(fout.fileno())
        return size, digest.hexdigest()

    def export_checkpoint(self, directory: str) -> str:
        """Write a self-contained checkpoint directory (C2).

        Data files are copied into ``directory`` under unique names; the
        manifest (``checkpoint.json``) stores relative file names, sizes and
        SHA-256 checksums.  The directory can be moved to another machine and
        imported with :meth:`import_checkpoint`.
        """
        directory = os.path.abspath(os.fspath(directory))
        os.makedirs(directory, exist_ok=True)
        self.writer.wait_all_flushed()
        manifest_path = os.path.join(directory, CHECKPOINT_MANIFEST_NAME)

        old_files = set()
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    old = json.load(f)
                for info in old.get("params", {}).values():
                    for rec in info.get("files", {}).values():
                        if isinstance(rec, dict) and "name" in rec:
                            old_files.add(rec["name"])
            except (OSError, ValueError):
                old_files = set()

        stamp = (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                 + f"-{os.getpid()}-{os.urandom(3).hex()}")
        manifest = {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "algorithm": ALGORITHM,
            "saved_at": _utc_now(),
            "global_step": int(self._global_step),
            "k": float(self._current_k()),
            "weight_decay_mode": self.param_groups[0]["weight_decay_mode"],
            "param_names": list(self.param_names),
            "params": {},
        }

        new_files = set()
        with self.writer._lock:
            for name in self.param_names:
                shape = self._param_shapes[name]
                info = {"shape": list(shape), "files": {}}
                for suffix in ("w", "m", "v", "t"):
                    mm = self.state_files[name][suffix]
                    mm.flush()
                    filename = f"{name}_{suffix}_{stamp}.dat"
                    dst = os.path.join(directory, filename)
                    size, sha = self._copy_and_hash(self._paths[name][suffix], dst)
                    info["files"][suffix] = {
                        "name": filename, "size": int(size), "sha256": sha}
                    new_files.add(filename)
                manifest["params"][name] = info

        self._atomic_json_write(manifest_path, manifest)

        # Remove files of the previous export only after the new manifest is
        # durable: an interrupted export leaves the old checkpoint valid.
        for filename in sorted(old_files - new_files):
            try:
                os.remove(os.path.join(directory, filename))
            except OSError:
                pass
        return manifest_path

    def _resolve_checkpoint(self, path: str) -> Tuple[str, str]:
        path = os.path.abspath(os.fspath(path))
        if os.path.isdir(path):
            return os.path.join(path, CHECKPOINT_MANIFEST_NAME), path
        if os.path.isfile(path):
            if os.path.basename(path) == CHECKPOINT_MANIFEST_NAME:
                return path, os.path.dirname(path)
            raise RuntimeError(
                f"{path} — файл, а не каталог чекпоинта. Начиная с 0.3.0 "
                f"чекпоинт — это каталог с checkpoint.json и файлами данных; "
                f"используйте export_checkpoint(dir)/import_checkpoint(dir). "
                f"Манифесты 0.2.0 с абсолютными путями не поддерживаются.")
        raise RuntimeError(f"чекпоинт {path} не найден")

    def import_checkpoint(self, path: str, *, verify_hashes: bool = True) -> dict:
        """Import a checkpoint directory exported by :meth:`export_checkpoint`."""
        manifest_path, directory = self._resolve_checkpoint(path)
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"чекпоинт {manifest_path} не читается: {exc!r}") from exc
        if manifest.get("format") != CHECKPOINT_FORMAT:
            raise RuntimeError(
                f"{manifest_path} не является чекпоинтом DSA "
                f"(format={manifest.get('format')!r})")
        if int(manifest.get("version", -1)) != CHECKPOINT_VERSION:
            raise RuntimeError(
                f"чекпоинт {manifest_path}: версия {manifest.get('version')!r} "
                f"не поддерживается (ожидается {CHECKPOINT_VERSION}). "
                f"Чекпоинты 0.2.0 содержали абсолютные пути и не переносимы; "
                f"создайте новый чекпоинт.")
        if manifest.get("algorithm") != ALGORITHM:
            raise RuntimeError(
                f"чекпоинт {manifest_path}: алгоритм "
                f"{manifest.get('algorithm')!r}, ожидался {ALGORITHM!r}")
        manifest_params = manifest.get("params")
        if not isinstance(manifest_params, dict):
            raise RuntimeError(f"чекпоинт {manifest_path}: поле 'params' повреждено")
        saved_names = list(manifest_params.keys())
        if set(saved_names) != set(self.param_names):
            raise RuntimeError(
                f"чекпоинт содержит параметры {sorted(saved_names)}, "
                f"оптимизатор — {sorted(self.param_names)}")
        k = _validate_k(manifest.get("k"), where="k чекпоинта")

        self.writer.wait_all_flushed()
        staged: Dict[Tuple[str, str], str] = {}
        try:
            for name in self.param_names:
                info = manifest_params[name]
                if not isinstance(info, dict):
                    raise RuntimeError(
                        f"чекпоинт {manifest_path}: описание '{name}' повреждено")
                shape = self._param_shapes[name]
                saved_shape = info.get("shape")
                if not isinstance(saved_shape, (list, tuple)):
                    raise RuntimeError(
                        f"чекпоинт {manifest_path}: форма '{name}' повреждена")
                if tuple(saved_shape) != shape:
                    raise RuntimeError(
                        f"несовпадение формы для '{name}': чекпоинт "
                        f"{info.get('shape')}, оптимизатор {list(shape)}")
                info_files = info.get("files")
                if not isinstance(info_files, dict):
                    raise RuntimeError(
                        f"чекпоинт {manifest_path}: файлы '{name}' повреждены")
                for suffix in ("w", "m", "v", "t"):
                    rec = info_files.get(suffix)
                    if not isinstance(rec, dict) or "name" not in rec:
                        raise RuntimeError(
                            f"чекпоинт не содержит файл '{suffix}' для '{name}'")
                    filename = rec["name"]
                    if not isinstance(filename, str):
                        raise RuntimeError(
                            f"недопустимое имя файла в чекпоинте: {filename!r}")
                    if os.path.basename(filename) != filename:
                        raise RuntimeError(
                            f"недопустимое имя файла в чекпоинте: {filename!r}")
                    src = os.path.join(directory, filename)
                    if not os.path.exists(src):
                        raise RuntimeError(f"файл чекпоинта отсутствует: {src}")
                    actual_size = os.path.getsize(src)
                    if "size" in rec and int(rec["size"]) != actual_size:
                        raise RuntimeError(
                            f"файл {src}: размер {actual_size} байт, в манифесте "
                            f"{rec['size']} — чекпоинт повреждён")
                    expected_size = (shape[0] * shape[1] * 4 if suffix != "t"
                                     else shape[0] * 4)
                    if actual_size != expected_size:
                        raise RuntimeError(
                            f"файл {src}: размер {actual_size} байт, по форме "
                            f"{list(shape)} ожидается {expected_size}")
                    tmp = os.path.join(
                        self.disk_dir, f".import-{os.getpid()}-{name}-{suffix}")
                    size, sha = self._copy_and_hash(src, tmp)
                    if "size" in rec and size != int(rec["size"]):
                        raise RuntimeError(f"файл {src}: размер изменился при копировании")
                    if verify_hashes and "sha256" in rec and sha != rec["sha256"]:
                        raise RuntimeError(
                            f"файл {src}: SHA-256 не совпадает с манифестом — "
                            f"чекпоинт повреждён")
                    staged[(name, suffix)] = tmp

            # Mark the state transition.  If the process dies between the
            # individual os.replace calls, the next resume=True must refuse
            # the mixed state instead of accepting it (same rule as R2).
            for name in self.param_names:
                self._write_meta(self._meta_path(name), name,
                                 self._param_shapes[name], self._paths[name],
                                 reinit_in_progress=True, ball_validated=False)

            # Replace the live state only after every file has been staged and
            # verified.  Close memmaps first: on Windows an open file cannot
            # be replaced.
            try:
                with self.writer._lock:
                    self._close_memmaps(self.state_files)
                    for (name, suffix), tmp in staged.items():
                        os.replace(tmp, self._paths[name][suffix])
                    self.state_files = self._open_memmaps()
                    self.writer.memmaps = {
                        f"{n}_{s}": mm
                        for n, per_param in self.state_files.items()
                        for s, mm in per_param.items()
                    }
            except Exception:
                # Reopen whatever is on disk so the object is not left with
                # closed memmaps; the reinit marker still blocks resume=True.
                try:
                    self.state_files = self._open_memmaps()
                    self.writer.memmaps = {
                        f"{n}_{s}": mm
                        for n, per_param in self.state_files.items()
                        for s, mm in per_param.items()
                    }
                except Exception:
                    pass
                raise
        except Exception:
            for tmp in staged.values():
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            raise

        self.param_groups[0]["k"] = k
        self.defaults["k"] = k
        self._global_step = int(manifest.get("global_step", 0))
        for name in self.param_names:
            self._write_meta(self._meta_path(name), name,
                             self._param_shapes[name], self._paths[name],
                             ball_validated=False)
        self._write_global_meta()
        self._validate_ball_on_open()
        logger.info(
            "DSA: импортирован чекпоинт %s (global_step=%d, k=%s)",
            manifest_path, self._global_step, k)
        return manifest

    def save_checkpoint(self, path: str) -> str:
        """Deprecated alias: ``path`` is a checkpoint directory (C2)."""
        path = os.path.abspath(os.fspath(path))
        if os.path.isfile(path) or (path.endswith(".json") and not os.path.isdir(path)):
            raise ValueError(
                "save_checkpoint(path) в 0.3.0 принимает КАТАЛОГ, а не файл "
                "манифеста: используйте save_checkpoint('checkpoints/run1') "
                "или export_checkpoint('checkpoints/run1').")
        return self.export_checkpoint(path)

    def load_checkpoint(self, path: str, **kwargs) -> dict:
        """Deprecated alias: ``path`` is a checkpoint directory (C2)."""
        return self.import_checkpoint(path, **kwargs)

    # ───────────────────────── compatibility ─────────────────────────

    def state_dict(self):
        raise NotImplementedError(
            "DSA хранит состояние на диске (файлы <name>_w/_m/_v/_t.dat в "
            f"{self.disk_dir}); сериализовать его через state_dict() нельзя. "
            "Используйте export_checkpoint()/import_checkpoint() либо "
            "shutdown() и повторное открытие с resume=True.")

    def load_state_dict(self, state_dict):
        raise NotImplementedError(
            "DSA восстанавливается с диска: создайте оптимизатор с resume=True "
            "на том же disk_dir (и при необходимости import_checkpoint()). "
            "load_state_dict() не поддерживается.")

    def shutdown(self):
        """Drain the writer, flush the state and persist the global step.

        Retryable: if a previous call failed (e.g. a stuck disk), calling it
        again resumes the drain.  Do not use the optimizer for further steps
        after a successful shutdown.
        """
        error = None
        try:
            self.writer.shutdown()
        except BaseException as exc:  # preserve the writer error
            error = exc
        try:
            self._write_global_meta()
        except BaseException as exc:
            if error is None:
                error = exc
            else:
                logger.error("DSA: не удалось обновить глобальный meta: %r", exc)
        if error is not None:
            raise error

    def zero_grad(self, set_to_none: bool = False):
        """No-op: gradients are passed explicitly to :meth:`step`."""
        pass

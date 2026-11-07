"""Regression and acceptance tests for DSA 0.3.0.

The suite is organised by the audit finding it covers:

  C1  legacy 0.1.0 state can be reinitialised and migrated
  C2  portable checkpoint directory (no absolute paths)
  C3  source tensors are not retained (O(batch) memory contract)
  C4  explicit weight-decay semantics (L2 vs decoupled)
  C5  one packaging metadata source
  A1  shutdown is retryable and never reports a false success
  A2  global_step survives resume and checkpoint
  A3  negative/out-of-range indices are rejected
  A4  a non-finite row is fully reset (w, m, v, t)
  A9  gradient shape/finiteness is validated
  H1  initial/resumed weights are inside the ball
  H2  full Riemannian Adam (tangent moments, exp map, parallel transport)
  H3  curvature changes reproject the whole table
  H4  k is validated early (NaN/inf/negative)
  F1-F8, R1-R3  the 0.2.0 correctness guarantees
"""

import gc
import json
import logging
import math
import os
import shutil
import tempfile
import threading
import time
import weakref

import numpy as np
import pytest
import torch

from dsa import manifold
from dsa.optimizer import DiskSparseRiemannianAdam
from tests.reference import RiemannianAdamReference, expmap

BETA1, BETA2, EPS = 0.9, 0.999, 1e-8


# ───────────────────────── helpers ─────────────────────────

def make_opt(disk_dir, w0, lr=0.05, k=0.0, resume=False, **kw):
    params = torch.from_numpy(np.array(w0, dtype=np.float32))
    kw.setdefault("lr", lr)
    kw.setdefault("k", k)
    # Numeric tests need exactly the weights they passed; H1 tests exercise
    # the default initial scaling explicitly.
    kw.setdefault("init_max_norm_fraction", 1.0)
    return DiskSparseRiemannianAdam(params={"ent": params},
                                    disk_dir=disk_dir, resume=resume, **kw)


def do_step(opt, idx, g):
    idx_t = torch.tensor(np.asarray(idx, dtype=np.int64))
    g_t = torch.tensor(np.asarray(g, dtype=np.float32))
    opt.step(updates={"ent": (idx_t, g_t)})


def read_state(opt, name="ent"):
    opt.writer.wait_all_flushed()
    sf = opt.state_files[name]
    return (np.array(sf["w"]), np.array(sf["m"]), np.array(sf["v"]),
            np.array(sf["t"]))


def assert_matches_reference(opt, ref, rows=None, atol=1e-5):
    w, m, v, t = read_state(opt)
    rows = np.arange(ref.n) if rows is None else np.asarray(rows)
    assert np.array_equal(t[rows], ref.t[rows]), \
        f"пер-строчные счётчики разошлись: {t[rows]} vs {ref.t[rows]}"
    assert np.allclose(w[rows], ref.w[rows], atol=atol), "w разошёлся с эталоном"
    assert np.allclose(m[rows], ref.m[rows], atol=atol), "m разошёлся с эталоном"
    assert np.allclose(v[rows], ref.v[rows], atol=atol), "v разошёлся с эталоном"


def slow_down_writer(opt, delay):
    orig = opt.writer._apply_write

    def slow(name, indices, w_np, m_np, v_np, t_np=None):
        time.sleep(delay)
        return orig(name, indices, w_np, m_np, v_np, t_np)

    opt.writer._apply_write = slow


def gate_writer(opt):
    orig = opt.writer._apply_write
    gate = threading.Event()

    def gated(name, indices, w_np, m_np, v_np, t_np=None):
        gate.wait()
        return orig(name, indices, w_np, m_np, v_np, t_np)

    opt.writer._apply_write = gated
    return gate.set


def _write_legacy_010(directory, w, m, v):
    for suffix, arr in (("w", w), ("m", m), ("v", v)):
        mm = np.memmap(os.path.join(directory, f"ent_{suffix}.dat"),
                       dtype="float32", mode="w+", shape=arr.shape)
        mm[:] = arr.astype(np.float32)
        mm.flush()
        del mm


def _write_legacy_020(directory, w, m, v, t):
    _write_legacy_010(directory, w, m, v)
    mm = np.memmap(os.path.join(directory, "ent_t.dat"), dtype="int32",
                   mode="w+", shape=(len(t),))
    mm[:] = t
    mm.flush()
    del mm
    meta = {
        "format_version": 1,
        "param_name": "ent",
        "shape": list(w.shape),
        "dtype": "float32",
        "counter_dtype": "int32",
        "files": {s: f"ent_{s}.dat" for s in ("w", "m", "v", "t")},
    }
    with open(os.path.join(directory, "ent_meta.json"), "w") as f:
        json.dump(meta, f)


# ───────────────────────── F1: no dropped batches ─────────────────────────

def test_f1_slow_disk_loses_nothing(tmp_path):
    n, d = 300, 4
    rng = np.random.default_rng(0)
    w0 = rng.normal(0, 0.05, (n, d))
    opt = make_opt(str(tmp_path), w0, lr=0.05, k=0.0, max_queue_size=2)
    slow_down_writer(opt, 0.01)

    ref = RiemannianAdamReference(w0, lr=0.05, k=0.0)
    for s in range(12):
        rows = (s * 25 + np.arange(40)) % n
        g = rng.normal(0, 1.0, (len(rows), d))
        do_step(opt, rows, g)
        ref.step(rows, g)

    opt.shutdown()
    assert_matches_reference(opt, ref)


def test_f1_sync_fallback_on_stalled_writer(tmp_path):
    n, d = 200, 4
    rng = np.random.default_rng(1)
    w0 = rng.normal(0, 0.05, (n, d))
    opt = make_opt(str(tmp_path), w0, lr=0.05, k=0.0,
                   max_queue_size=1, put_timeout=0.15)
    open_gate = gate_writer(opt)

    ref = RiemannianAdamReference(w0, lr=0.05, k=0.0)
    threading.Timer(0.5, open_gate).start()

    for s in range(5):
        rows = np.arange(s * 20, s * 20 + 20)
        g = rng.normal(0, 1.0, (len(rows), d))
        do_step(opt, rows, g)
        ref.step(rows, g)

    opt.shutdown()
    assert opt.writer._sync_fallbacks >= 1
    assert_matches_reference(opt, ref, rows=np.arange(100))


# ───────────────────────── F2: step/writer race ─────────────────────────

def test_f2_repeated_rows_never_stale(tmp_path):
    n, d = 16, 3
    w0 = np.zeros((n, d))
    opt = make_opt(str(tmp_path), w0, lr=0.05, k=0.0)
    slow_down_writer(opt, 0.03)

    ref = RiemannianAdamReference(w0, lr=0.05, k=0.0)
    rows = [3, 7]
    for g_val in (1.0, -2.0, 0.5):
        g = np.full((len(rows), d), g_val)
        do_step(opt, rows, g)
        ref.step(rows, g)

    opt.shutdown()
    assert_matches_reference(opt, ref, rows=rows)


def test_f2_disjoint_rows_do_not_drain(tmp_path):
    n, d = 32, 3
    w0 = np.zeros((n, d))
    opt = make_opt(str(tmp_path), w0, lr=0.05, k=0.0)
    open_gate = gate_writer(opt)

    do_step(opt, [0, 1], np.ones((2, d)))
    assert opt.writer.intersects_inflight("ent", np.array([0, 1]))
    assert not opt.writer.intersects_inflight("ent", np.array([5, 6]))

    drains = {"n": 0}

    def record_only_wait(*a, **kw):
        drains["n"] += 1

    opt.writer.wait_all_flushed = record_only_wait
    do_step(opt, [5, 6], np.ones((2, d)))
    assert drains["n"] == 0
    do_step(opt, [0, 1], np.ones((2, d)))
    assert drains["n"] == 1

    open_gate()
    opt.shutdown()


# ───────────────────────── F3: state lifecycle ─────────────────────────

def test_f3_meta_written_on_fresh_init(tmp_path):
    n, d = 10, 2
    opt = make_opt(str(tmp_path), np.zeros((n, d)))
    meta_path = os.path.join(str(tmp_path), "ent_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta["format_version"] == 2
    assert meta["algorithm"] == "riemannian_adam"
    assert meta["shape"] == [n, d]
    assert meta["dtype"] == "float32" and meta["counter_dtype"] == "int32"
    assert meta["k"] == 0.0 and meta["ball_validated"] is True
    opt.shutdown()


def test_f3_resume_true_preserves_state_and_global_step(tmp_path):
    n, d = 12, 3
    rng = np.random.default_rng(2)
    w0 = rng.normal(0, 0.1, (n, d))
    opt = make_opt(str(tmp_path), w0, lr=0.05, k=0.0)
    rows = [1, 4, 9]
    for g_val in (1.0, -0.5):
        do_step(opt, rows, np.full((len(rows), d), g_val))
    opt.shutdown()

    w1, m1, v1, t1 = read_state(opt)
    assert t1[rows].tolist() == [2, 2, 2]

    other = rng.normal(0, 5.0, (n, d))
    opt2 = make_opt(str(tmp_path), other, lr=0.05, k=0.0, resume=True)
    assert opt2._global_step == 2
    w2, m2, v2, t2 = read_state(opt2)
    assert np.array_equal(w1, w2) and np.array_equal(m1, m2)
    assert np.array_equal(v1, v2) and np.array_equal(t1, t2)

    ref = RiemannianAdamReference.__new__(RiemannianAdamReference)
    ref.w, ref.m, ref.v, ref.t = (np.array(w1, dtype=np.float64),
                                  np.array(m1, dtype=np.float64),
                                  np.array(v1, dtype=np.float64),
                                  np.array(t1, dtype=np.int64))
    ref.n, ref.d = n, d
    ref.lr, ref.b1, ref.b2, ref.eps = 0.05, BETA1, BETA2, EPS
    ref.k, ref.wd, ref.wd_mode = 0.0, 0.0, "l2"
    do_step(opt2, rows, np.full((len(rows), d), 2.0))
    ref.step(rows, np.full((len(rows), d), 2.0))
    opt2.shutdown()
    assert_matches_reference(opt2, ref, rows=rows)


def test_f3_existing_state_requires_explicit_resume(tmp_path):
    n, d = 8, 3
    opt = make_opt(str(tmp_path), np.zeros((n, d)))
    opt.shutdown()
    with pytest.raises(RuntimeError, match="resume=True"):
        make_opt(str(tmp_path), np.zeros((n, d)), resume=None)


def test_f3_resume_false_reinitialises_and_warns(tmp_path, caplog):
    n, d = 12, 3
    w0 = np.full((n, d), 0.25)
    opt = make_opt(str(tmp_path), w0, lr=0.05, k=0.0)
    do_step(opt, [0, 1], np.ones((2, d)))
    opt.shutdown()

    fresh = np.random.default_rng(3).normal(0, 0.3, (n, d))
    with caplog.at_level(logging.WARNING, logger="DSA"):
        opt2 = make_opt(str(tmp_path), fresh, lr=0.05, k=0.0, resume=False)
    assert any("переинициализация" in r.message for r in caplog.records)

    w, m, v, t = read_state(opt2)
    assert np.allclose(w, fresh, atol=1e-7)
    assert np.all(m == 0) and np.all(v == 0) and np.all(t == 0)
    assert opt2._global_step == 0
    opt2.shutdown()


def test_f3_shape_mismatch_raises(tmp_path):
    opt = make_opt(str(tmp_path), np.zeros((50, 4)))
    opt.shutdown()
    with pytest.raises(RuntimeError, match="несовпадение формы"):
        make_opt(str(tmp_path), np.zeros((60, 4)), resume=True)
    with pytest.raises(RuntimeError, match="несовпадение формы"):
        make_opt(str(tmp_path), np.zeros((50, 8)), resume=False)


def test_f3_partial_state_raises(tmp_path):
    opt = make_opt(str(tmp_path), np.zeros((10, 2)))
    opt.shutdown()
    os.remove(os.path.join(str(tmp_path), "ent_v.dat"))
    with pytest.raises(RuntimeError, match="частичное состояние"):
        make_opt(str(tmp_path), np.zeros((10, 2)), resume=True)


def test_f3_missing_meta_raises(tmp_path):
    opt = make_opt(str(tmp_path), np.zeros((10, 2)))
    opt.shutdown()
    os.remove(os.path.join(str(tmp_path), "ent_meta.json"))
    with pytest.raises(RuntimeError, match="meta"):
        make_opt(str(tmp_path), np.zeros((10, 2)), resume=True)


@pytest.mark.parametrize("suffix", ["w", "m", "v", "t"])
def test_f3_truncated_data_file_raises_on_resume(tmp_path, suffix):
    n, d = 10, 2
    opt = make_opt(str(tmp_path), np.random.default_rng(5).normal(0, 0.1, (n, d)),
                   lr=0.05, k=0.0)
    do_step(opt, [0, 5], np.ones((2, d)))
    opt.shutdown()
    p = os.path.join(str(tmp_path), f"ent_{suffix}.dat")
    sz = os.path.getsize(p)
    with open(p, "r+b") as f:
        f.truncate(sz // 2)
    with pytest.raises(RuntimeError, match="повреждён: размер"):
        make_opt(str(tmp_path), np.zeros((n, d)), resume=True)
    assert os.path.getsize(p) == sz // 2


def test_f3_oversized_data_file_raises_on_resume(tmp_path):
    n, d = 10, 2
    opt = make_opt(str(tmp_path), np.zeros((n, d)))
    opt.shutdown()
    p = os.path.join(str(tmp_path), "ent_w.dat")
    with open(p, "ab") as f:
        f.write(b"\x00" * 16)
    with pytest.raises(RuntimeError, match="повреждён: размер"):
        make_opt(str(tmp_path), np.zeros((n, d)), resume=True)


def test_f3_truncated_file_resume_false_reinits(tmp_path):
    n, d = 10, 2
    opt = make_opt(str(tmp_path), np.zeros((n, d)))
    opt.shutdown()
    p = os.path.join(str(tmp_path), "ent_v.dat")
    with open(p, "r+b") as f:
        f.truncate(os.path.getsize(p) // 3)
    fresh = np.random.default_rng(7).normal(0, 0.3, (n, d))
    opt2 = make_opt(str(tmp_path), fresh, lr=0.05, k=0.0, resume=False)
    w, m, v, t = read_state(opt2)
    assert os.path.getsize(p) == n * d * 4
    assert np.allclose(w, fresh, atol=1e-7)
    assert np.all(m == 0) and np.all(v == 0) and np.all(t == 0)
    opt2.shutdown()


def test_r2_crash_mid_reinit_is_refused_on_resume(tmp_path, monkeypatch):
    """kill -9 посреди resume=False переинициализации (дописан новый w, m/v/t
    нет, meta старый валидный): размеры всех файлов совпадают, поэтому без
    маркера resume=True принимает смешанное состояние молча. Маркер обязан
    стоять ДО перезаписи файлов; resume=True отказывает, resume=False лечит."""
    from dsa import optimizer as dsa_opt

    n, d = 6, 3
    opt = make_opt(str(tmp_path), np.full((n, d), 0.2), lr=0.05, k=0.0)
    do_step(opt, [0, 1], np.ones((2, d)))
    opt.shutdown()
    meta_path = os.path.join(str(tmp_path), "ent_meta.json")

    # имитация смерти процесса после записи w, но до m/v/t
    def crash_after_w(self, name, shape, paths, param):
        mm = np.memmap(paths["w"], dtype="float32", mode="w+", shape=shape)
        mm[:] = np.full(shape, 0.9, dtype=np.float32)
        mm.flush()
        del mm
        raise RuntimeError("имитация kill -9 посреди переинициализации")

    monkeypatch.setattr(
        dsa_opt.DiskSparseRiemannianAdam, "_write_fresh_state", crash_after_w)
    with pytest.raises(RuntimeError, match="kill -9"):
        make_opt(str(tmp_path), np.full((n, d), 0.3), lr=0.05, k=0.0,
                 resume=False)
    monkeypatch.undo()

    # маркер в meta — свидетельство незавершённой переинициализации
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta.get("reinit_in_progress") is True, \
        "переинициализация не помечена: смешанное состояние примут молча"

    with pytest.raises(RuntimeError, match="reinit_in_progress"):
        make_opt(str(tmp_path), np.full((n, d), 0.3), lr=0.05, k=0.0,
                 resume=True)

    # resume=False — документированный путь восстановления, маркер снимается
    opt2 = make_opt(str(tmp_path), np.full((n, d), 0.25), lr=0.05, k=0.0,
                    resume=False)
    w, m, v, t = read_state(opt2)
    assert np.allclose(w, 0.25, atol=1e-7)
    assert np.all(m == 0) and np.all(v == 0) and np.all(t == 0)
    with open(meta_path) as f:
        assert "reinit_in_progress" not in json.load(f)
    opt2.shutdown()


# ───────────────────────── C1: legacy 0.1.0 / 0.2.0 ─────────────────────────

def test_c1_legacy_010_resume_true_migrates(tmp_path):
    n, d = 6, 3
    w = np.full((n, d), 0.05)
    _write_legacy_010(str(tmp_path), w, np.full((n, d), 0.3),
                      np.full((n, d), 0.4))

    opt = make_opt(str(tmp_path), np.zeros((n, d)), k=0.0, resume=True)
    st = read_state(opt)
    assert np.allclose(st[0], w, atol=1e-7), "веса 0.1.0 должны сохраниться"
    assert np.all(st[1] == 0) and np.all(st[2] == 0) and np.all(st[3] == 0)
    with open(os.path.join(str(tmp_path), "ent_meta.json")) as f:
        meta = json.load(f)
    assert meta["format_version"] == 2
    assert meta["legacy_migrated_from"] == "0.1.0"
    opt.shutdown()


def test_c1_legacy_010_resume_false_reinitialises(tmp_path):
    n, d = 6, 3
    _write_legacy_010(str(tmp_path), np.full((n, d), 0.05),
                      np.full((n, d), 0.3), np.full((n, d), 0.4))
    fresh = np.full((n, d), 0.02)
    opt = make_opt(str(tmp_path), fresh, k=0.0, resume=False)
    st = read_state(opt)
    assert np.allclose(st[0], fresh, atol=1e-7)
    assert np.all(st[1] == 0) and np.all(st[2] == 0) and np.all(st[3] == 0)
    opt.shutdown()


def test_c1_legacy_010_resume_none_raises(tmp_path):
    n, d = 6, 3
    _write_legacy_010(str(tmp_path), np.zeros((n, d)), np.zeros((n, d)),
                      np.zeros((n, d)))
    with pytest.raises(RuntimeError, match="resume=True"):
        make_opt(str(tmp_path), np.zeros((n, d)), resume=None)


def test_c1_legacy_020_resets_incompatible_moments(tmp_path):
    n, d = 4, 2
    w = np.full((n, d), 0.04)
    _write_legacy_020(str(tmp_path), w, np.full((n, d), 0.5),
                      np.full((n, d), 0.6), np.full(n, 7))
    opt = make_opt(str(tmp_path), np.zeros((n, d)), k=0.0, resume=True)
    st = read_state(opt)
    assert np.allclose(st[0], w, atol=1e-7)
    assert np.all(st[1] == 0) and np.all(st[2] == 0) and np.all(st[3] == 0)
    with open(os.path.join(str(tmp_path), "ent_meta.json")) as f:
        assert json.load(f)["legacy_migrated_from"] == 1
    opt.shutdown()


# ───────────────────────── F4 / C4: Adam and weight decay ─────────────────────────

def test_f4_first_step_of_fresh_row_is_lr_sign(tmp_path):
    n, d = 10, 3
    lr = 0.1
    opt = make_opt(str(tmp_path), np.zeros((n, d)), lr=lr, k=0.0)
    g = np.array([[1.0, -2.0, 0.5]])
    do_step(opt, [4], g)
    opt.shutdown()

    w, m, v, t = read_state(opt)
    expected_delta = lr * g[0] / (np.abs(g[0]) + EPS)
    assert np.allclose(w[4], -expected_delta, atol=1e-7)
    assert np.allclose(m[4], (1 - BETA1) * g[0], atol=1e-7)
    assert np.allclose(v[4], (1 - BETA2) * g[0] ** 2, atol=1e-7)
    assert t[4] == 1


def test_f4_row_counter_independent_of_global_steps(tmp_path):
    n, d = 8, 3
    lr = 0.1
    opt = make_opt(str(tmp_path), np.zeros((n, d)), lr=lr, k=0.0)
    for _ in range(5):
        do_step(opt, [0], np.ones((1, d)))
    do_step(opt, [1], np.ones((1, d)))
    opt.shutdown()

    w, _, _, t = read_state(opt)
    expected = lr / (1.0 + EPS)
    assert t[0] == 5 and t[1] == 1 and t[2:].sum() == 0
    assert np.allclose(w[1], -expected, atol=1e-7)


@pytest.mark.parametrize("mode", ["l2", "decoupled"])
def test_c4_euclidean_weight_decay_matches_reference(tmp_path, mode):
    n, d = 40, 5
    rng = np.random.default_rng(7)
    w0 = rng.normal(0, 0.1, (n, d))
    lr, wd = 0.05, 0.1
    opt = make_opt(str(tmp_path), w0, lr=lr, k=0.0, weight_decay=wd,
                   weight_decay_mode=mode)
    ref = RiemannianAdamReference(w0, lr=lr, k=0.0, weight_decay=wd,
                                  weight_decay_mode=mode)
    for _ in range(5):
        rows = rng.integers(0, n, 20)
        g = rng.normal(0, 1.0, (len(rows), d))
        do_step(opt, rows, g)
        ref.step(rows, g)
    opt.shutdown()
    assert_matches_reference(opt, ref)


def test_c4_l2_is_adam_l2_not_adamw(tmp_path):
    """The default must be classic Adam L2 (decay inside the gradient), not AdamW."""
    n, d = 6, 2
    w0 = np.full((n, d), 0.5)
    lr, wd = 0.1, 0.1
    g = np.ones((1, d))

    opt = make_opt(str(tmp_path), w0, lr=lr, k=0.0, weight_decay=wd,
                   weight_decay_mode="l2")
    do_step(opt, [0], g)
    opt.shutdown()
    w, *_ = read_state(opt)

    ref = RiemannianAdamReference(w0, lr=lr, k=0.0, weight_decay=wd,
                                  weight_decay_mode="l2")
    ref.step([0], g)
    assert np.allclose(w[0], ref.w[0], atol=1e-6)

    # AdamW with the same coefficients differs substantially.
    x = w0[0].copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    m = BETA1 * m + (1 - BETA1) * g[0]
    v = BETA2 * v + (1 - BETA2) * g[0] ** 2
    direction = (m / (1 - BETA1)) / (np.sqrt(v / (1 - BETA2)) + EPS)
    adamw = x * (1 - lr * wd) - lr * direction
    assert np.max(np.abs(w[0] - adamw)) > 1e-3, \
        "L2-режим не должен совпадать с AdamW"


def test_c4_hyperbolic_l2_is_riemannian_gradient(tmp_path):
    """L2 decay is part of the gradient, not a coordinate shrink.

    With ``g_E = -wd * w`` the L2 gradient is exactly zero, so Adam has no
    direction and the point must not move.  The decoupled variant has no such
    cancellation: it applies ``wd * w`` as a separate tangent term.
    """
    n, d = 2, 2
    k, lr, wd = 1.0, 0.1, 0.5
    w0 = np.array([[0.05, 0.0], [0.9, 0.0]])
    g = -wd * w0

    opt_l2 = make_opt(str(tmp_path / "l2"), w0, lr=lr, k=k,
                      weight_decay=wd, weight_decay_mode="l2")
    do_step(opt_l2, [0, 1], g)
    opt_l2.shutdown()
    w_l2, *_ = read_state(opt_l2)
    assert np.allclose(w_l2, w0, atol=1e-6), \
        "L2-штраф обязан сократиться с g = -wd*w"

    opt_dec = make_opt(str(tmp_path / "dec"), w0, lr=lr, k=k,
                       weight_decay=wd, weight_decay_mode="decoupled")
    do_step(opt_dec, [0, 1], g)
    opt_dec.shutdown()
    w_dec, *_ = read_state(opt_dec)
    assert not np.allclose(w_dec, w0, atol=1e-4), \
        "decoupled-штраф не участвует в градиенте и не должен сокращаться"


def test_c4_invalid_weight_decay_mode(tmp_path):
    with pytest.raises(ValueError, match="weight_decay_mode"):
        make_opt(str(tmp_path), np.zeros((3, 2)), weight_decay_mode="adamw")


# ───────────────────────── F7 / H1: ball invariants ─────────────────────────

def test_h1_init_is_scaled_into_ball(tmp_path):
    n, d = 4, 8
    w0 = np.full((n, d), 0.5)  # norm 1.414 > radius 1
    opt = DiskSparseRiemannianAdam(
        {"ent": torch.tensor(w0)}, lr=0.01, k=1.0, disk_dir=str(tmp_path),
        resume=False)  # default init_max_norm_fraction=0.5
    w = np.array(opt.state_files["ent"]["w"])
    radius = math.sqrt((1 - manifold.BALL_EPS) / 1.0)
    norms = np.linalg.norm(w, axis=1)
    assert np.all(norms <= 0.5 * radius + 1e-6)
    assert np.isfinite(manifold.distance(
        torch.tensor(w[:1]), torch.tensor(w[1:2]), 1.0).numpy()).all()
    opt.shutdown()


def test_h1_init_fraction_one_only_projects(tmp_path):
    n, d = 4, 8
    w0 = np.full((n, d), 0.5)
    opt = DiskSparseRiemannianAdam(
        {"ent": torch.tensor(w0)}, lr=0.01, k=1.0, disk_dir=str(tmp_path),
        resume=False, init_max_norm_fraction=1.0)
    w = np.array(opt.state_files["ent"]["w"])
    radius = math.sqrt((1 - manifold.BALL_EPS) / 1.0)
    assert np.all(np.linalg.norm(w, axis=1) <= radius + 1e-6)
    opt.shutdown()


def test_h1_read_rows_projects(tmp_path):
    n, d, k = 8, 4, 1.0
    opt = make_opt(str(tmp_path), np.full((n, d), 0.1), k=k)
    opt.state_files["ent"]["w"][2] = np.full(d, 2.5, dtype=np.float32)
    opt.state_files["ent"]["w"].flush()

    rows = opt.read_rows([2]).numpy()
    radius = math.sqrt((1 - manifold.BALL_EPS) / k)
    assert np.linalg.norm(rows[0]) <= radius + 1e-6
    opt.shutdown()


def test_h1_resume_projects_corrupted_rows(tmp_path):
    n, d, k = 6, 3, 1.0
    opt = make_opt(str(tmp_path), np.full((n, d), 0.1), k=k)
    opt.shutdown()
    with open(os.path.join(str(tmp_path), "ent_w.dat"), "r+b") as f:
        f.seek(0)
        f.write(np.full(d, 3.0, dtype=np.float32).tobytes())
        f.write(np.full(d, -3.0, dtype=np.float32).tobytes())

    opt2 = make_opt(str(tmp_path), np.zeros((n, d)), k=k, resume=True)
    w = np.array(opt2.state_files["ent"]["w"])
    radius = math.sqrt((1 - manifold.BALL_EPS) / k)
    assert np.all(np.linalg.norm(w, axis=1) <= radius + 1e-6)
    assert opt2._resume_rows_projected == 2
    opt2.shutdown()


def test_retraction_keeps_norm_in_ball(tmp_path):
    """An extreme outward step keeps every row inside the ball and finite.

    The float64 reference cannot be used here: without projection it reaches
    the boundary exactly (lambda = inf), which is not a point of the manifold.
    Exact numerical agreement is checked by the H2 tests away from the
    boundary; this test checks the invariant that matters at the boundary.
    """
    n, d, k = 6, 4, 1.0
    max_radius = math.sqrt((1.0 - manifold.BALL_EPS) / k)
    w0 = np.full((n, d), 0.45)
    opt = make_opt(str(tmp_path), w0, lr=10.0, k=k)
    for step in range(8):
        g = np.full((n, d), 100.0)
        do_step(opt, np.arange(n), g)
        w, m, v, t = read_state(opt)
        assert np.all(np.linalg.norm(w, axis=1) <= max_radius + 1e-6)
        assert np.isfinite(w).all() and np.isfinite(m).all() and np.isfinite(v).all()
        assert np.all(t == step + 1), "строка была сброшена вместо шага"
    opt.shutdown()


def test_f7_corrupted_row_is_projected_on_step(tmp_path):
    n, d, k = 8, 4, 1.0
    max_radius = math.sqrt((1.0 - manifold.BALL_EPS) / k)
    opt = make_opt(str(tmp_path), np.full((n, d), 0.1), lr=0.01, k=k)
    opt.state_files["ent"]["w"][2] = np.full(d, 2.5, dtype=np.float32)
    opt.state_files["ent"]["w"].flush()

    do_step(opt, [2], np.full((1, d), 0.1))
    opt.shutdown()
    w, m, v, t = read_state(opt)
    assert np.linalg.norm(w[2]) <= max_radius + 1e-6
    assert np.isfinite(w[2]).all() and np.isfinite(m[2]).all()
    assert t[2] == 1


# ───────────────────────── H2: full Riemannian Adam ─────────────────────────

@pytest.mark.parametrize("k", [0.0, 0.7])
@pytest.mark.parametrize("mode", ["l2", "decoupled"])
def test_h2_matches_independent_reference(tmp_path, k, mode):
    n, d = 30, 5
    rng = np.random.default_rng(9)
    w0 = rng.normal(0, 0.1, (n, d))
    if k > 0:
        w0 *= 0.8 * math.sqrt(1 / k) / np.linalg.norm(w0, axis=1).max()
    lr, wd = 0.03, 0.02
    opt = make_opt(str(tmp_path), w0, lr=lr, k=k, weight_decay=wd,
                   weight_decay_mode=mode)
    ref = RiemannianAdamReference(w0, lr=lr, k=k, weight_decay=wd,
                                  weight_decay_mode=mode)
    for _ in range(6):
        rows = rng.integers(0, n, 12)
        g = rng.normal(0, 0.5, (len(rows), d))
        do_step(opt, rows, g)
        ref.step(rows, g)
    opt.shutdown()
    assert_matches_reference(opt, ref, atol=1e-5)


def test_h2_moments_are_transported_isometrically(tmp_path):
    n, d, k = 1, 3, 0.7
    lr = 0.05
    w0 = np.array([[0.2, -0.1, 0.15]])
    g = np.array([[0.3, -0.4, 0.2]])
    opt = make_opt(str(tmp_path), w0, lr=lr, k=k)
    do_step(opt, [0], g)
    opt.shutdown()
    w, m, v, _ = read_state(opt)

    g_riem = g[0] * (1 - k * np.dot(w0[0], w0[0])) ** 2 / 4
    m_before = (1 - BETA1) * g_riem
    lam0 = 2.0 / (1 - k * np.dot(w0[0], w0[0]))
    lam1 = 2.0 / (1 - k * np.dot(w[0], w[0]))
    assert np.isclose(lam1 * np.linalg.norm(m[0]),
                      lam0 * np.linalg.norm(m_before), rtol=1e-5, atol=1e-7)
    v_before = (1 - BETA2) * g_riem ** 2
    assert np.isclose(lam1 ** 2 * np.sum(v[0]),
                      lam0 ** 2 * np.sum(v_before), rtol=1e-5, atol=1e-9)


def test_h2_first_step_from_origin_closed_form(tmp_path):
    n, d, k = 1, 4, 0.7
    lr = 0.1
    g = np.array([[0.5, -1.0, 2.0, -0.25]])
    opt = make_opt(str(tmp_path), np.zeros((n, d)), lr=lr, k=k)
    do_step(opt, [0], g)
    opt.shutdown()
    w, *_ = read_state(opt)

    direction = g[0] / (np.abs(g[0]) + EPS)  # first Adam step at t=1
    # exp_0(-lr * direction) = -tanh(sqrt(k)*lr*||direction||) /
    #                          (sqrt(k)*||direction||) * direction
    norm = np.linalg.norm(direction)
    expected = (-math.tanh(math.sqrt(k) * lr * norm)
                / (math.sqrt(k) * norm) * direction)
    assert np.allclose(w[0], expected, atol=1e-6)


def test_h2_zero_gradient_is_identity(tmp_path):
    n, d, k = 3, 4, 1.0
    w0 = np.full((n, d), 0.1)
    opt = make_opt(str(tmp_path), w0, lr=0.1, k=k)
    do_step(opt, np.arange(n), np.zeros((n, d)))
    opt.shutdown()
    w, m, v, t = read_state(opt)
    assert np.allclose(w, w0, atol=1e-6)
    assert np.all(m == 0) and np.all(v == 0) and np.all(t == 1)


# ───────────────────────── H3: curvature changes ─────────────────────────

def test_h3_set_curvature_reprojects_whole_table(tmp_path):
    n, d = 8, 3
    k0, k1 = 0.25, 4.0
    radius0 = math.sqrt((1 - manifold.BALL_EPS) / k0)
    w0 = np.full((n, d), 0.95 * radius0 / math.sqrt(d))
    opt = make_opt(str(tmp_path), w0, k=k0)
    opt.set_curvature(k1)
    w = np.array(opt.state_files["ent"]["w"])
    radius1 = math.sqrt((1 - manifold.BALL_EPS) / k1)
    assert np.all(np.linalg.norm(w, axis=1) <= radius1 + 1e-6)
    assert opt.param_groups[0]["k"] == k1
    with open(os.path.join(str(tmp_path), "dsa_global_meta.json")) as f:
        assert json.load(f)["k"] == k1
    opt.shutdown()


def test_h3_current_k_reprojects_whole_table(tmp_path):
    n, d = 6, 3
    k0, k1 = 0.25, 4.0
    radius0 = math.sqrt((1 - manifold.BALL_EPS) / k0)
    w0 = np.full((n, d), 0.9 * radius0 / math.sqrt(d))
    opt = make_opt(str(tmp_path), w0, k=k0)
    do_step(opt, [0], np.ones((1, d)))
    opt.step(updates={"ent": (torch.tensor([1]), torch.ones(1, d))},
             current_k=k1)
    w = np.array(opt.state_files["ent"]["w"])
    radius1 = math.sqrt((1 - manifold.BALL_EPS) / k1)
    assert np.all(np.linalg.norm(w, axis=1) <= radius1 + 1e-6)
    opt.shutdown()


def test_h3_set_curvature_can_reset_moments(tmp_path):
    n, d = 4, 3
    opt = make_opt(str(tmp_path), np.full((n, d), 0.05), k=0.5)
    do_step(opt, [0, 1], np.ones((2, d)))
    opt.set_curvature(0.5, reset_moments=True)
    m, v, t = read_state(opt)[1:]
    assert np.all(m == 0) and np.all(v == 0) and np.all(t == 0)
    opt.shutdown()


# ───────────────────────── H4: curvature validation ─────────────────────────

@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_h4_invalid_k_rejected_at_construction(tmp_path, bad):
    with pytest.raises((ValueError, TypeError)):
        make_opt(str(tmp_path), np.zeros((3, 2)), k=bad)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_h4_invalid_k_rejected_at_runtime(tmp_path, bad):
    opt = make_opt(str(tmp_path), np.zeros((3, 2)), k=0.0)
    with pytest.raises((ValueError, TypeError)):
        opt.set_curvature(bad)
    with pytest.raises((ValueError, TypeError)):
        opt.step(updates={"ent": (torch.tensor([0]), torch.ones(1, 2))},
                 current_k=bad)
    opt.shutdown()


# ───────────────────────── A1: retryable shutdown ─────────────────────────

def test_a1_shutdown_retry_after_timeout(tmp_path):
    n, d = 20, 3
    opt = make_opt(str(tmp_path), np.zeros((n, d)), k=0.0)
    open_gate = gate_writer(opt)
    do_step(opt, [0, 1], np.ones((2, d)))
    do_step(opt, [2, 3], np.ones((2, d)))

    with pytest.raises(RuntimeError, match="НЕ дописаны"):
        opt.writer.shutdown(timeout=0.2)
    assert opt.writer.state.name == "FAILED"
    assert opt.writer.worker_thread.is_alive()
    assert opt.writer.queue.unfinished_tasks == 2

    open_gate()
    assert opt.writer.shutdown(timeout=10.0) is True
    assert opt.writer.state.name == "STOPPED"
    assert not opt.writer.worker_thread.is_alive()
    assert opt.writer.queue.unfinished_tasks == 0

    # Idempotent after a clean stop.
    assert opt.writer.shutdown(timeout=1.0) is True


def test_a1_shutdown_reports_persistent_write_error(tmp_path):
    n, d = 8, 2
    opt = make_opt(str(tmp_path), np.zeros((n, d)), k=0.0)

    def broken(*a, **kw):
        raise OSError("simulated disk failure")

    opt.writer._apply_write = broken
    do_step(opt, [0, 1], np.ones((2, d)))
    with pytest.raises(RuntimeError, match="НЕ записан"):
        opt.shutdown()
    assert opt.writer.state.name == "FAILED"
    assert not opt.writer.worker_thread.is_alive(), \
        "после сбора ошибок воркер должен быть остановлен"
    # A second call must not pretend that the failed write succeeded.
    with pytest.raises(RuntimeError, match="НЕ записан"):
        opt.shutdown()


def test_a1_put_after_shutdown_raises(tmp_path):
    n, d = 6, 2
    opt = make_opt(str(tmp_path), np.zeros((n, d)), k=0.0)
    opt.shutdown()
    with pytest.raises(RuntimeError, match="запись невозможна"):
        opt.writer.put_update("ent", torch.tensor([0]),
                              torch.zeros(1, d), torch.zeros(1, d),
                              torch.zeros(1, d), torch.tensor([1]))


# ───────────────────────── A2: global step ─────────────────────────

def test_a2_global_step_survives_resume_and_checkpoint(tmp_path):
    n, d = 6, 2
    opt = make_opt(str(tmp_path), np.zeros((n, d)), k=0.0)
    for _ in range(3):
        do_step(opt, [0], np.ones((1, d)))
    assert opt._global_step == 3
    opt.shutdown()

    opt2 = make_opt(str(tmp_path), np.zeros((n, d)), k=0.0, resume=True)
    assert opt2._global_step == 3

    ckpt_dir = str(tmp_path / "ckpt")
    opt2.export_checkpoint(ckpt_dir)
    opt2.shutdown()

    opt3 = make_opt(str(tmp_path / "fresh"), np.zeros((n, d)), k=0.0)
    manifest = opt3.import_checkpoint(ckpt_dir)
    assert manifest["global_step"] == 3
    assert opt3._global_step == 3
    opt3.shutdown()


# ───────────────────────── A3: index validation ─────────────────────────

def test_a3_negative_and_out_of_range_indices_rejected(tmp_path):
    n, d = 6, 3
    opt = make_opt(str(tmp_path), np.zeros((n, d)), k=0.0)
    with pytest.raises(IndexError, match="отрицательный"):
        do_step(opt, [-1], np.ones((1, d)))
    with pytest.raises(IndexError, match="за границы"):
        do_step(opt, [n], np.ones((1, d)))
    with pytest.raises(TypeError, match="целочисленными"):
        opt.step(updates={"ent": (torch.tensor([0.0]), torch.ones(1, d))})
    with pytest.raises(ValueError, match="1-D"):
        opt.step(updates={"ent": (torch.tensor([[0]]), torch.ones(1, d))})
    _, _, _, t = read_state(opt)
    assert np.all(t == 0), "невалидный индекс не должен менять состояние"
    opt.shutdown()


# ───────────────────────── A4: NaN row reset ─────────────────────────

def test_a4_nan_row_is_fully_reset(tmp_path):
    n, d = 6, 3
    lr = 0.1
    opt = make_opt(str(tmp_path), np.zeros((n, d)), lr=lr, k=0.0)
    opt.writer.wait_all_flushed()

    sf = opt.state_files["ent"]
    sf["w"][2] = np.nan
    sf["m"][2] = 7.0
    sf["v"][2] = 9.0
    sf["t"][2] = 50
    for mm in sf.values():
        mm.flush()

    g = np.array([[1.0, -2.0, 0.5]])
    do_step(opt, [2], g)
    opt.shutdown()

    w, m, v, t = read_state(opt)
    expected = -lr * g[0] / (np.abs(g[0]) + EPS)
    assert np.allclose(w[2], expected, atol=1e-7)
    assert np.allclose(m[2], (1 - BETA1) * g[0], atol=1e-7)
    assert np.allclose(v[2], (1 - BETA2) * g[0] ** 2, atol=1e-7)
    assert t[2] == 1, f"счётчик строки должен перезапуститься, получено {t[2]}"
    assert opt._nan_rows_replaced == 1


def test_a4_zero_counter_discards_stale_moments(tmp_path):
    """A row with t=0 is fresh; leftover m/v from a lost counter must not leak."""
    n, d = 4, 2
    opt = make_opt(str(tmp_path), np.zeros((n, d)), lr=0.1, k=0.0)
    opt.writer.wait_all_flushed()
    sf = opt.state_files["ent"]
    sf["m"][1] = 5.0
    sf["v"][1] = 7.0
    sf["t"][1] = 0
    for mm in sf.values():
        mm.flush()

    g = np.array([[1.0, -2.0]])
    do_step(opt, [1], g)
    opt.shutdown()
    _, m, v, t = read_state(opt)
    assert np.allclose(m[1], (1 - BETA1) * g[0], atol=1e-7)
    assert np.allclose(v[1], (1 - BETA2) * g[0] ** 2, atol=1e-7)
    assert t[1] == 1


def test_a4_nan_gradient_rejected(tmp_path):
    n, d = 4, 2
    opt = make_opt(str(tmp_path), np.zeros((n, d)), k=0.0)
    g = np.ones((1, d))
    g[0, 1] = np.nan
    with pytest.raises(ValueError, match="NaN/inf"):
        do_step(opt, [0], g)
    _, _, _, t = read_state(opt)
    assert np.all(t == 0)
    opt.shutdown()


def test_a9_gradient_shape_validation(tmp_path):
    n, d = 4, 2
    opt = make_opt(str(tmp_path), np.zeros((n, d)), k=0.0)
    with pytest.raises(ValueError, match="форму"):
        do_step(opt, [0], np.ones((2, d)))
    with pytest.raises(ValueError, match="форму"):
        do_step(opt, [0], np.ones((1, d + 1)))
    opt.shutdown()


# ───────────────────────── dedup / scaling ─────────────────────────

def test_dedup_equals_manual_aggregation(tmp_path):
    for k in (0.0, 1.0):
        dirA = tempfile.mkdtemp()
        dirB = tempfile.mkdtemp()
        try:
            g1, g2, g3 = (np.array([[0.3, -1.2, 0.7]]),
                          np.array([[-0.4, 0.2, 1.1]]),
                          np.array([[1.5, -0.6, 0.2]]))
            w0 = np.array([[0.05, -0.02, 0.03] * 8]).reshape(8, 3)
            optA = make_opt(dirA, w0, lr=0.05, k=k)
            do_step(optA, [3, 3, 5], np.concatenate([g1, g2, g3]))
            optA.shutdown()
            optB = make_opt(dirB, w0, lr=0.05, k=k)
            do_step(optB, [3, 5], np.concatenate([g1 + g2, g3]))
            optB.shutdown()

            wA, mA, vA, tA = read_state(optA)
            wB, mB, vB, tB = read_state(optB)
            assert np.allclose(wA[[3, 5]], wB[[3, 5]], atol=1e-8)
            assert np.allclose(mA[[3, 5]], mB[[3, 5]], atol=1e-8)
            assert np.allclose(vA[[3, 5]], vB[[3, 5]], atol=1e-8)
            assert np.array_equal(tA, tB) and tA[3] == 1 and tA[5] == 1
        finally:
            shutil.rmtree(dirA, ignore_errors=True)
            shutil.rmtree(dirB, ignore_errors=True)


def test_riemannian_gradient_scaling_formula(tmp_path):
    n, d, k = 4, 3, 0.7
    lr = 0.05
    w0 = np.zeros((n, d))
    w0[0] = np.array([0.3, -0.4, 0.2657])
    opt = make_opt(str(tmp_path), w0, lr=lr, k=k)
    g = np.array([[1.0, -0.5, 2.0]])
    do_step(opt, [0], g)
    opt.shutdown()
    w, m, _, _ = read_state(opt)

    sq = float(np.sum(w0[0] ** 2))
    g_riem = g[0] * (1.0 - k * sq) ** 2 / 4.0
    lam0 = 2.0 / (1 - k * sq)
    lam1 = 2.0 / (1 - k * np.sum(w[0] ** 2))
    assert np.isclose(lam1 * np.linalg.norm(m[0]),
                      lam0 * np.linalg.norm((1 - BETA1) * g_riem),
                      rtol=1e-5, atol=1e-7)
    expected = expmap(w0[0][None, :],
                      -lr * np.sign(g_riem)[None, :], k)[0]
    assert np.allclose(w[0], expected, atol=1e-6)


# ───────────────────────── F5: sorted indices ─────────────────────────

def test_f5_indices_sorted_before_write(tmp_path):
    n, d = 16, 3
    opt = make_opt(str(tmp_path), np.zeros((n, d)))
    captured = []
    orig = opt.writer._apply_write

    def capturing(name, indices, w_np, m_np, v_np, t_np=None):
        captured.append(np.asarray(indices).copy())
        return orig(name, indices, w_np, m_np, v_np, t_np)

    opt.writer._apply_write = capturing
    do_step(opt, [9, 2, 7, 4, 0], np.ones((5, d)))
    do_step(opt, [5, 1, 5, 3], np.ones((4, d)))
    opt.shutdown()

    assert len(captured) == 2
    for batch in captured:
        assert np.all(np.diff(batch) > 0), f"индексы не отсортированы: {batch}"


def test_f5_apply_write_maps_sorted_correctly(tmp_path):
    n, d = 10, 3
    opt = make_opt(str(tmp_path), np.zeros((n, d)))
    opt.writer.wait_all_flushed()
    idx = np.array([7, 2])
    w_rows = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32)
    zeros = np.zeros_like(w_rows)
    t_rows = np.array([3, 4], dtype=np.int32)
    opt.writer._apply_write("ent", idx, w_rows, zeros, zeros, t_rows)
    w, _, _, t = read_state(opt)
    assert np.allclose(w[7], 1.0) and np.allclose(w[2], 2.0)
    assert t[7] == 3 and t[2] == 4
    opt.shutdown()


# ───────────────────────── F6: honest shutdown ─────────────────────────

def test_f6_shutdown_raises_when_cannot_finish(tmp_path):
    n, d = 40, 3
    w0 = np.zeros((n, d))
    opt = make_opt(str(tmp_path), w0, lr=0.05, k=0.0)
    open_gate = gate_writer(opt)
    do_step(opt, np.arange(0, 10), np.ones((10, d)))
    do_step(opt, np.arange(10, 20), np.ones((10, d)))

    with pytest.raises(RuntimeError, match="НЕ дописаны"):
        opt.writer.shutdown(timeout=0.3)
    open_gate()
    assert opt.writer.shutdown(timeout=10.0) is True


# ───────────────────────── F8 / C2: checkpoints ─────────────────────────

def test_c2_checkpoint_is_portable_and_self_contained(tmp_path):
    n, d = 24, 3
    rng = np.random.default_rng(11)
    w0 = rng.normal(0, 0.1, (n, d))
    opt = make_opt(str(tmp_path / "state"), w0, lr=0.05, k=0.0)
    rows = [2, 5, 11]
    for g_val in (1.0, -0.7):
        do_step(opt, rows, np.full((len(rows), d), g_val))
    ckpt_dir = str(tmp_path / "checkpoint")
    manifest_path = opt.export_checkpoint(ckpt_dir)
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest["format"] == "DSA_checkpoint"
    assert manifest["version"] == 2
    assert manifest["global_step"] == 2
    for info in manifest["params"].values():
        for rec in info["files"].values():
            assert os.path.basename(rec["name"]) == rec["name"]
            assert not os.path.isabs(rec["name"])
    opt.shutdown()
    w1, m1, v1, t1 = read_state(opt)

    # Move the checkpoint: absolute paths would break here.
    moved = str(tmp_path / "moved")
    shutil.copytree(ckpt_dir, moved)

    opt2 = make_opt(str(tmp_path / "restored"), np.zeros((n, d)), lr=0.05, k=0.0)
    opt2.import_checkpoint(moved)
    w2, m2, v2, t2 = read_state(opt2)
    assert np.array_equal(w1, w2) and np.array_equal(m1, m2)
    assert np.array_equal(v1, v2) and np.array_equal(t1, t2)
    assert opt2._global_step == 2

    ref = RiemannianAdamReference.__new__(RiemannianAdamReference)
    ref.w, ref.m, ref.v, ref.t = (np.array(w1, dtype=np.float64),
                                  np.array(m1, dtype=np.float64),
                                  np.array(v1, dtype=np.float64),
                                  np.array(t1, dtype=np.int64))
    ref.n, ref.d = n, d
    ref.lr, ref.b1, ref.b2, ref.eps = 0.05, BETA1, BETA2, EPS
    ref.k, ref.wd, ref.wd_mode = 0.0, 0.0, "l2"
    do_step(opt2, rows, np.full((len(rows), d), 3.0))
    ref.step(rows, np.full((len(rows), d), 3.0))
    opt2.shutdown()
    assert_matches_reference(opt2, ref, rows=rows)


def test_c2_corrupted_checkpoint_rejected(tmp_path):
    n, d = 6, 2
    opt = make_opt(str(tmp_path / "state"), np.zeros((n, d)))
    ckpt_dir = str(tmp_path / "ckpt")
    opt.export_checkpoint(ckpt_dir)
    opt.shutdown()

    manifest = json.load(open(os.path.join(ckpt_dir, "checkpoint.json")))
    victim = os.path.join(ckpt_dir,
                          manifest["params"]["ent"]["files"]["w"]["name"])
    with open(victim, "r+b") as f:
        f.seek(0)
        f.write(b"\x01\x02\x03\x04")

    opt2 = make_opt(str(tmp_path / "restored"), np.zeros((n, d)))
    before = np.array(opt2.state_files["ent"]["w"])
    with pytest.raises(RuntimeError, match="SHA-256"):
        opt2.import_checkpoint(ckpt_dir)
    after = np.array(opt2.state_files["ent"]["w"])
    assert np.array_equal(before, after), "неудачный импорт изменил живое состояние"
    opt2.shutdown()


def test_c2_old_manifest_and_json_paths_rejected(tmp_path):
    n, d = 6, 2
    opt = make_opt(str(tmp_path), np.zeros((n, d)))
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"format": "DSA_checkpoint", "version": 1,
                               "params": {}}))
    with pytest.raises(RuntimeError, match="каталог"):
        opt.import_checkpoint(str(old))
    with pytest.raises(ValueError, match="КАТАЛОГ"):
        opt.save_checkpoint(str(old))
    opt.shutdown()


def test_c2_checkpoint_immediately_after_init_is_valid(tmp_path):
    n, d, k = 5, 4, 1.0
    opt = DiskSparseRiemannianAdam(
        {"ent": torch.full((n, d), 0.5)}, lr=0.01, k=k,
        disk_dir=str(tmp_path / "state"), resume=False)
    ckpt_dir = str(tmp_path / "ckpt")
    opt.export_checkpoint(ckpt_dir)
    opt.shutdown()

    opt2 = make_opt(str(tmp_path / "restored"), np.zeros((n, d)), k=k)
    opt2.import_checkpoint(ckpt_dir)
    w = np.array(opt2.state_files["ent"]["w"])
    radius = math.sqrt((1 - manifold.BALL_EPS) / k)
    assert np.all(np.linalg.norm(w, axis=1) <= radius + 1e-6)
    opt2.shutdown()


def test_state_dict_raises(tmp_path):
    opt = make_opt(str(tmp_path), np.zeros((4, 2)))
    with pytest.raises(NotImplementedError):
        opt.state_dict()
    with pytest.raises(NotImplementedError):
        opt.load_state_dict({})
    opt.shutdown()


# ───────────────────────── C3: source tensors released ─────────────────────────

def test_c3_source_tensor_is_not_retained(tmp_path):
    source = torch.zeros(8, 4)
    ref = weakref.ref(source)
    opt = DiskSparseRiemannianAdam({"ent": source}, lr=0.01, k=0.0,
                                   disk_dir=str(tmp_path), resume=False)
    del source
    gc.collect()
    assert ref() is None, "оптимизатор удержал исходный тензор (O(N) память)"
    assert opt.param_groups[0]["params"][0].numel() == 0
    opt.shutdown()


def test_c3_retain_params_opt_in(tmp_path):
    source = torch.zeros(8, 4)
    ref = weakref.ref(source)
    opt = DiskSparseRiemannianAdam({"ent": source}, lr=0.01, k=0.0,
                                   disk_dir=str(tmp_path), resume=False,
                                   retain_params=True)
    del source
    gc.collect()
    assert ref() is not None
    assert opt.param_groups[0]["params"][0].shape == (8, 4)
    opt.shutdown()
    del opt
    gc.collect()
    assert ref() is None


# ───────────────────────── public API ─────────────────────────

def test_public_api_compat(tmp_path):
    n, d = 10, 2
    table = torch.randn(n, d)
    opt = DiskSparseRiemannianAdam(
        params={"ent": table}, lr=0.01, k=0.5, disk_dir=str(tmp_path))
    assert hasattr(opt, "state_files") and "ent" in opt.state_files
    assert {"w", "m", "v", "t"} <= set(opt.state_files["ent"])
    assert hasattr(opt.writer, "_lock") and hasattr(opt.writer, "queue")
    opt.writer.queue.join()

    opt.param_groups[0]["lr"] *= 0.5
    assert opt.param_groups[0]["lr"] == 0.005

    idx = torch.tensor([1, 2])
    opt.step(updates={"ent": (idx, torch.ones(2, d))})
    opt.zero_grad()
    opt.shutdown()

    opt2 = DiskSparseRiemannianAdam([torch.randn(6, 3)], lr=0.01, k=0.0,
                                    disk_dir=str(tmp_path / "b"))
    opt2.step(updates={"param_0": (torch.tensor([0]), torch.ones(1, 3))})
    opt2.shutdown()


# ───────────────────────── R1: non-finite rows ─────────────────────────

def test_r1_nan_row_is_reset_on_step(tmp_path, caplog):
    n, d, k = 8, 4, 1.0
    max_radius = math.sqrt((1.0 - manifold.BALL_EPS) / k)
    w0 = np.full((n, d), 0.1)
    opt = make_opt(str(tmp_path), w0, lr=0.01, k=k)
    opt.shutdown()

    with open(os.path.join(str(tmp_path), "ent_w.dat"), "r+b") as f:
        f.seek(2 * d * 4)
        f.write(np.full(d, np.nan, dtype=np.float32).tobytes())
        f.seek(3 * d * 4)
        f.write(np.full(d, np.inf, dtype=np.float32).tobytes())

    opt2 = make_opt(str(tmp_path), w0, lr=0.01, k=k, resume=True)
    with caplog.at_level(logging.WARNING, logger="DSA"):
        do_step(opt2, [2, 3], np.full((2, d), 0.1))
    opt2.shutdown()

    w, m, v, t = read_state(opt2)
    for state in (w, m, v):
        assert np.isfinite(state[[2, 3]]).all()
    assert np.linalg.norm(w[2]) <= max_radius + 1e-6
    assert np.linalg.norm(w[3]) <= max_radius + 1e-6
    assert t[2] == 1 and t[3] == 1
    assert opt2._nan_rows_replaced == 2
    assert any("NaN" in r.message for r in caplog.records)


# ───────────────────────── R2: reinit marker ─────────────────────────

def test_r2_reinit_marker_blocks_resume(tmp_path):
    n, d = 8, 3
    opt = make_opt(str(tmp_path), np.zeros((n, d)))
    do_step(opt, [1, 2], np.ones((2, d)))
    opt.shutdown()

    meta_path = os.path.join(str(tmp_path), "ent_meta.json")
    meta = json.load(open(meta_path))
    meta["reinit_in_progress"] = True
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    with pytest.raises(RuntimeError, match="reinit_in_progress"):
        make_opt(str(tmp_path), np.zeros((n, d)), resume=True)


def test_r2_reinit_marker_cleared_by_full_reinit(tmp_path):
    n, d = 8, 3
    opt = make_opt(str(tmp_path), np.zeros((n, d)))
    opt.shutdown()
    meta_path = os.path.join(str(tmp_path), "ent_meta.json")
    meta = json.load(open(meta_path))
    meta["reinit_in_progress"] = True
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    fresh = np.random.default_rng(13).normal(0, 0.3, (n, d))
    opt2 = make_opt(str(tmp_path), fresh, lr=0.05, k=0.0, resume=False)
    opt2.shutdown()
    with open(meta_path) as f:
        assert "reinit_in_progress" not in json.load(f)
    w, m, v, t = read_state(opt2)
    assert np.allclose(w, fresh, atol=1e-7)
    assert np.all(m == 0) and np.all(v == 0) and np.all(t == 0)

    opt3 = make_opt(str(tmp_path), fresh, lr=0.05, k=0.0, resume=True)
    opt3.shutdown()


# ───────────────────────── R3: unknown update key ─────────────────────────

def test_r3_unknown_updates_key_warns_and_counts(tmp_path, caplog):
    n, d = 6, 3
    opt = make_opt(str(tmp_path), np.full((n, d), 0.1))
    before = opt._unknown_key_warnings
    with caplog.at_level(logging.WARNING, logger="DSA"):
        do_step(opt, [1], np.ones((1, d)))
        opt.step(updates={"entt": (torch.tensor([2]), torch.ones(1, d))})
    opt.shutdown()

    assert opt._unknown_key_warnings == before + 1
    assert any("entt" in r.message for r in caplog.records)
    w, *_ = read_state(opt)
    assert np.allclose(w[2], 0.1, atol=1e-7)


def test_r3_unknown_key_raises_in_debug_mode(tmp_path, monkeypatch):
    import dsa.optimizer as dsa_opt
    n, d = 6, 3
    opt = make_opt(str(tmp_path), np.zeros((n, d)))
    monkeypatch.setattr(dsa_opt, "DEBUG", True)
    with pytest.raises(KeyError, match="entt"):
        opt.step(updates={"entt": (torch.tensor([2]), torch.ones(1, d))})
    assert opt._unknown_key_warnings == 0
    opt.shutdown()

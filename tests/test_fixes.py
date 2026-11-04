"""Тесты исправлений дефектов F1–F8 из разбора DSA (см. ../CHANGELOG.md).

Каждый тест привязан к дефекту:
  F1 — очередь не теряет записи при медленном диске; синхронный fallback.
  F2 — два последовательных шага с теми же строками не читают устаревшее
        состояние; при непересекающихся строках дрена нет.
  F3 — resume/reinit/shape-mismatch/частичное состояние/отсутствие meta.
  F4 — пер-строчная bias-коррекция; первый шаг свежей строки = lr*sign(g).
  F5 — индексы сортируются перед чтением и записью.
  F6 — честный shutdown: ошибка вместо «всё записано».
  F7 — прочитанные из файла w проецируются в шар.
  F8 — save_checkpoint/load_checkpoint; state_dict честно бросает исключение.
  R1 — не-конечные строки (NaN/inf) из битого файла обрабатываются на чтении.
  R2 — маркер незавершённой переинициализации в meta.
  R3 — неизвестный ключ updates: warning со счётчиком, исключение в debug.
Плюс математические проверки: ретракция, дедупликация, k=0 (евклидов путь),
формула масштабирования (1-k*||w||^2)^2/4 == egrad/lambda^2.
"""
import json
import os
import shutil
import tempfile
import threading
import time

import numpy as np
import pytest
import torch

from dsa.optimizer import DiskSparseRiemannianAdam

BETA1, BETA2, EPS = 0.9, 0.999, 1e-8


# ───────────────────────── эталонная модель ─────────────────────────

class RefAdam:
    """Та же математика, что в DiskSparseRiemannianAdam.step, но во float64
    и в оперативной памяти. Используется как эталон для аналитических проверок."""

    def __init__(self, w0, lr, beta1=BETA1, beta2=BETA2, eps=EPS, k=0.0, wd=0.0):
        self.w = np.array(w0, dtype=np.float64)
        self.n, self.d = self.w.shape
        self.m = np.zeros_like(self.w)
        self.v = np.zeros_like(self.w)
        self.t = np.zeros(self.n, dtype=np.int64)
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.k, self.wd = k, wd

    @classmethod
    def from_state(cls, w, m, v, t, **kw):
        obj = cls(w, lr=kw.get('lr', 1e-3))
        obj.w = np.array(w, dtype=np.float64)
        obj.m = np.array(m, dtype=np.float64)
        obj.v = np.array(v, dtype=np.float64)
        obj.t = np.array(t, dtype=np.int64)
        obj.n, obj.d = obj.w.shape
        obj.b1 = kw.get('beta1', obj.b1)
        obj.b2 = kw.get('beta2', obj.b2)
        obj.eps = kw.get('eps', obj.eps)
        obj.k = kw.get('k', obj.k)
        obj.wd = kw.get('wd', obj.wd)
        return obj

    def _project(self, w):
        r = np.sqrt((1.0 - 1e-5) / self.k)
        n = np.linalg.norm(w, axis=-1, keepdims=True)
        return np.where(n > r, w / n * r, w)

    def step(self, idx, g):
        idx = np.asarray(idx, dtype=np.int64)
        g = np.asarray(g, dtype=np.float64)
        # дедупликация как в оптимизаторе (сумма градиентов дубликатов)
        uniq, inv = np.unique(idx, return_inverse=True)
        agg = np.zeros((len(uniq), g.shape[1]))
        for j, i in enumerate(inv):
            agg[i] += g[j]
        idx, g = uniq, agg
        order = np.argsort(idx)
        idx, g = idx[order], g[order]

        self.t[idx] += 1
        w = self.w[idx]
        if self.k > 1e-9:
            w = self._project(w)
            sq = np.sum(w * w, axis=-1, keepdims=True)
            sq = np.minimum(sq, (1.0 - 1e-5) / self.k)
            g = g * ((1.0 - self.k * sq) ** 2 / 4.0)
        if self.wd > 0:
            w = w * (1.0 - self.lr * self.wd)
        m, v = self.m[idx], self.v[idx]
        t = self.t[idx].astype(np.float64)
        m = self.b1 * m + (1 - self.b1) * g
        v = self.b2 * v + (1 - self.b2) * g * g
        m_hat = m / (1 - self.b1 ** t)[:, None]
        v_hat = v / (1 - self.b2 ** t)[:, None]
        w = w - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        if self.k > 1e-9:
            w = self._project(w)
        self.w[idx], self.m[idx], self.v[idx] = w, m, v


# ───────────────────────── помощники ─────────────────────────

def make_opt(disk_dir, w0, lr=0.05, k=0.0, **kw):
    params = torch.from_numpy(np.array(w0, dtype=np.float32))
    kw.setdefault('lr', lr)
    kw.setdefault('k', k)
    return DiskSparseRiemannianAdam(params={'ent': params}, disk_dir=disk_dir, **kw)


def do_step(opt, idx, g):
    idx_t = torch.tensor(np.asarray(idx, dtype=np.int64))
    g_t = torch.tensor(np.asarray(g, dtype=np.float32))
    opt.step(updates={'ent': (idx_t, g_t)})


def read_state(opt, name='ent'):
    opt.writer.wait_all_flushed()
    sf = opt.state_files[name]
    return (np.array(sf['w']), np.array(sf['m']), np.array(sf['v']),
            np.array(sf['t']))


def assert_matches_reference(opt, ref, name='ent', rows=None, atol=1e-5):
    w, m, v, t = read_state(opt, name)
    rows = np.arange(ref.n) if rows is None else np.asarray(rows)
    assert np.array_equal(t[rows], ref.t[rows]), \
        f"пер-строчные счётчики разошлись: {t[rows]} vs {ref.t[rows]}"
    assert np.allclose(w[rows], ref.w[rows], atol=atol), "w разошёлся с эталоном"
    assert np.allclose(m[rows], ref.m[rows], atol=atol), "m разошёлся с эталоном"
    assert np.allclose(v[rows], ref.v[rows], atol=atol), "v разошёлся с эталоном"


def slow_down_writer(opt, delay):
    """Подставной медленный писатель: задержка перед каждой записью батча."""
    orig = opt.writer._apply_write

    def slow(name, indices, w_np, m_np, v_np, t_np=None):
        time.sleep(delay)
        return orig(name, indices, w_np, m_np, v_np, t_np)

    opt.writer._apply_write = slow


def gate_writer(opt):
    """Полная остановка писателя; возвращает функцию открытия."""
    orig = opt.writer._apply_write
    gate = threading.Event()

    def gated(name, indices, w_np, m_np, v_np, t_np=None):
        gate.wait()
        return orig(name, indices, w_np, m_np, v_np, t_np)

    opt.writer._apply_write = gated
    return gate.set


# ───────────────────────── F1: потери записей ─────────────────────────

def test_f1_slow_disk_loses_nothing(tmp_path):
    """Медленный диск (подставной писатель): ни один батч не теряется,
    включая шаги с перекрытием строк (дрены по F2)."""
    n, d = 400, 4
    rng = np.random.default_rng(0)
    w0 = rng.normal(0, 0.05, (n, d))
    opt = make_opt(str(tmp_path), w0, lr=0.05, k=0.0, max_queue_size=2)
    slow_down_writer(opt, 0.02)

    ref = RefAdam(w0, lr=0.05, k=0.0)
    for s in range(15):
        rows = (s * 25 + np.arange(40)) % n
        g = rng.normal(0, 1.0, (len(rows), d))
        do_step(opt, rows, g)
        ref.step(rows, g)

    opt.shutdown()
    assert_matches_reference(opt, ref)


def test_f1_sync_fallback_on_stalled_writer(tmp_path):
    """Очередь полна и писатель стоит: put_update уходит в синхронную запись,
    данные не теряются (счётчик fallbacks растёт, состояние совпадает)."""
    n, d = 200, 4
    rng = np.random.default_rng(1)
    w0 = rng.normal(0, 0.05, (n, d))
    opt = make_opt(str(tmp_path), w0, lr=0.05, k=0.0,
                   max_queue_size=1, put_timeout=0.15)
    open_gate = gate_writer(opt)

    ref = RefAdam(w0, lr=0.05, k=0.0)
    threading.Timer(0.5, open_gate).start()  # открыть шлюз через 0.5 с

    for s in range(5):  # 5 батчей по 20 непересекающихся строк
        rows = np.arange(s * 20, s * 20 + 20)
        g = rng.normal(0, 1.0, (len(rows), d))
        do_step(opt, rows, g)
        ref.step(rows, g)

    opt.shutdown()
    assert opt.writer._sync_fallbacks >= 1, "синхронный fallback не сработал"
    assert_matches_reference(opt, ref, rows=np.arange(100))


# ───────────────────────── F2: гонка step/писатель ─────────────────────────

def test_f2_repeated_rows_never_stale(tmp_path):
    """Два+ последовательных шага по тем же строкам обязаны видеть результаты
    предыдущего шага (дрен), а не устаревшее состояние с диска."""
    n, d = 16, 3
    w0 = np.zeros((n, d))
    opt = make_opt(str(tmp_path), w0, lr=0.05, k=0.0)
    slow_down_writer(opt, 0.05)

    ref = RefAdam(w0, lr=0.05, k=0.0)
    rows = [3, 7]
    for g_val in (1.0, -2.0, 0.5):
        g = np.full((len(rows), d), g_val)
        do_step(opt, rows, g)
        ref.step(rows, g)

    opt.shutdown()
    assert_matches_reference(opt, ref, rows=rows)


def test_f2_disjoint_rows_do_not_drain(tmp_path):
    """Пересечение с строками в полёте пусто → step() не ждёт дрен очереди
    (асинхронность сохраняется в общем случае); непусто → ждёт."""
    n, d = 32, 3
    w0 = np.zeros((n, d))
    opt = make_opt(str(tmp_path), w0, lr=0.05, k=0.0)
    open_gate = gate_writer(opt)  # первый батч гарантированно висит в полёте

    do_step(opt, [0, 1], np.ones((2, d)))
    assert opt.writer.intersects_inflight('ent', np.array([0, 1]))
    assert not opt.writer.intersects_inflight('ent', np.array([5, 6]))

    drains = {'n': 0}

    def record_only_wait(*a, **kw):
        # заглушка: фиксируем сам факт запроса дрена; реальный join при
        # закрытом шлюзе завис бы, а проверяется здесь только логика step()
        drains['n'] += 1

    opt.writer.wait_all_flushed = record_only_wait
    do_step(opt, [5, 6], np.ones((2, d)))   # непересекающиеся → без дрена
    assert drains['n'] == 0, "лишний дрен на непересекающихся строках"
    do_step(opt, [0, 1], np.ones((2, d)))   # пересечение → дрен запрошен
    assert drains['n'] == 1

    open_gate()
    opt.shutdown()


# ───────────────────────── F3: состояние на диске ─────────────────────────

def test_f3_meta_written_on_fresh_init(tmp_path):
    n, d = 10, 2
    opt = make_opt(str(tmp_path), np.zeros((n, d)))
    meta_path = os.path.join(str(tmp_path), 'ent_meta.json')
    assert os.path.exists(meta_path)
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta['shape'] == [n, d]
    assert meta['dtype'] == 'float32' and meta['counter_dtype'] == 'int32'
    opt.shutdown()


def test_f3_resume_true_preserves_state(tmp_path):
    n, d = 12, 3
    rng = np.random.default_rng(2)
    w0 = rng.normal(0, 0.1, (n, d))
    opt = make_opt(str(tmp_path), w0, lr=0.05, k=0.0)
    rows = [1, 4, 9]
    for g_val in (1.0, -0.5):
        do_step(opt, rows, np.full((len(rows), d), g_val))
    opt.shutdown()

    w1, m1, v1, t1 = read_state(opt)
    assert t1[rows].tolist() == [2, 2, 2] and t1.sum() == 6

    # продолжаем: w из params НЕ должен перезаписать диск
    other = rng.normal(0, 5.0, (n, d))
    opt2 = make_opt(str(tmp_path), other, lr=0.05, k=0.0, resume=True)
    w2, m2, v2, t2 = read_state(opt2)
    assert np.array_equal(w1, w2) and np.array_equal(m1, m2)
    assert np.array_equal(v1, v2) and np.array_equal(t1, t2)

    # следующий шаг продолжает с того же места (счётчик строки = 3)
    ref = RefAdam.from_state(w1, m1, v1, t1, lr=0.05, k=0.0)
    do_step(opt2, rows, np.full((len(rows), d), 2.0))
    ref.step(rows, np.full((len(rows), d), 2.0))
    opt2.shutdown()
    assert_matches_reference(opt2, ref, rows=rows)


def test_f3_default_reinits_and_warns(tmp_path, caplog):
    import logging
    n, d = 12, 3
    w0 = np.full((n, d), 0.25)
    opt = make_opt(str(tmp_path), w0, lr=0.05, k=0.0)
    do_step(opt, [0, 1], np.ones((2, d)))
    opt.shutdown()

    fresh = rng_vals(3, n, d)
    with caplog.at_level(logging.WARNING, logger="DSA"):
        opt2 = make_opt(str(tmp_path), fresh, lr=0.05, k=0.0)  # resume=False по умолчанию
    assert any("переинициализация" in r.message for r in caplog.records)

    w, m, v, t = read_state(opt2)
    assert np.allclose(w, fresh, atol=1e-7), "w должен быть взят заново из params"
    assert np.all(m == 0) and np.all(v == 0) and np.all(t == 0)
    opt2.shutdown()


def rng_vals(seed, n, d):
    return np.random.default_rng(seed).normal(0, 0.3, (n, d))


def test_f3_shape_mismatch_raises(tmp_path):
    opt = make_opt(str(tmp_path), np.zeros((50, 4)))
    opt.shutdown()
    with pytest.raises(RuntimeError, match="несовпадение формы"):
        make_opt(str(tmp_path), np.zeros((60, 4)), resume=True)
    # то же и при resume=False: тихо выбрасывать чужое состояние нельзя
    with pytest.raises(RuntimeError, match="несовпадение формы"):
        make_opt(str(tmp_path), np.zeros((50, 8)))


def test_f3_partial_state_raises(tmp_path):
    opt = make_opt(str(tmp_path), np.zeros((10, 2)))
    opt.shutdown()
    os.remove(os.path.join(str(tmp_path), 'ent_v.dat'))
    with pytest.raises(RuntimeError, match="частичное состояние"):
        make_opt(str(tmp_path), np.zeros((10, 2)), resume=True)


def test_f3_missing_meta_raises(tmp_path):
    opt = make_opt(str(tmp_path), np.zeros((10, 2)))
    opt.shutdown()
    os.remove(os.path.join(str(tmp_path), 'ent_meta.json'))
    with pytest.raises(RuntimeError, match="meta"):
        make_opt(str(tmp_path), np.zeros((10, 2)), resume=True)


@pytest.mark.parametrize("suffix", ["w", "m", "v", "t"])
def test_f3_truncated_data_file_raises_on_resume(tmp_path, suffix):
    # Контрпример ревью: валидный meta + обрезанный файл данных (сбой диска
    # или копирования). np.memmap в r+ молча продлевает файл нулями —
    # открытие обязано быть явной ошибкой, файл не должен «вырастать обратно».
    n, d = 10, 2
    opt = make_opt(str(tmp_path), rng_vals(5, n, d), lr=0.05, k=0.0)
    do_step(opt, [0, 5], np.ones((2, d)))
    opt.shutdown()
    p = os.path.join(str(tmp_path), f'ent_{suffix}.dat')
    sz = os.path.getsize(p)
    with open(p, 'r+b') as f:
        f.truncate(sz // 2)
    with pytest.raises(RuntimeError, match="повреждён: размер"):
        make_opt(str(tmp_path), np.zeros((n, d)), resume=True)
    # memmap так и не открылся — файл НЕ продлился нулями молча
    assert os.path.getsize(p) == sz // 2


def test_f3_oversized_data_file_raises_on_resume(tmp_path):
    # Слишком длинный файл (обрыв копирования поверх старого состояния)
    # np.memmap отобразил бы молча только начало — тоже явная ошибка.
    n, d = 10, 2
    opt = make_opt(str(tmp_path), np.zeros((n, d)))
    opt.shutdown()
    p = os.path.join(str(tmp_path), 'ent_w.dat')
    with open(p, 'ab') as f:
        f.write(b'\x00' * 16)
    with pytest.raises(RuntimeError, match="повреждён: размер"):
        make_opt(str(tmp_path), np.zeros((n, d)), resume=True)


def test_f3_truncated_file_resume_false_reinits(tmp_path):
    # resume=False — документированный путь восстановления: полная
    # переинициализация пересоздаёт файлы правильного размера.
    n, d = 10, 2
    opt = make_opt(str(tmp_path), np.zeros((n, d)))
    opt.shutdown()
    p = os.path.join(str(tmp_path), 'ent_v.dat')
    with open(p, 'r+b') as f:
        f.truncate(os.path.getsize(p) // 3)
    fresh = rng_vals(7, n, d)
    opt2 = make_opt(str(tmp_path), fresh, lr=0.05, k=0.0)  # resume=False
    w, m, v, t = read_state(opt2)
    assert os.path.getsize(p) == n * d * 4, "файл должен быть пересоздан целиком"
    assert np.allclose(w, fresh, atol=1e-7)
    assert np.all(m == 0) and np.all(v == 0) and np.all(t == 0)
    opt2.shutdown()


# ───────────────────────── F4: пер-строчная bias-коррекция ─────────────────────────

def test_f4_first_step_of_fresh_row_is_lr_sign(tmp_path):
    """Аналитически: первый шаг свежей строки обязан давать w -= lr*sign(g),
    точнее w -= lr*g/(|g|+eps) — m_hat=g, v_hat=g^2 при t=1."""
    n, d = 10, 3
    lr = 0.1
    w0 = np.zeros((n, d))
    opt = make_opt(str(tmp_path), w0, lr=lr, k=0.0)

    g = np.array([[1.0, -2.0, 0.5]])
    do_step(opt, [4], g)
    opt.shutdown()

    w, m, v, t = read_state(opt)
    expected_delta = lr * g[0] / (np.abs(g[0]) + EPS)
    assert np.allclose(w[4], w0[4] - expected_delta, atol=1e-7)
    assert np.allclose(m[4], (1 - BETA1) * g[0], atol=1e-7)
    assert np.allclose(v[4], (1 - BETA2) * g[0] ** 2, atol=1e-7)
    assert t[4] == 1
    assert np.all(np.sign(w0[4] - w[4]) == np.sign(g[0])), \
        "направление первого шага должно быть -sign(g)"


def test_f4_row_counter_independent_of_global_steps(tmp_path):
    """Строка, впервые обновлённая на глобальном шаге 6, получает bias-коррекцию
    по своему счётчику (t=1), а не по глобальному: движение = lr*g/(|g|+eps).
    Старый код с глобальным t давал бы ~lr*0.213*g — тест ловит регрессию."""
    n, d = 8, 3
    lr = 0.1
    w0 = np.zeros((n, d))
    opt = make_opt(str(tmp_path), w0, lr=lr, k=0.0)

    for _ in range(5):                       # пять шагов только по строке 0
        do_step(opt, [0], np.ones((1, d)))
    do_step(opt, [1], np.ones((1, d)))       # строка 1 — впервые, глобальный шаг 6
    opt.shutdown()

    w, _, _, t = read_state(opt)
    expected = lr * 1.0 / (1.0 + EPS)
    assert t[0] == 5 and t[1] == 1 and t[2:].sum() == 0
    assert np.allclose(w[1], -expected, atol=1e-7), \
        f"первый шаг свежей строки: {w[1]}, ожидалось {-expected}"


def test_f4_full_adam_reference_euclidean(tmp_path):
    """k=0 — чистый евклидов путь: несколько шагов с дублями, weight_decay,
    случайными градиентами — полное совпадение с эталоном."""
    n, d = 100, 8
    rng = np.random.default_rng(7)
    w0 = rng.normal(0, 0.1, (n, d))
    lr, wd = 0.05, 0.01
    opt = make_opt(str(tmp_path), w0, lr=lr, k=0.0, weight_decay=wd)

    ref = RefAdam(w0, lr=lr, k=0.0, wd=wd)
    torch.manual_seed(7)
    for _ in range(6):
        rows = torch.randint(0, n, (37,)).numpy()
        g = torch.randn(len(rows), d).numpy()
        do_step(opt, rows, g)
        ref.step(rows, g)

    opt.shutdown()
    assert_matches_reference(opt, ref)


# ───────────────────────── ретракция и F7 ─────────────────────────

def test_retraction_keeps_norm_in_ball(tmp_path):
    """Ретракция держит норму каждой строки ≤ sqrt((1-1e-5)/k) после каждого шага."""
    n, d = 6, 4
    k = 1.0
    max_radius = np.sqrt((1.0 - 1e-5) / k)
    w0 = np.full((n, d), 0.45)  # норма 0.9 — внутри шара
    opt = make_opt(str(tmp_path), w0, lr=10.0, k=k)

    ref = RefAdam(w0, lr=10.0, k=k)
    for _ in range(8):
        g = np.full((n, d), 100.0)
        do_step(opt, np.arange(n), g)
        ref.step(np.arange(n), g)
        w, *_ = read_state(opt)  # read_state ждёт дрен — тот же приём, что step
        norms = np.linalg.norm(w, axis=1)
        assert np.all(norms <= max_radius + 1e-6), \
            f"норма вышла за шар: {norms} > {max_radius}"

    opt.shutdown()
    assert_matches_reference(opt, ref, atol=1e-4)


def test_f7_corrupted_row_is_projected_on_read(tmp_path):
    """Битая строка (норма вне шара) в файле w не расходить геометрию не должна:
    после шага строка снова внутри шара, без NaN."""
    n, d = 8, 4
    k = 1.0
    max_radius = np.sqrt((1.0 - 1e-5) / k)
    w0 = np.full((n, d), 0.1)
    opt = make_opt(str(tmp_path), w0, lr=0.01, k=k)

    # портим строку 2 напрямую в memmap (имитация битого файла)
    opt.state_files['ent']['w'][2] = np.full(d, 2.5, dtype=np.float32)  # норма 5
    opt.state_files['ent']['w'].flush()

    do_step(opt, [2], np.full((1, d), 0.1))
    opt.shutdown()

    w, m, v, t = read_state(opt)
    assert np.linalg.norm(w[2]) <= max_radius + 1e-6
    assert not np.isnan(w[2]).any() and not np.isnan(m[2]).any()


# ───────────────────────── дедупликация и формула масштабирования ─────────────────────────

def test_dedup_equals_manual_aggregation(tmp_path):
    """Батч с дублями [3,3,5] и градиентами g1,g2,g3 == батч [3,5] с g1+g2, g3.
    Проверено для k=0 и k=1 (масштабирование не ломает эквивалентность)."""
    for k in (0.0, 1.0):
        dirA = tempfile.mkdtemp()
        dirB = tempfile.mkdtemp()
        try:
            g1, g2, g3 = (np.array([[0.3, -1.2, 0.7]]),
                          np.array([[-0.4, 0.2, 1.1]]),
                          np.array([[1.5, -0.6, 0.2]]))
            w0 = np.array([[0.05, -0.02, 0.03] * 8]).reshape(8, 3) * np.arange(1, 9)[:, None]

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


def test_riemannian_scaling_formula(tmp_path):
    """Масштабирование градиента: g_riem = g*(1-k||w||^2)^2/4 == egrad/lambda^2,
    где lambda = 2/(1-k||w||^2). Проверяем по m после первого шага и по сдвигу w."""
    n, d = 4, 3
    k = 0.7
    lr = 0.05
    w0 = np.zeros((n, d))
    w0[0] = np.array([0.3, -0.4, 0.2657])  # норма ≈0.57, внутри шара
    opt = make_opt(str(tmp_path), w0, lr=lr, k=k)

    g = np.array([[1.0, -0.5, 2.0]])
    do_step(opt, [0], g)
    opt.shutdown()

    w, m, v, t = read_state(opt)
    sq = float(np.sum(w0[0] ** 2))          # 0.36, внутри шара — проекция не меняет
    rescale = (1.0 - k * sq) ** 2 / 4.0
    lam_sq = 4.0 / (1.0 - k * sq) ** 2      # λ² для шара Пуанкаре с кривизной k
    g_riem_formula = g[0] * rescale
    g_riem_via_lambda = g[0] / lam_sq
    assert np.allclose(g_riem_formula, g_riem_via_lambda, rtol=0, atol=0), \
        "формулы (1-k||w||^2)^2/4 и egrad/λ^2 обязаны совпадать"

    assert np.allclose(m[0], (1 - BETA1) * g_riem_formula, atol=1e-7)
    expected_w = w0[0] - lr * g_riem_formula / (np.abs(g_riem_formula) + EPS)
    assert np.allclose(w[0], expected_w, atol=1e-6)


# ───────────────────────── F5: сортировка индексов ─────────────────────────

def test_f5_indices_sorted_before_write(tmp_path):
    n, d = 16, 3
    opt = make_opt(str(tmp_path), np.zeros((n, d)))
    captured = []
    orig = opt.writer._apply_write

    def capturing(name, indices, w_np, m_np, v_np, t_np=None):
        captured.append(np.asarray(indices).copy())
        return orig(name, indices, w_np, m_np, v_np, t_np)

    opt.writer._apply_write = capturing
    do_step(opt, [9, 2, 7, 4, 0], np.ones((5, d)))   # без дублей, несортировано
    do_step(opt, [5, 1, 5, 3], np.ones((4, d)))      # с дублями
    opt.shutdown()

    assert len(captured) == 2
    for batch in captured:
        assert np.all(np.diff(batch) > 0), f"индексы не отсортированы: {batch}"


def test_f5_apply_write_maps_sorted_correctly(tmp_path):
    """Прямой вызов _apply_write с несортированными индексами пишет строки
    в правильные места (внутренняя сортировка не путает данные)."""
    n, d = 10, 3
    opt = make_opt(str(tmp_path), np.zeros((n, d)))
    opt.writer.wait_all_flushed()
    idx = np.array([7, 2])
    w_rows = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32)
    m_rows = np.zeros_like(w_rows)
    v_rows = np.zeros_like(w_rows)
    t_rows = np.array([3, 4], dtype=np.int32)
    opt.writer._apply_write('ent', idx, w_rows, m_rows, v_rows, t_rows)

    w, m, v, t = read_state(opt)
    assert np.allclose(w[7], 1.0) and np.allclose(w[2], 2.0)
    assert t[7] == 3 and t[2] == 4
    opt.shutdown()


# ───────────────────────── F6: честный shutdown ─────────────────────────

def test_f6_shutdown_raises_when_cannot_finish(tmp_path):
    """Писатель стоит, очередь не опустела за таймаут → shutdown честно бросает
    ошибку вместо «всё записано»; после открытия шлюза данные дописываются."""
    n, d = 40, 3
    w0 = np.zeros((n, d))
    opt = make_opt(str(tmp_path), w0, lr=0.05, k=0.0)
    open_gate = gate_writer(opt)

    ref = RefAdam(w0, lr=0.05, k=0.0)
    do_step(opt, np.arange(0, 10), np.ones((10, d)))    # висит в полёте
    ref.step(np.arange(0, 10), np.ones((10, d)))
    do_step(opt, np.arange(10, 20), np.ones((10, d)))   # стоит в очереди
    ref.step(np.arange(10, 20), np.ones((10, d)))

    with pytest.raises(RuntimeError, match="НЕ дописаны"):
        opt.writer.shutdown(timeout=0.3)

    # прибираемся вручную: открываем шлюз, ждём дрен, останавливаем поток
    open_gate()
    opt.writer.queue.join()
    opt.writer.queue.put(opt.writer._stop_signal)
    opt.writer.worker_thread.join(timeout=10)

    assert_matches_reference(opt, ref, rows=np.arange(20))


# ───────────────────────── F8: чекпоинты ─────────────────────────

def test_f8_checkpoint_save_load_and_state_dict(tmp_path):
    n, d = 24, 3
    rng = np.random.default_rng(11)
    w0 = rng.normal(0, 0.1, (n, d))
    opt = make_opt(str(tmp_path), w0, lr=0.05, k=0.0)
    rows = [2, 5, 11]
    for g_val in (1.0, -0.7):
        do_step(opt, rows, np.full((len(rows), d), g_val))

    ckpt = opt.save_checkpoint(str(tmp_path / 'ckpt.json'))
    assert os.path.exists(ckpt)
    with open(ckpt) as f:
        manifest = json.load(f)
    assert manifest['format'] == 'DSA_checkpoint'
    assert manifest['params']['ent']['shape'] == [n, d]
    opt.shutdown()

    w1, m1, v1, t1 = read_state(opt)

    # state_dict/load_state_dict — честные NotImplementedError, не мусор
    with pytest.raises(NotImplementedError):
        opt.state_dict()
    with pytest.raises(NotImplementedError):
        opt.load_state_dict({})

    # восстановление: resume=True + load_checkpoint
    opt2 = make_opt(str(tmp_path), rng.normal(0, 9.0, (n, d)),
                    lr=0.05, k=0.0, resume=True)
    opt2.load_checkpoint(ckpt)
    w2, m2, v2, t2 = read_state(opt2)
    assert np.array_equal(w1, w2) and np.array_equal(m1, m2)
    assert np.array_equal(v1, v2) and np.array_equal(t1, t2)

    # продолжение обучения с чекпоинта сходится с эталоном
    ref = RefAdam.from_state(w1, m1, v1, t1, lr=0.05, k=0.0)
    do_step(opt2, rows, np.full((len(rows), d), 3.0))
    ref.step(rows, np.full((len(rows), d), 3.0))
    opt2.shutdown()
    assert_matches_reference(opt2, ref, rows=rows)


def test_f8_load_checkpoint_foreign_paths_rejected(tmp_path):
    n, d = 10, 2
    opt = make_opt(str(tmp_path), np.zeros((n, d)))
    opt.shutdown()
    foreign = {
        'format': 'DSA_checkpoint',
        'params': {'ent': {'shape': [n, d],
                           'files': {'w': '/nonexistent/x_w.dat',
                                     'm': '/nonexistent/x_m.dat',
                                     'v': '/nonexistent/x_v.dat',
                                     't': '/nonexistent/x_t.dat'}}},
    }
    bad = tmp_path / 'foreign.json'
    bad.write_text(json.dumps(foreign))
    with pytest.raises(RuntimeError, match="скопируйте файлы"):
        opt.load_checkpoint(str(bad))


# ───────────────────────── совместимость публичного API ─────────────────────────

def test_public_api_compat(tmp_path):
    """Поверхность, которую используют внешние интеграции:
    конструктор (params dict, lr, k, disk_dir), state_files, writer._lock,
    writer.queue.join(), param_groups, step(updates=...), shutdown()."""
    n, d = 10, 2
    table = torch.randn(n, d)
    opt = DiskSparseRiemannianAdam(
        params={'ent': table}, lr=0.01, k=0.5, disk_dir=str(tmp_path))
    assert hasattr(opt, 'state_files') and 'ent' in opt.state_files
    assert {'w', 'm', 'v'} <= set(opt.state_files['ent'])
    assert hasattr(opt.writer, '_lock') and hasattr(opt.writer, 'queue')
    opt.writer.queue.join()  # wait_idle() обёртки

    opt.param_groups[0]['lr'] *= 0.5
    assert opt.param_groups[0]['lr'] == 0.005

    idx = torch.tensor([1, 2])
    opt.step(updates={'ent': (idx, torch.ones(2, d))})
    opt.zero_grad()
    opt.shutdown()

    # список параметров тоже поддерживается (как в исходном API)
    opt2 = DiskSparseRiemannianAdam([torch.randn(6, 3)], lr=0.01, k=0.0,
                                    disk_dir=str(tmp_path / 'b'))
    opt2.step(updates={'param_0': (torch.tensor([0]), torch.ones(1, 3))})
    opt2.shutdown()


# ───────────────────────── R1: NaN/inf строки в файле состояния ─────────────────────────

def test_r1_nan_row_is_sanitized_on_step(tmp_path, caplog):
    """Битая строка w (NaN/inf) не проходит проекцию молча (torch.where при
    NaN даёт False) и расползлась бы в m/v и обучение. После одного шага
    в w/m/v нет NaN/inf, строка внутри шара; в логе warning, счётчик растёт."""
    import logging
    n, d, k = 8, 4, 1.0
    max_radius = np.sqrt((1.0 - 1e-5) / k)
    w0 = np.full((n, d), 0.1)
    opt = make_opt(str(tmp_path), w0, lr=0.01, k=k)
    opt.shutdown()

    # портим файл w напрямую (имитация битого файла): строка 2 = NaN, 3 = inf
    with open(os.path.join(str(tmp_path), 'ent_w.dat'), 'r+b') as f:
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
        assert np.isfinite(state[[2, 3]]).all(), "NaN/inf выжил после шага"
    assert np.linalg.norm(w[2]) <= max_radius + 1e-6
    assert np.linalg.norm(w[3]) <= max_radius + 1e-6
    assert t[2] == 1 and t[3] == 1
    assert opt2._nan_rows_replaced == 2, "счётчик не-конечных строк не вырос"
    assert any("NaN" in r.message for r in caplog.records), "нет warning в логе"


# ───────────────────────── R2: маркер переинициализации ─────────────────────────

def test_r2_reinit_marker_blocks_resume(tmp_path):
    """Смешанное состояние после смерти процесса посреди переинициализации
    (маркер 'reinit_in_progress' в meta + частично перезаписанные файлы,
    воспроизведённые руками): resume=True обязан дать явную ошибку, а не
    принять его молча."""
    n, d = 8, 3
    opt = make_opt(str(tmp_path), np.zeros((n, d)))
    do_step(opt, [1, 2], np.ones((2, d)))
    opt.shutdown()

    meta_path = os.path.join(str(tmp_path), 'ent_meta.json')
    with open(meta_path) as f:
        meta = json.load(f)
    meta['reinit_in_progress'] = True
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f)
    with open(os.path.join(str(tmp_path), 'ent_w.dat'), 'r+b') as f:
        f.seek(0)
        f.write(np.full((n, d), 0.5, dtype=np.float32).tobytes())

    with pytest.raises(RuntimeError, match="reinit_in_progress"):
        make_opt(str(tmp_path), np.zeros((n, d)), resume=True)


def test_r2_reinit_marker_cleared_by_full_reinit(tmp_path):
    """resume=False — документированный путь восстановления: полная
    переинициализация пересоздаёт файлы и снимает маркер; после неё
    resume=True снова работает."""
    n, d = 8, 3
    opt = make_opt(str(tmp_path), np.zeros((n, d)))
    opt.shutdown()
    meta_path = os.path.join(str(tmp_path), 'ent_meta.json')
    with open(meta_path) as f:
        meta = json.load(f)
    meta['reinit_in_progress'] = True
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f)

    fresh = rng_vals(13, n, d)
    opt2 = make_opt(str(tmp_path), fresh, lr=0.05, k=0.0)  # resume=False
    opt2.shutdown()
    with open(meta_path) as f:
        assert 'reinit_in_progress' not in json.load(f), "маркер не снят"
    w, m, v, t = read_state(opt2)
    assert np.allclose(w, fresh, atol=1e-7)
    assert np.all(m == 0) and np.all(v == 0) and np.all(t == 0)

    opt3 = make_opt(str(tmp_path), fresh, lr=0.05, k=0.0, resume=True)
    opt3.shutdown()


# ───────────────────────── R3: неизвестный ключ updates ─────────────────────────

def test_r3_unknown_updates_key_warns_and_counts(tmp_path, caplog):
    """Ключ-опечатка в updates не отбрасывается молча: warning в логе,
    счётчик растёт, чужие строки не изменяются."""
    import logging
    n, d = 6, 3
    opt = make_opt(str(tmp_path), np.full((n, d), 0.1))
    before = opt._unknown_key_warnings
    with caplog.at_level(logging.WARNING, logger="DSA"):
        do_step(opt, [1], np.ones((1, d)))                       # валидный ключ
        opt.step(updates={'entt': (torch.tensor([2]), torch.ones(1, d))})
    opt.shutdown()

    assert opt._unknown_key_warnings == before + 1, "счётчик не вырос"
    assert any("entt" in r.message for r in caplog.records), "нет warning в логе"
    w, *_ = read_state(opt)
    assert np.allclose(w[2], 0.1, atol=1e-7), "строка 2 не должна была измениться"


def test_r3_unknown_key_raises_in_debug_mode(tmp_path, monkeypatch):
    """В отладочном режиме (dsa.optimizer.DEBUG, env DSA_DEBUG=1) неизвестный
    ключ updates — исключение KeyError, а не warning."""
    import dsa.optimizer as dsa_opt
    n, d = 6, 3
    opt = make_opt(str(tmp_path), np.zeros((n, d)))
    monkeypatch.setattr(dsa_opt, 'DEBUG', True)
    with pytest.raises(KeyError, match="entt"):
        opt.step(updates={'entt': (torch.tensor([2]), torch.ones(1, d))})
    assert opt._unknown_key_warnings == 0  # до warning дело не дошло
    opt.shutdown()

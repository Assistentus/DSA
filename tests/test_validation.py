"""Validation and corruption paths that the behavioural suite does not hit."""

import json
import os

import numpy as np
import pytest
import torch

from dsa.optimizer import DiskSparseRiemannianAdam


def _opt(tmp_path, **kwargs):
    kwargs.setdefault("disk_dir", str(tmp_path))
    return DiskSparseRiemannianAdam({"e": torch.zeros(3, 2)}, **kwargs)


@pytest.mark.parametrize("kwargs,exc", [
    ({"lr": -1}, ValueError),
    ({"beta1": 1.0}, ValueError),
    ({"beta2": -0.1}, ValueError),
    ({"eps": 0.0}, ValueError),
    ({"weight_decay": -1.0}, ValueError),
    ({"max_queue_size": 0}, ValueError),
    ({"put_timeout": -1.0}, ValueError),
    ({"init_max_norm_fraction": 0.0}, ValueError),
    ({"init_max_norm_fraction": 1.1}, ValueError),
    ({"weight_decay_mode": "adamw"}, ValueError),
    ({"k": -1.0}, ValueError),
    ({"k": float("nan")}, ValueError),
    ({"k": float("inf")}, ValueError),
    ({"resume": "yes"}, TypeError),
])
def test_constructor_rejects_invalid_arguments(tmp_path, kwargs, exc):
    with pytest.raises(exc):
        _opt(tmp_path, **kwargs)


def test_constructor_rejects_bad_params(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        DiskSparseRiemannianAdam({}, disk_dir=str(tmp_path))
    with pytest.raises(ValueError, match="2-D"):
        DiskSparseRiemannianAdam({"e": torch.zeros(3)}, disk_dir=str(tmp_path))
    with pytest.raises(TypeError, match="floating"):
        DiskSparseRiemannianAdam({"e": torch.zeros(3, 2, dtype=torch.int64)},
                                 disk_dir=str(tmp_path))
    with pytest.raises(TypeError, match="Tensor"):
        DiskSparseRiemannianAdam({"e": np.zeros((3, 2))}, disk_dir=str(tmp_path))
    with pytest.raises(ValueError, match="path separators"):
        DiskSparseRiemannianAdam({"a/b": torch.zeros(3, 2)},
                                 disk_dir=str(tmp_path))


def test_nonfinite_initial_weights_rejected(tmp_path):
    with pytest.raises(ValueError, match="NaN/inf"):
        DiskSparseRiemannianAdam({"e": torch.full((3, 2), float("nan"))},
                                 disk_dir=str(tmp_path))


def test_resume_true_on_empty_directory_raises(tmp_path):
    with pytest.raises(RuntimeError, match="нет состояния"):
        _opt(tmp_path, resume=True)


def _valid_state(tmp_path):
    opt = _opt(tmp_path)
    opt.shutdown()
    return os.path.join(str(tmp_path), "e_meta.json")


@pytest.mark.parametrize("mutator,match", [
    (lambda m: m.__setitem__("format_version", 99), "версия формата"),
    (lambda m: m.__setitem__("algorithm", "other"), "алгоритм"),
    (lambda m: m.__setitem__("k", 0.5), "кривизна"),
    (lambda m: m.__setitem__("dtype", "float64"), "типы данных"),
])
def test_corrupt_meta_is_rejected(tmp_path, mutator, match):
    meta_path = _valid_state(tmp_path)
    meta = json.load(open(meta_path))
    mutator(meta)
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    with pytest.raises(RuntimeError, match=match):
        _opt(tmp_path, resume=True)


def test_unreadable_meta_is_rejected(tmp_path):
    meta_path = _valid_state(tmp_path)
    with open(meta_path, "w") as f:
        f.write("{not json")
    with pytest.raises(RuntimeError, match="не читается"):
        _opt(tmp_path, resume=True)


@pytest.mark.parametrize("mutator,match", [
    (lambda m: m.__setitem__("format_version", 99), "глобальный meta"),
    (lambda m: m.__setitem__("global_step", "many"), "global_step"),
])
def test_corrupt_global_meta_is_rejected(tmp_path, mutator, match):
    _valid_state(tmp_path)
    path = os.path.join(str(tmp_path), "dsa_global_meta.json")
    meta = json.load(open(path))
    mutator(meta)
    with open(path, "w") as f:
        json.dump(meta, f)
    with pytest.raises((RuntimeError, ValueError), match=match):
        _opt(tmp_path, resume=True)


def test_sparse_gradient_is_rejected(tmp_path):
    opt = _opt(tmp_path)
    idx = torch.tensor([0])
    sparse = torch.sparse_coo_tensor(
        torch.tensor([[0], [0]]), torch.tensor([1.0]), size=(1, 2))
    with pytest.raises(TypeError, match="разреженный"):
        opt.step(updates={"e": (idx, sparse)})
    opt.shutdown()


def test_updates_contract_and_numpy_inputs(tmp_path):
    opt = _opt(tmp_path)
    with pytest.raises(TypeError, match="dict"):
        opt.step(updates=[])
    with pytest.raises(TypeError, match="indices, grads"):
        opt.step(updates={"e": 5})
    with pytest.raises(ValueError, match="1-D"):
        opt.step(updates={"e": (np.array([[0]]),
                               np.ones((1, 2), dtype=np.float32))})
    with pytest.raises(TypeError, match="целочисленными"):
        opt.step(updates={"e": (np.array([0.0]),
                               np.ones((1, 2), dtype=np.float32))})
    # numpy indices and gradients are accepted
    opt.step(updates={"e": (np.array([0]), np.ones((1, 2), dtype=np.float32))})
    opt.step(updates={"e": (torch.tensor([1]), np.ones((1, 2), dtype=np.float32))})
    w = np.array(opt.state_files["e"]["w"])
    assert np.isfinite(w).all()
    opt.shutdown()


def test_zero_initial_weights_hyperbolic(tmp_path):
    opt = DiskSparseRiemannianAdam(
        {"e": torch.zeros(3, 2)}, k=1.0, disk_dir=str(tmp_path))
    assert np.all(np.array(opt.state_files["e"]["w"]) == 0)
    opt.shutdown()


def test_writer_timeout_and_shutdown_validation(tmp_path):
    import threading
    opt = _opt(tmp_path)
    gate = threading.Event()
    orig = opt.writer._apply_write

    def gated(*args, **kwargs):
        gate.wait()
        return orig(*args, **kwargs)

    opt.writer._apply_write = gated
    opt.step(updates={"e": (torch.tensor([0]), torch.ones(1, 2))})
    with pytest.raises(RuntimeError, match="очередь не опустела"):
        opt.writer.wait_all_flushed(timeout=0.2)
    with pytest.raises(TypeError, match="timeout"):
        opt.writer.shutdown(timeout=None)
    with pytest.raises(ValueError, match="timeout"):
        opt.writer.shutdown(timeout=-1)
    gate.set()
    opt.shutdown()


def test_malformed_checkpoint_manifest_is_rejected(tmp_path):
    opt = _opt(tmp_path)
    ckpt = tmp_path / "ckpt"
    opt.export_checkpoint(str(ckpt))
    opt.shutdown()

    manifest_path = ckpt / "checkpoint.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["params"] = []
    manifest_path.write_text(json.dumps(manifest))
    opt2 = _opt(tmp_path / "new")
    with pytest.raises(RuntimeError, match="params"):
        opt2.import_checkpoint(str(ckpt))
    opt2.shutdown()

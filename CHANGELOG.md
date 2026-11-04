# Changelog

## 0.3.0 — 2026-11-04

Correctness release for hyperbolic training. The headline change is that
`k > 0` is now a real Riemannian Adam (exponential map, tangent-space first
moment, parallel transport) instead of Euclidean Adam with gradient scaling
and a clip. Several APIs change incompatibly; see **Breaking changes**.

### Breaking changes

- **`resume` defaults to `None`.** An existing state directory is never
  silently reused or silently reinitialized: `None` raises, `True` continues,
  `False` reinitializes from `params` with a warning.
- **Checkpoints are directories.** `save_checkpoint`/`load_checkpoint` are
  directory-only aliases; the primary API is
  `export_checkpoint(dir)`/`import_checkpoint(dir)`. 0.2.0 manifests with
  absolute paths are rejected.
- **State format v2.** 0.2.0 (`format_version=1`) and 0.1.0 (no meta, no `t`)
  states are migrated on `resume=True`: weights are preserved, moments and
  per-row counters are reset because the old math is incompatible with
  tangent-space moments.
- **`weight_decay_mode` added.** `"l2"` (default) is part of the Euclidean
  gradient before the Riemannian conversion (classic Adam L2 / Riemannian
  gradient of `f + wd/2 ||w||^2`); `"decoupled"` is a tangent-space term and
  equals AdamW when `k = 0`. The old coordinate shrink is gone.
- **Source tensors are released by default** (`retain_params=False`):
  `param_groups[...]["params"]` holds zero-length CPU placeholders after
  initialization, so a full embedding table is not kept alive in VRAM. Use
  `read_rows()` for weights, or `retain_params=True` for the old behavior.
- **Global step is persisted** in `dsa_global_meta.json` and in checkpoints;
  `resume=True` restores it.
- **Stricter validation:** negative/out-of-range indices, wrong gradient
  shapes, non-finite gradients, invalid `k`, and unknown update keys now
  produce explicit errors or warnings.

### Fixed — audit findings

- **C1 — legacy 0.1.0 could not be reinitialized.** `resume=False` now
  reinitializes even a partial/legacy directory (missing `t`/meta); weights
  from `params` are written, moments/counters zeroed. `resume=True` migrates
  a 0.1.0 directory (weights preserved, `m/v/t` reset, meta v2 written) and a
  0.2.0 directory (`format_version=1`, same reset rule).
- **C2 — checkpoints were not portable.** `export_checkpoint(dir)` copies
  `w/m/v/t` into a directory under relative names and writes
  `checkpoint.json` with sizes and SHA-256 checksums. `import_checkpoint(dir)`
  verifies shape, size and hash, stages all files, writes a reinitialization
  marker, then replaces the live state atomically per file. A moved/copied
  checkpoint imports successfully; a corrupted one is rejected before the
  live state changes.
- **C3 — the optimizer retained the full source tensor.** With
  `retain_params=False` (default), source tensors are replaced by zero-length
  placeholders after the on-disk state is written; a weak-reference test
  confirms the tensor can be collected. `retain_params=True` restores the old
  PyTorch-style retention.
- **C4 — weight decay was coordinate shrink.** `weight_decay_mode="l2"` now
  adds `wd * w` to the Euclidean gradient (Riemannian gradient of the L2
  penalty), and `"decoupled"` adds `wd * w` to the tangent direction. The
  Euclidean limit matches Adam L2 and AdamW respectively, and the hyperbolic
  L2 penalty cancels exactly with `g = -wd * w`.
- **C5 — two packaging metadata sources.** `setup.py` is removed;
  `pyproject.toml` is the single source (`torch>=2.0.0`, `numpy>=1.21.0`),
  matching the README badge. `dsa.__version__` is `0.3.0`.
- **A1 — a failed `shutdown()` could report success on retry.** The writer is
  now a `RUNNING → DRAINING → STOPPED` / `FAILED` state machine. A timeout
  leaves the worker alive and the queue unfinished; the next call retries and
  only a clean drain/join/flush returns success. Persistent write errors are
  re-raised, never hidden.
- **A2 — `global_step` was lost on load.** It is stored in
  `dsa_global_meta.json`, exported in checkpoints, restored on `resume=True`
  and `import_checkpoint()`.
- **A3 — negative indices silently updated the last row.** Indices are
  validated: 1-D integer dtype, `0 <= i < N`; negative, out-of-range, float
  and 2-D indices raise before any state is touched.
- **A4 — a NaN row left a corrupted counter/moment.** A non-finite (or
  negative-variance) row is fully reset (`w=m=v=0, t=0`), so the current step
  is a genuine first Adam step, not a continuation of a broken counter.
- **A9 — gradient shape/finiteness was unchecked.** Wrong shapes, non-float
  dtypes and NaN/inf gradients raise `ValueError` before moments are updated.
- **H1 — initialization and resume could leave rows outside the ball.**
  Fresh hyperbolic weights are globally scaled so the largest row norm is at
  most `init_max_norm_fraction` (default 0.5) of the radius; rows outside are
  still projected. On resume, finite out-of-ball rows are projected and
  counted; `read_rows()` always projects. `export_checkpoint()` immediately
  after initialization is valid.
- **H2 — the step was not Riemannian Adam.** The step now uses
  `grad_R = lambda^-2 g_E`, tangent-space moments, the exponential map
  `exp_w(-lr * m_hat/(sqrt(v_hat)+eps))` and parallel transport of `m` (and
  of `sqrt(v)`, squared back to keep the diagonal second moment
  non-negative). Active rows are computed in `float64` (storage stays
  `float32`) to avoid catastrophic cancellation in the gyration formula near
  the boundary. Tests compare against an independent float64 reference and
  check the defining geometric properties (exp map distance, transport
  identity/isometry/round-trip, log-exp inverse).
- **H3 — changing `k` left untouched rows outside the new ball.**
  `set_curvature(k)` and `step(current_k=...)` reproject the whole table;
  `reset_moments=True` optionally restarts the adaptive state.
- **H4 — `k` was not validated.** `k = NaN` no longer falls through to the
  Euclidean path, `k = inf` no longer collapses the ball, and `k < 0` raises
  at construction instead of a later `math domain error`.
- **F1–F8, R1–R3** from 0.2.0 remain covered: lossless queue fallback,
  in-flight read consistency, strict state lifecycle, per-row bias
  correction, sorted indices, honest shutdown, projection on read, NaN-row
  handling, reinitialization marker, and unknown-key diagnostics.

### Tests

The suite now has explicit acceptance tests for C1–C5, A1–A4, A9, H1–H4,
F1–F8 and R1–R3, plus geometric property tests and an independent float64
reference. Run with a loaded `.env` if your environment requires it:

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q
```

### Known limitations

- The Poincaré chart is numerically singular at the boundary. Points are kept
  at least `BALL_EPS = 1e-5` inside, but very long training with a large `lr`
  can reach that margin; the optimizer stays finite and projects, while the
  chart itself is ill-conditioned there.
- `v` is a per-coordinate diagonal moment. Transporting it directly as a
  tangent vector can produce negative components and NaNs, so its
  non-negative square root is transported and squared back. This is the
  positivity-preserving version of the paper's operation, not a
  component-wise `sqrt` of a rotated vector.
- CUDA-specific memory behavior is tested only through object lifetime on CPU
  in CI; no GPU is required for the test suite.

## 0.2.0 — 2026-11-04

Correctness release. The public API is preserved: the constructor signature
(with new optional `resume` and `put_timeout` arguments), `step(updates,
current_k=None)`, `shutdown()` and the `state_files` attribute are compatible
with 0.1.0. Two files per parameter were added (`<name>_t.dat`,
`<name>_meta.json`); 0.1.0 state directories are not compatible with 0.2.0
and must be reinitialized (the default `resume=False` does exactly that).

### Fixed

- **F1 — a full write queue silently dropped batches**
  (`AsyncDiskWriter.put_update`). `put_nowait` + "sacrifice the batch" on
  `queue.Full` discarded both the Adam moments and the row updates of whole
  batches. `put_update` now blocks for `put_timeout` seconds (new optional
  constructor argument, default 5.0) and, if the queue is still full, writes
  the batch synchronously into the memmaps under the writer lock. Writes are
  never dropped; a `_sync_fallbacks` counter is kept for diagnostics.

- **F2 — race between `step()` and the background writer.** `step()` read
  `w/m/v` without knowing whether the writer had finished writing the same
  rows: a step could compute from stale state and overwrite newer values
  with older ones. The writer now tracks in-flight rows per parameter under
  its lock (`intersects_inflight`), marks rows in flight before enqueueing
  and removes them before `task_done`; `step()` drains the queue
  (`wait_all_flushed`) only when the batch actually overlaps in-flight rows,
  so the asynchronous path is preserved in the common case.

- **F3 — silent reuse of leftover state files.** The old init picked up any
  existing `w/m/v` files in `disk_dir` silently (weights from `params`
  ignored, stale/foreign moments kept). Each parameter now has a
  `<name>_meta.json` (format version, shape, dtypes, file list; written
  atomically). `resume=True` continues the existing state as is;
  `resume=False` (the new default) fully reinitializes it with a warning;
  shape/dtype mismatch, a partial file set, missing meta, and truncated or
  oversized data files all raise explicit `RuntimeError`s with recovery
  instructions. Before `np.memmap` opens the files, their actual sizes are
  verified against the meta shape — memmap in `r+` mode silently
  zero-extends truncated files, which would have turned lost rows into
  quiet zeros.

- **F4 — Adam bias correction used the global step counter.** A row updated
  for the first time at global step N received almost no bias correction
  (its `m_hat` was understated by orders of magnitude), and the first step
  of a fresh row was not `lr*sign(g)`. A fourth memmap `<name>_t.dat`
  (int32, N×4 bytes, per-row counters) now drives the bias correction; the
  counter travels through the same write queue as `w/m/v`, so the on-disk
  state stays consistent, and the first step of a fresh row analytically
  gives `w -= lr*g/(|g|+eps) ≈ lr*sign(g)`.

- **F5 — unsorted indices caused random-access I/O.** `step()` now sorts the
  deduplicated batch indices (`torch.argsort`) so both reads and writes are
  sequential; `_apply_write` sorts defensively on its side as well, so
  direct calls cannot write rows to wrong places.

- **F6 — dishonest shutdown.** `shutdown()` joined the worker for 5 seconds
  and then reported "all data written" even on timeout (msync may not have
  finished), and flush errors were swallowed by a bare `except: pass`.
  Shutdown now drains the queue with a bounded wait (a stuck disk yields an
  error, not an infinite hang), joins the worker with a sufficient timeout
  (default 300 s), accumulates write/flush errors of the whole session and
  raises `RuntimeError` if data was not fully written. `put_update` after
  shutdown raises instead of silently doing nothing.

- **F7 — corrupted rows were read without projection.** Rows of `w` read
  from disk are now projected into the Poincaré ball (radius
  `sqrt((1-1e-5)/k)`) before use, so a corrupted file cannot break the
  geometry; retraction after the step is unchanged.

- **F8 — `state_dict()` silently returned an almost-empty state.** The
  inherited implementation returned a dict with a single `step` entry,
  giving the false impression of a serialized optimizer. Replaced with an
  explicit checkpoint API: `save_checkpoint(path)` (queue drain + flush of
  `w/m/v/t` + atomic manifest) and `load_checkpoint(path)` (verifies
  format, shapes, paths and file sizes, reopens the memmaps; a checkpoint
  from a foreign directory is an explicit error).
  `state_dict()`/`load_state_dict()` now raise `NotImplementedError` with
  instructions.

### Added

- **R1 — non-finite rows are sanitized on read.** A `w` row with NaN/inf
  from a corrupted file previously passed the projection silently
  (`torch.where(norm > max)` evaluates False for NaN) and poisoned `m/v`
  and the training. Non-finite rows (in `w`, `m` or `v`) are replaced via
  `nan_to_num` on read — the row restarts from the origin and the ball
  projection restores the geometry — with a warning and a
  `_nan_rows_replaced` counter.

- **R2 — reinitialization is marked in meta.** A hard kill between the
  per-file writes of a reinitialization and the meta write used to leave a
  mixed state (part new files, part old) that `resume=True` accepted
  silently. Meta now carries a `reinit_in_progress` marker while the files
  are being rewritten: `resume=True` on a marked state raises an explicit
  error; `resume=False` performs the documented full-reinitialization
  recovery and clears the marker.

- **R3 — unknown `updates` keys are no longer dropped silently.**
  `step(updates={'typo': ...})` used to discard the update without any
  message — lost table updates with no trace. An unknown key now logs a
  warning and increments `_unknown_key_warnings`; in debug mode
  (`dsa.optimizer.DEBUG = True` or environment variable `DSA_DEBUG=1`) it
  raises `KeyError` instead.

### Tests

- New `tests/test_fixes.py` regression suite (34 test items) checking every
  fix above against a float64 in-memory reference model: losses on a slow
  disk, stale reads, resume/reinit/size checks, analytic bias correction,
  sorted indices, honest shutdown, checkpoint round-trips, retraction and
  the Riemannian scaling formula, plus the R1–R3 behaviors. The existing
  `test_optimizer.py` was hardened with queue drains so manual memmap reads
  cannot race the background writer. Full suite: 36 tests.

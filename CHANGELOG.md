# Changelog

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

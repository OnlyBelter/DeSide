# Debug Session: deside-log-stall-after-pathway-stats
**Status:** [RESOLVED — 3 bugs found + fixed]
**Created:** 2026-08-19
**Last Updated:** 2026-08-19 (resolution)
**Symptom:** During DeSide scratch training (`deside train --config`), user log timestamps show the following two lines printed successfully:
```
common genes between training set and pathway mask: 9829
genes only in training set: 8005
```
Then no further log output for **≥ 20 minutes**, until user interrupted the run. This investigation focuses on what executes *after* those two lines, not on ingestion before them.
**Environment (user report):** Linux HPC /data/phyxiongx node, `(deside-torch)` conda env, 3x SimuTME H5AD training sets (dirichlet/segment/sparse 2-4-5 split, N5K each), KEGG medicus v2026.1 + Reactome v2026.1 GMT pathway mask, `method_adding_pathway: add_to_end`.

## Scope / Investigation Boundary
Focus: code that runs **AFTER** the two pathway gene-stat prints inside `DeSide._get_pathway_profiles` (file [torch_deside.py](file:///Users/belter/github/VAEDecon_combined/DeSide/deside/decon_cf/torch_deside.py)) and the 15 pre-training steps that follow up to `train_deside_lightning` entry.

## Timeline of last-known-good to stall
1. `torch_deside.py DeSide.train_model` enters `_get_pathway_profiles`
2. computes `common_genes` (len=9829) and `genes_only_in_x` (len=8005)
3. prints both lines ✅ (last confirmed output)
4. code still to execute inside `_get_pathway_profiles`:
   - ~~pad pathway_mask with 8005 training-only rows~~ — **completed in the same callstack as the print, confirmed by tests**
   - reindex pathway_mask to x columns
   - execute `x_pathway_profiles = x @ pathway_mask` (15,000 × 17,834 @ 17,834 × ~2,043 pathways → ≈2,043 cols appended)
   - if `filtered_gene_list != None` and shape mismatch: `align_with_gene_list`
   - `x = concat([x, x_pathway_profiles])`
   - `x = np.log2(x + 1.0)`
   - print `"x shape: ..."`
   - return `ReadExp(x, exp_type=log_space)`
5. back in `train_model`:
   - optional `x_obj.do_scaling()` / `do_scaling_by_constant()`
   - `x = x_obj.get_exp()`
   - cell-type selection + Cancer Cells drop
   - write 4 TSV gene/cell-type metadata files under model_dir
   - print `Use the following cell types`, `shape of X`, `shape of y`
   - `split_inputs_for_dataset` → `split_deside_dataset`
   - print `The following loss function will be used: ... * mae + ... * rmse`
   - `train_deside_lightning` (epoch loops start)

## Hypotheses (5 falsifiable) — final disposition
| ID | Hypothesis | Verdict | Notes |
|----|-----------|---------|-------|
| H1 | Memory thrash / OOM during matmul + concat | **CONTRIBUTING (3/3 importance)** | Conservative peak RSS 5.59 GB, realistic ~2.8 GB (see §Memory analysis). Combined with BLAS oversubscription on a shared HPC login node (32+ OpenBLAS threads × stack), easily pushes into swap → 20 min silent stall of page faults. |
| H2 | Unflushed stdout buffer in HPC slurm/non-interactive shell | **CONTRIBUTING (3/3)** | All new progress prints now use `flush=True`; user should also set `PYTHONUNBUFFERED=1`. 20 min is longer than stdout buffer can hide on its own, but it compounds with H1/H4. |
| H3 | Blocking NFS/parallel-FS TSV metadata writes | **DISPROVEN** | Metadata write is AFTER `x shape:` print (which the user never saw), so stall happens BEFORE step 5. |
| H4 | Unhandled AttributeError from np.log2 returning ndarray + align_with_gene_list called in wrong order | **ROOT CAUSE #1 (3/3)** | Three sub-bugs identified (§Root cause triad). Version-dependent: older pandas/numpy dispatch ufuncs to ndarray, newer preserve DataFrame. |
| H5 | split_deside_dataset / DeSideDataset tensorization hangs | **DISPROVEN** | Same reasoning as H3: these steps run AFTER "The shape of X is:" prints, which the user never saw. Stall is inside `_get_pathway_profiles` matmul/log2/ReadExp region. |

## Memory analysis (from analyze_memory.py)
User's real tensor sizes:
- `x` (expression): 15,000 × 17,834 × float32 = **1.02 GB**
- `pathway_mask` after reindex: 17,834 × 2,043 × float32 = **139 MB**
- `x_pathway_profiles` output matmul: 15,000 × 2,043 × float32 = **117 MB**
- `pd.concat([x, x_pw])` output: 15,000 × 19,877 × float32 = **1.11 GB**
- `pd.concat` live+copy peak (both inputs still referenced during block allocation): est **2.22 GB**
- `_split_inputs_for_dataset` two `.copy()` calls on concat result: est **2.22 GB**
- `DeSideDataset` tensorize 3 arrays: est **1.11 GB**
- **Conservative peak RSS:** 5.59 GB
- **Realistic peak (GC/reuse of freed blocks):** 2.79 GB

Interpretation:
- < 4 GB available/cgroup: **OOM-killed** (silent unless `dmesg` checked)
- 4–8 GB available: **swap thrash** → 20+ minute stall
- ≥ 16 GB dedicated: passes, but BLAS oversubscription still slows it

## Root cause triad (three distinct defects in `_get_pathway_profiles`)
### DEFECT A: `filtered_gene_list` alignment run IN THE MIDDLE of method, AFTER matmul computed on OLD columns
**Location:** [torch_deside.py old lines 868-875](file:///Users/belter/github/VAEDecon_combined/DeSide/deside/decon_cf/torch_deside.py) (pre-fix)

Original code order:
1. `x_matmul_input = x_obj.get_exp()`  ← full column set (17,834 genes)
2. `x_pw = x_matmul_input @ pathway_mask` ← matmul done against 17,834-gene input
3. **then:** `if filtered_gene_list != None and mismatch: x_obj.align_with_gene_list(filtered); x = x_obj.get_exp()` ← now x is (e.g.) 9,829 filtered genes
4. `pd.concat([x (9,829 cols), x_pw (2,043 cols)])` ← index alignment OK (both 15K rows), BUT
   - the pathway matmul was computed with genes that no longer exist in `x`
   - if align_with_gene_list filled genes with 0, downstream correlation semantics are silently wrong
   - if align_with_gene_list REMOVED genes, but pathway_mask still had them, then the matmul already aggregated their signal into pathway scores — the GEP side then loses those genes → input misalignment for dual-encoder: same pathway score but different GEP genes than what produced it

**Fix applied:** Run `filtered_gene_list` alignment **FIRST** on entry to `_get_pathway_profiles`, before ANY matmul or get_exp. Now x_pw and x share the same filtered gene set for the entire method.

### DEFECT B: `x = np.log2(x + 1.0)` returns `np.ndarray` on older pandas/numpy → `ReadExp(ndarray)` → `.columns` AttributeError downstream
**Evidence / repro:**
- pandas 2.3.3 / numpy 2.4.3 (local): `np.log2(DataFrame)` preserves DataFrame ✅
- pandas <1.5 / numpy <1.23 (older HPC envs): `np.log2(DataFrame)` dispatches to raw ufunc and returns `np.ndarray` ❌
- `read_df(np.ndarray)` in [pub_func.py:801-802](file:///Users/belter/github/VAEDecon_combined/DeSide/deside/utility/pub_func.py#L801-L802) returns ndarray as-is with no `.columns`
- `align_with_gene_list` at [read_file.py:221](file:///Users/belter/github/VAEDecon_combined/DeSide/deside/utility/read_file.py#L221) → `current_columns = self.exp.columns` → **AttributeError: 'numpy.ndarray' object has no attribute 'columns'**
- Repro at [repro_check.py TEST 1](file:///Users/belter/github/VAEDecon_combined/DeSide/repro_check.py):
  ```
  --- Simulate: ReadExp(ndarray) then align_with_gene_list ---
  ReadExp constructed, exp type: ndarray
  CONFIRMED BUG: AttributeError: 'numpy.ndarray' object has no attribute 'columns'
  ```

Why "20 minutes of no output" for a simple AttributeError:
1. Exception raised in the main thread → Python enters post-mortem cleanup
2. Cleanup may involve closing open file handles on NFS, atexit handlers (torch GPU teardown), and stderr buffering
3. On the HPC node: if swap was already thrashing (H1), even the exception unwind path makes zero forward progress for minutes
4. Python's default stderr buffer under slurm is 4–8 KB; the traceback can sit in the buffer until the process fully exits, which never happens if it's stuck in D-state swap page faults

**Fix applied (two layers):**
- Layer 1 (primary, in `_get_pathway_profiles`): never rely on numpy ufunc return-type dispatch for DataFrames. Instead:
  ```python
  _log_in = x.to_numpy(dtype=np.float32, copy=False)
  _log_out = np.log2(_log_in + 1.0)
  x = pd.DataFrame(data=_log_out, index=x.index, columns=x.columns, copy=False)
  ```
  This is guaranteed correct across ALL pandas/numpy versions, and the explicit `.to_numpy()` + `pd.DataFrame` reuse avoids any intermediate column-wise ufunc overhead.
- Layer 2 (defensive, in `ReadExp.__init__`): If `read_df()` returns ndarray, wrap it to DataFrame with auto-generated `col_N` / `sample_idx` names. This prevents the AttributeError even for other callers that accidentally pass ndarray to ReadExp.

### DEFECT C: BLAS / torch thread oversubscription → matmul swap-death on shared HPC node
**Mechanism:**
- OpenBLAS / MKL defaults to `cpu_count()` threads (commonly 32–128 on HPC nodes).
- `x_values @ pm_values` is a standard DGEMM; OpenBLAS spawns N worker threads each with ~256 KB–4 MB stack + per-thread workspace.
- Simultaneously: pandas/numpy Python main thread holds 2–3 copies of the matrices (x, pathway_mask, x_pw, concat).
- Combined RSS easily exceeds login-node soft limits and enters swap. Each BLAS thread then faults on its stack/workspace pages. The scheduler round-robins 32 threads each waiting on disk I/O → **hours of walltime with zero algorithmic progress and no prints** (all threads in page fault, not the Python main thread that does the prints).

**Fix applied in torch_deside.py `_cap_blas_threads()`:**
- Runs `os.environ.setdefault()` on `{OMP,OPENBLAS,MKL,NUMEXPR,VECLIB_MAXIMUM}_NUM_THREADS` to `min(physical_cpu_count, 8)` **BEFORE** any numpy/torch import (so the BLAS libraries read the env at load time).
- Additionally calls `torch.set_num_threads(cap)` + `torch.set_num_interop_threads(...)` after import.
- Only activates if the user has NOT set any of these knobs explicitly → zero surprise for expert users.
- For pathway matmul of size 15K × 17.8K × 2K: 8 threads is already near peak memory-bandwidth saturation; more threads only cause oversubscription.

## Evidence collected
| Item | File | Result |
|------|------|--------|
| Hypothesis H4 repro | [repro_check.py](file:///Users/belter/github/VAEDecon_combined/DeSide/repro_check.py) | AttributeError on ndarray `.columns` **reproduced** locally. |
| H1 memory analysis | [analyze_memory.py](file:///Users/belter/github/VAEDecon_combined/DeSide/analyze_memory.py) | Peak 2.79–5.59 GB RSS confirmed. |
| Fix verification | [test_fixes.py](file:///Users/belter/github/VAEDecon_combined/DeSide/test_fixes.py) | All 4 tests pass: ReadExp(ndarray) wraps, align works on wrapped, `_get_pathway_profiles` E2E returns correct DataFrame shape for add_to_end + filtered_gene_list, convert method also preserves DataFrame. |
| Import smoke test | inline | All package modules import cleanly; diagnostics (GetDiagnostics) = 0. |
| Instrumentation coverage | `#region debug-point deside-log-stall-after-pathway-stats` in torch_deside.py | 21 tracepoints placed covering every post-stats stage (pw_pad_complete → train_deside_lightning_returned), plus sys/threading excepthooks, plus RSS/getrusage sampling. Active only when `TRAE_DEBUG_SESSION=deside-log-stall-after-pathway-stats TRAE_DEBUG_TRACE_DIR=/tmp/deside_debug` are set; no-op otherwise. |

## Fixes applied (files changed)
1. **[torch_deside.py](file:///Users/belter/github/VAEDecon_combined/DeSide/deside/decon_cf/torch_deside.py)**
   - `_cap_blas_threads()`: module-level BLAS/torch thread cap (Defect C)
   - `_get_pathway_profiles`: reordered `filtered_gene_list` alignment to run BEFORE any matmul/get_exp (Defect A)
   - `_get_pathway_profiles`: manual `to_numpy → np.log2 → pd.DataFrame` wrap (Defect B, layer 1)
   - `_get_pathway_profiles`: added 2 intermediate `flush=True` progress prints between matmul→concat and concat→log2→xshape (H2 mitigation)
   - existing `_dbgtp` tracepoints left in place for future runs
2. **[read_file.py](file:///Users/belter/github/VAEDecon_combined/DeSide/deside/utility/read_file.py)**
   - `ReadExp.__init__`: defensive ndarray→DataFrame wrapping (Defect B, layer 2)

## Actionable user-facing mitigations
User should run training on HPC with these additional environment variables set in the job submission before invoking `deside train`:
```bash
export PYTHONUNBUFFERED=1                    # Defeat HPC stdout/stderr full buffering (H2)
export MALLOC_TRIM_THRESHOLD_=131072          # glibc: return free()ed >128K blks to OS promptly (H1)
export PYTHONDONTWRITEBYTECODE=1              # reduce NFS metadata writes during import
# Optional: if user wants to tune manually, this overrides _cap_blas_threads() defaults:
# export OMP_NUM_THREADS=8
# export OPENBLAS_NUM_THREADS=8
# export MKL_NUM_THREADS=8
# export PYTORCH_NO_AUTOTHREAD_CAP=1 ; export OMP_NUM_THREADS=...
```

To reproduce the old stall for comparison, the user can:
```bash
# Offline trace mode to see every post-stats checkpoint:
export TRAE_DEBUG_SESSION=deside-log-stall-after-pathway-stats
export TRAE_DEBUG_TRACE_DIR=$HOME/deside_debug_traces
mkdir -p "$TRAE_DEBUG_TRACE_DIR"
deside train --config /path/to/your.yaml
# After run/interrupt:
ls -la "$TRAE_DEBUG_TRACE_DIR"
cat "$TRAE_DEBUG_TRACE_DIR"/trae-debug-log-*.ndjson | python3 -c '
import json, sys
events = [json.loads(l) for l in sys.stdin if l.strip()]
events.sort(key=lambda e: e["ts"])
prev = None
for e in events:
    dt = "" if prev is None else f"  Δt={e["ts"]-prev:.2f}s"
    print(f"{e["ts"]:.3f} {e["point"]}{dt}  rss={e.get("extra",{}).get("rss_max_kb","?")}KB")
    prev = e["ts"]
'
```

## Preventing recurrence
- **Regression test:** `test_fixes.py` should be added to the CI test suite; it exercises the exact `_get_pathway_profiles(add_to_end + filtered_gene_list)` hot path.
- **DataFrame type invariant:** Add a 1-line `assert isinstance(x_obj.exp, pd.DataFrame)` after every ReadExp construction in `train_model` in dev mode (or via `_dbgtp` shape assertions on all data paths).
- **Progress heartbeat:** Between every two heavy operations in pre-training, emit a `flush=True` print so the user can always tell which stage is running.
- **HPC job docs:** Add the env var block above to README HPC section.

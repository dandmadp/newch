# CLAUDE.md

newch: a small machine-learning library for low-spec machines. C11 engine + zero-dependency Python package
(ctypes). PyTorch-style tensors/autograd, Keras-style `compile`/`fit`. Public repo: https://github.com/dandmadp/newch

## Working with this user

- **Reply in Korean**, including long technical turns.
- The goal is a general low-spec ML library that *also* runs on SBCs, not a board-centric project. Prefer general
  features over board-specific tuning. ARM board measurement (Orange Pi) is deliberately on hold.
- Before anything public or irreversible (PyPI upload, new repos, force-push), confirm first. The user uploads to
  PyPI themselves; never ask for or handle their PyPI token.

## Environment (Windows)

- No system C compiler. Everything builds with `zig cc` from the venv: `.venv\Scripts\python.exe` (has ziglang,
  numpy, build, twine, wheel).
- GitHub CLI: `"C:\Program Files\GitHub CLI\gh.exe"` (logged in as `dandmadp`, has `workflow` scope).
- git identity is set in this repo's local config (dandmadp, GitHub noreply email). `.gitattributes` forces LF.
- Shell heredocs containing backticks can break the Bash tool; write patch scripts to a file and run them instead.

## Commands

```
.venv\Scripts\python.exe tools/build.py                 # build + C tests + Python tests (poison mode on)
.venv\Scripts\python.exe tools/build.py --debug          # -O0 + UBSan
.venv\Scripts\python.exe tools/build.py --target all --no-run   # every cross target, must stay warning-free
.venv\Scripts\python.exe -m unittest discover -s tests   # Python tests only (set NC_DEBUG_POISON=1)
.venv\Scripts\python.exe -m build && .venv\Scripts\python.exe -m twine check dist/*   # PyPI artifacts
```

When piping test output through `grep`, check the real exit code (`set -o pipefail` / `${PIPESTATUS[0]}`); a
filtered pipeline once hid a failing test that then got pushed.

## Layout

- `engine/include/nc.h`: public C API. Only `NC_API` functions are exported (`-fvisibility=hidden`).
- `engine/src/core/nc_internal.h`: op registry (`NC_OP_LIST`), `nc_op_def`, tensor/ctx structs.
- `engine/src/ops/`: every op = `infer` (shape only, never reads data) + optional `view` + `forward` + `backward`,
  run through `nc__apply()`. New op: define `nc_op_def`, add to `NC_OP_LIST`, add public fn to `nc.h`, add a
  gradcheck case in `engine/tests/test_main.c`.
- `engine/src/kernels/`: `ref/` (truth), `neon/`, `x86/` (AVX2 via per-function `target` attributes). Ops call
  `ctx->k` or `nc__gemm()` (threaded), never ISA code directly. `test_kernel_parity` compares tiers to ref.
- `newch/`: `tensor.py` (Tensor, Context, functional, NCW1 files), `net/` (package: `__init__.py` Module, layers,
  containers, losses, Model, Sequential, quantize_; `functional.py` = torch.nn.functional; `init.py` = torch.nn.init), `data.py` (DataLoader, FileArray), `training.py`, `optim.py` (optimizers, lr_scheduler), `callbacks.py`, `_lib.py` (ctypes
  signatures; keep in sync with `nc.h`, including struct layouts like `nc_config`).
- Docs: `docs/PROGRESS.md` (state, decisions, naming table, verification), `docs/FORMAT.md`, `docs/PUBLISHING.md`.

## Rules that are easy to break

- **Memory model:** temporaries live in the step arena until `step_end()`; persistent tensors always own their
  data (no persistent views), and the owning Python wrapper frees them in `__del__`. Backward releases memory as
  it goes (`nc_backward_ex`): node temporaries, intermediate grads, and activations nobody holds. Python passes
  its live temporaries (`Context._live`, kept by `Tensor._from_ptr` / `_w` / `__del__`) so they survive. Arena
  allocations >= chunk/4 are separate blocks with a budgeted cache (`ctx.c`). Backward ops must only allocate
  through `nc__arena_alloc` (logged and freed after the node) or `nc__grad_buf` (kept). Ops that are not recorded
  free their forward scratch on return; under `no_grad()` Python calls `nc_release_temp` when a temporary's last
  wrapper dies (`Tensor._free`). Data a recorded node uses is flagged `NC_TF_ON_TAPE` on its owner (views carry
  `owner`) and is never released early.
- **Never pass `make_tensor().ptr` straight into a C call.** CPython drops the temporary after `.ptr` is read,
  before the call runs; a persistent temporary is then freed (use-after-free). Bind it to a name first. Tests run
  with `NC_DEBUG_POISON=1` so this shows up on every platform.
- **Naming:** an API with a PyTorch/Keras name must behave the same; if behavior differs, use a similar but
  different name (e.g. `amax`, `mT`, `nc.net`, `quantize_`). Unsupported cases raise instead of silently differing.
  Record changes in the naming tables in `docs/PROGRESS.md`.
- Errors never abort: first error sticks in the context, ops return NULL and propagate it; Python raises `NcError`.
- GEMM threading must stay deterministic (one thread per output element).
- Version lives in three places: `pyproject.toml`, `newch/__init__.py`, `NC_VERSION_*` in `nc.h`.

## Status (2026-11-02)

- PyPI has `newch` 0.3.0 (uploaded by the user). Released versions can never be re-uploaded: any fix ships as a new
  version (bump all three version locations). The user runs `twine upload` themselves (`docs/PUBLISHING.md`).
- 0.3.1 (branch `torch-style-models`, not pushed yet): PyTorch-style model code (`newch.net` package with
  `functional`/`init`), new ops (nll_loss, compare, cat, to_dtype, i32 sum), backward memory release (-48% reserved), inference memory
  release, optimizer/scheduler `state_dict` with nested `save_tensors` checkpoints, `uint8` data dtype,
  transformer blocks (batched matmul, layer_norm, masked_fill, SDPA, MultiheadAttention, TransformerEncoder),
  lower Python call overhead, and the dedupe cleanup listed at the top of `docs/PROGRESS.md` (torch style is the
  base; Keras reduced to `fit()`; `newch.ml` removed; NN functions only in `F`; one save path).
- API surface is frozen at this level: do not add more library-mirroring APIs without discussing it first.
- Decided (2026-11-06): stage-3 scope grows, but GPU backends and large-model-only features (f16/bf16, KV cache,
  LLM-scale) go to an optional module in this repo (`engine/src/ext/` -> separate `nc_ext` library, Python
  `newch.ext`, loaded only on import) so the core stays small. Ops that also help small devices (transformer
  blocks, depthwise conv) stay in the core. The ext module does not exist yet; create it with the first GPU work.
- The project direction (memory-budget training, single-file C export, on-device learning) is still to be
  discussed with the user. NEON kernels are verified under QEMU in CI only, not on a real board.

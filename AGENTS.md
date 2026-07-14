# AGENTS.md — pyAQSC

## What this repo is

**pyAQSC** (`aqsc` on PyPI) constructs globally **quasi-symmetric (QS) stellarator equilibria** with **anisotropic pressure** via **near-axis expansion (NAE)** to arbitrary order. It is GPU-accelerated and JAX-differentiable; it does not rely on costly optimization for the construction itself.

Package: `src/aqsc/` · Import: `import aqsc` · Docs: https://pyaqsc.readthedocs.io/ · Version: see `pyproject.toml`.

Primary references: weakly QS NAE to all orders; overdetermination / generalized force balance; circular-axis solutions; related code [pyQSC](https://github.com/landreman/pyQSC).

## Layout

| Path | Role |
|------|------|
| `src/aqsc/` | Installable Python package |
| `test/` | Unit tests |
| `docs/` | MkDocs site source (`mkdocs.yml`) |
| `examples/` | Jupyter notebooks (iteration, JAX, looped solver) |
| `magnetic_recursion_relations/` | Maxima notebooks → magnetic-only recursion |
| `MHD_recursion_relations/` | Maxima notebooks → magnetic + force balance |
| `maxima_scripts/` | Maxima libs (governing eqs, series matching, Python codegen) |
| `ARCHITECTURE_PLAN.md` | Proposal for padded/`lax.fori_loop` JAX performance work (not fully implemented) |

Maxima notebooks are **sources of truth for formulas**; generated Python lives under `parsed/` and `MHD_parsed/`. Runtime does not need Maxima/wxMaxima.

## Core modules (`src/aqsc/`)

| Module | Purpose |
|--------|---------|
| `config.py` | Global flags: `use_pyQSC`, precision, FFT vs pseudospectral `diff_mode`, `compile_MHD_iteration` (JIT of full MHD iteration, **off by default**) |
| `chiphifunc.py` | `ChiPhiFunc`: \(f(\chi,\phi)\) as even/odd \(\chi\) Fourier coeffs on a \(\phi\) grid; arithmetic, derivatives, ODE/spectral solvers |
| `chiphiepsfunc.py` | `ChiPhiEpsFunc`: list-like power series in \(\epsilon=\sqrt{\psi}\) of `ChiPhiFunc`/scalars |
| `chiphifunc_padded.py` | WIP padded variants for JAX-friendly summation |
| `equilibrium.py` | `Equilibrium` container; `iterate_2` (full MHD), `iterate_2_magnetic_only` |
| `leading_orders.py` | `leading_orders` / `leading_orders_magnetic`, `circular_axis()`, axis helpers |
| `looped_solver.py` | Solves the “looped equations” that close the overdetermined system |
| `recursion_relations.py` | Wrappers around parsed recursion evaluations |
| `math_utilities.py` | Helpers used by generated/parsed expressions (`py_sum`, etc.) |
| `parsed/` | Machine-generated magnetic-only recursion relations. Do not edit. |
| `MHD_parsed/` | Machine-generated full (MHD + force balance) recursion relations. Do not edit. |
| `looped_coefs/` | Coefficients for looped equations |
| `aqsc_to_desc.py` | Export to DESC (`aqsc_to_desc_near_axis`, `aqsc_to_desc_boundary`) |
| `chiphifunc_test_suite.py` | Plotting / test helpers; optional pyQSC import |

`__init__.py` re-exports the public API from the modules above (not every submodule).

## Data model (must understand)

1. **`ChiPhiFunc`** — primary compute type (JAX-traced). `content` shape: axis 0 = \(\chi\) modes \(m=-n,-n+2,\ldots,n\); axis 1 = \(\phi\) grid over one field period. Different `len_phi` or `nfp` cannot mix.
2. **`ChiPhiEpsFunc`** — stores results as \(\sum_n \epsilon^n F_n(\chi,\phi)\). Supports `[]` / `append`, **not** `__setitem__`. Out-of-range / negative index → zero (`nfp=0`), required by Maxima-generated code.
3. **`Equilibrium`** — holds `unknown`, `constant`, `axis_info` (power series and axis geometry). Iteration returns a **new** `Equilibrium`.

Naming: `*_coef*` ≈ power series (`ChiPhiEpsFunc`); trailing `_c` / `_p` / `_cp` ≈ \(\chi\) / \(\phi\) / both dependence. Exception: `dl_p` is a scalar.

**Error/zero via `nfp`:** `nfp=0` means zero; `nfp<0` is a silent error code (JAX has no runtime exceptions). See docs/data-structure.md for codes (−2 … −17).

## Typical workflows

```python
import aqsc
eq = aqsc.circular_axis()          # demo circular-axis anisotropic eq to n=2
eq.display()
eq.unknown.keys(); eq.constant.keys(); eq.axis_info.keys()

# New full QS equilibrium (leading order n=2), then +2 orders:
eq = aqsc.leading_orders(...)      # see docs/init-and-iterate-eq.md
eq = aqsc.iterate_2(eq, ...)       # even orders; n_eval even if set

# Magnetic-only (no force balance):
eq = aqsc.leading_orders_magnetic(...)
eq = aqsc.iterate_2_magnetic_only(eq, ...)
```

Free parameters and solved quantities: `docs/background-free-params.md`, `docs/background-solves-for.md`.

## JAX / numerics conventions

- Dependencies: `numpy`, `matplotlib`, `jax`, `interpax`, `scipy` (optional pyQSC via `config.use_pyQSC`).
- Prefer `jax.numpy` and explicit `dtype=jnp.float64`. Enable x64 at process start: `jax.config.update("jax_enable_x64", True)` (GPU otherwise often drops to float32).
- Full MHD JIT is **off** by default (`compile_MHD_iteration = False`). Users wrap hot paths with `jax.jit(..., static_argnums=...)`.
- **Static** vs **traced**: ints/bools that set shapes/filters are usually static (changing them recompiles); physical floats are traced (AD-friendly). Documented per API.
- Term count / compile cost scales ~\(O(n^3)\). Optimal truncation often \(n\le 4\); high-\(n\) JIT can take tens of minutes to hours.
- Do not use Python `if` / list indexing on traced values inside jitted paths.

## Editing guidance for agents

- **Prefer editing hand-written modules** (`chiphifunc.py`, `chiphiepsfunc.py`, `equilibrium.py`, `looped_solver.py`, `leading_orders.py`, `math_utilities.py`, `config.py`). Treat `parsed/` and `MHD_parsed/` as generated; change Maxima sources + regenerate rather than hand-editing large eval files unless fixing a clear codegen bug.
- Ignore `.ipynb_checkpoints/` and do not commit them.
- Keep changes focused; match existing style (JAX arrays, operator overloads on `ChiPhiFunc` / `ChiPhiEpsFunc`).
- When adding physics recursions, preserve the Maxima → `python_parser.mac` → `parsed/` / `MHD_parsed/` pipeline described in README.
- `ARCHITECTURE_PLAN.md` describes future padded/`fori_loop` work; `padded` paths may be incomplete—verify before relying on them.
- Tests live in `test/`; examples in `examples/`. pyQSC-dependent tests need `use_pyQSC = True`.

## Docs map

| Need | File |
|------|------|
| Install / overview | `docs/index.md`, `README.MD` |
| Quick demo | `docs/quick-start.md` |
| Build / iterate eq | `docs/init-and-iterate-eq.md` |
| Magnetic-only | `docs/init-and-iterate-mag.md` |
| Data structures | `docs/data-structure.md` |
| JAX / AD | `docs/JAX.md` |
| API | `docs/api-*.md` |

## Contact

Maintainer: Lanke Fu (ffu@pppl.gov), PPPL.

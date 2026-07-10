# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**pyAQSC** is a Python library for constructing globally quasi-symmetric stellarator equilibria using near-axis expansion (NAE) to arbitrary orders with anisotropic pressure. It uses JAX for GPU acceleration and auto-differentiation, and spectral (FFT-based) numerical methods.

## Commands

### Install
```bash
pip install -e .
```

### Run Tests
```bash
# All tests
python -m unittest discover -s test -p "test_*.py"

# Single test file
python -m unittest test.test_circular_axis

# Single test case
python -m unittest test.test_solvers.TestSolver.test_solve_1d

# Verbose
python -m unittest discover -s test -p "test_*.py" -v
```

Tests use Python's built-in `unittest` framework (not pytest).

## Architecture

### Core Data Structures

- **`ChiPhiFunc`** (`chiphifunc.py`): Represents functions f(χ, φ) as χ-Fourier coefficients on φ-grids. Implements arithmetic operators, derivatives (`dchi`, `dphi`), and filtering. This is the fundamental computational unit.

- **`ChiPhiEpsFunc`** (`chiphiepsfunc.py`): A list of `ChiPhiFunc` objects forming a power series in ε (the near-axis expansion parameter). Handles out-of-bound indexing gracefully.

- **`Equilibrium`** (`equilibrium.py`): Container for all equilibrium coefficients (`unknown`, `constant`, `axis_info` dicts). Main user-facing class; provides `iterate_*` methods, plotting, and save/load.

### Computational Flow

```
Axis geometry (Rc, Rs, Zc, Zs) + physical parameters
    ↓
leading_orders()         — solves Riccati equation (orders 0 and 1)
    ↓
eq.iterate_*()           — solves higher-order terms iteratively
    ↓
looped_solver.py         — generates RHS from looped equations
    ↓
parsed/ or MHD_parsed/   — auto-generated recursion relations (from Maxima)
    ↓
math_utilities.py        — spectral ODE/PDE solvers (FFT-based)
    ↓
Equilibrium with all coefficients
```

### Auto-Generated Code

`src/aqsc/parsed/` and `src/aqsc/MHD_parsed/` contain Python files **auto-generated from Maxima symbolic algebra notebooks** (`magnetic_recursion_relations/` and `MHD_recursion_relations/`). Do not hand-edit these files. The two sets correspond to:
- `parsed/` — magnetic-only recursion relations
- `MHD_parsed/` — full MHD (magnetic + force balance) recursion relations

### JAX Specifics

JAX JIT-compiles functions and **recompiles whenever traced variable shapes or static variable values change**. Compilation is very slow (minutes to hours for high orders) but subsequent evaluation is fast (<100ms). The `config.py` sets global JAX options (double precision, FFT mode, grid sizes).

### Converters

- `aqsc_to_desc.py` — exports equilibria to DESC format
- `aqsc_to_qsc.py` — exports to pyQSC format (disabled by default in `config.py`)

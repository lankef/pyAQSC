# Architecture plan: padded ChiPhiFunc / ChiPhiEpsFunc

Status: proposal, not yet implemented. Written after reading `chiphifunc.py`,
`chiphiepsfunc.py`, `math_utilities.py`, `looped_solver.py`, `equilibrium.py`,
and a representative machine-generated file (`parsed/eval_Xn.py`), and after
reading the in-progress `padded` flag added to `ChiPhiFunc` on this branch.

## 0. What's actually happening today (revised per your clarification)

Correction: `jit` **is** applied — manually, wrapping the calls into
`iterate_2`/`iterate_looped` from outside the library — and the profiling
numbers you quoted ("very little time on the actual jitted algebra, tons of
time on logical overhead") were taken with that wrapping in place, not in
eager mode. That sharpens the diagnosis rather than changing it:

1. **This is a trace-time problem, and it's paid every single order, not
   once.** `n_eval` (the order being solved) is a different static value on
   every outer call, so JAX's compilation cache can't reuse a previous trace
   across orders — each order gets a fresh `jit` trace, which re-executes
   every Python-level operator-overload call in `parsed/eval_Xn.py` (and
   friends) from scratch to build that order's jaxpr. So the cost you
   profiled really is "cost of building the jaxpr for one order," and it
   recurs for every order of the whole solve — there's no way to amortize it
   across orders the way you could amortize a single upfront compile. That
   makes reducing trace-time Python-call-count the central lever, not a
   nice-to-have.
2. **The unrolling is inherent to the algorithm, not to `ChiPhiFunc`.** Look
   at `sum_arg_29`/`sum_arg_28`/`sum_arg_27` in `eval_Xn.py`: a 3-level nested
   `py_sum`, where the *bounds* of the inner sums (`0, i53-i54`) depend on the
   outer loop variables. `py_sum` currently unrolls with
   `reduce(add, (expr(i) for i in range(...)), ...)` — a Python loop, at trace
   time, over a Python range built from concrete Python ints (`n` is always a
   concrete Python int; every loop variable derived from it is therefore also
   concrete). This is why a triply-nested sum costs O(n³) individual Python
   calls into `ChiPhiFunc.__add__`/`__mul__`, each of which does real
   isinstance/shape/parity branching.

The existing WIP (`ChiPhiFunc.padded` flag + `py_sum_parallel`'s
`lax.fori_loop` branch) tries to address point 2, but as written it can't
actually fire correctly for the code that matters:

* Nothing in the codebase ever constructs a `ChiPhiFunc` with `padded=True`
  outside of propagating the flag through `+`/`*`/`dchi`/`dphi`. So
  `py_sum_parallel`'s `fori_loop` branch is currently **dead code** — the
  `isinstance(first, ChiPhiFunc) and first.padded` check never passes.
* Even if it did fire: `lax.fori_loop(lo, hi, body, init)` traces `body`
  **once** with a traced integer, then repeats it. If `body` is
  `lambda i, carry: carry + expr(i)`, then `expr(i)` — which is
  `sum_arg_N(i)`, which indexes `X_coef_cp[i]` — receives a **traced** `i`.
  But `ChiPhiEpsFunc.__getitem__` does `if index > len(self.chiphifunc_list)-1`,
  a Python `if` on a traced value. That raises (`TracerBoolConversionError`)
  for any `expr` that indexes a coefficient list by the loop variable — i.e.
  for essentially every `sum_arg_N` in the machine-generated code. So the
  fori_loop path, if it ever did fire, would break immediately on real input.

This second point is the answer to your Q1 worry, made concrete: **the
architecture question is not "where do operator overrides live," it's
"what does it take to make an indexing/arithmetic call survive being called
with a traced loop index."** That's a orthogonal, harder problem than class
layout, and it's the one worth designing around.

One thing in the existing generated code is a strong hint that this was
anticipated: `is_seq(a,b)` and `is_integer(a)` are implemented with
`jnp.where`, not Python `if`, even though at present they're only ever
called with concrete Python ints (where a plain `if` would've been simpler
and cheaper). They read like masks designed for a future where the argument
can be traced. We can lean on that.

## 1. Answering your four questions directly

### Q1 — Will a from-scratch `ChiPhiEpsFuncPadded`/`ChiPhiFuncPadded` still require the same unrolling? Can/should `nfp` be traced?

Your worry is justified but the fix isn't "put the logic somewhere else" —
it's "make shapes uniform enough that you can replace *many* Python-level
`__add__`/`__mul__` calls with *one* traced `lax.scan`/`fori_loop` body."
Moving the operator-overload code to a new class doesn't save anything by
itself. What saves time is:

* Uniform, statically-known `(chi_dim, len_phi)` shape for every element of
  a padded `ChiPhiEpsFuncPadded`, so a `scan` carry is well-typed and a
  `scan` body can be traced **once** instead of unrolled **N times**.
* An indexing path (`__getitem__`) that works when the index is a JAX
  tracer, using `jnp.take`/`dynamic_slice` + `jnp.where`-masking instead of
  Python `if`/list indexing — this is the load-bearing piece, not class
  hierarchy.

**`nfp` should stay static (a plain Python `int`), not traced.** Three
reasons:
* It's already static today (`self.nfp = nfp`, never wrapped in `jnp.array`),
  and every piece of branching logic that matters (`is_special()`,
  `self.nfp==other.nfp`, the whole error-code system) is a Python `if` on
  it. Tracing `nfp` would break every one of those and buy nothing, since...
* ...`nfp` never actually varies during a solve. It's one integer per
  `Equilibrium`. There's no shape-dependent recompilation cost from it being
  static — it's static the same way `n` (the order being evaluated) is
  static. Making it a JAX-traced scalar wouldn't reduce retracing (retracing
  is driven by `n` changing every `iterate_2` call, not by `nfp`), it would
  just make `nfp`-based control flow illegal.
* Where your "static list" instinct *is* right, and worth keeping, is for
  the **special/error-code bookkeeping**, not `nfp` itself: keep a small
  Python list of per-order sentinel codes (0 = normal, `<0` = error) living
  outside the traced array, exactly like today's `nfp<=0` convention but
  factored out so it can be inspected with real Python `if`s regardless of
  what happens to the numeric content. See §2.

So: yes, some logic has to live in the operator overloads of whatever class
you introduce — that's unavoidable, JAX needs *some* Python code to build
the jaxpr. The goal is to cut the *number of times* that logic executes at
trace time from O(number of terms) to O(number of distinct summation shapes),
and to make the majority of what remains branch-free arithmetic on uniform
arrays rather than isinstance/shape dispatch.

### Q2 — Optimal data split; does JAX "view" or "copy"?

**View vs. copy in JAX, concretely:** JAX arrays have no numpy-style
strided-view mechanism. `arr[i]` and `jnp.take` are logically
`slice`/`gather`/`dynamic-slice` primitives that produce a new logical
value. Whether a physical copy happens:
* **Outside `jit`:** yes, essentially every slice dispatches a real op and
  allocates a real buffer. There is no free view.
* **Inside `jit` (which is how this all actually runs, per §0):** XLA's
  buffer-assignment/fusion pass frequently elides the materialization of a
  slice that's immediately consumed downstream, especially a static slice
  feeding straight into an add/convolve. That's a compiler decision, made
  during the same trace whose Python-level cost we're trying to cut — it
  doesn't reduce trace time, it just means the *compiled* program is cheaper
  than the number of slices you wrote would suggest.

**Conclusion: don't design the class split around "avoid copies."** It's not
the lever that matters here. Chasing it (e.g. keeping `ChiPhiFuncPadded` as
a lazy view object with a back-reference to its parent) adds complexity for
a saving XLA already makes for you inside the compiled graph, without
touching the thing that's actually expensive (§0): the number of distinct
Python-level operator-overload calls executed while *building* the jaxpr in
the first place. That's the count to drive down (§4.1-§4.3), regardless of
what happens to the resulting compiled program's memory traffic.

**Data split**, given that (revised after your triangular-storage point —
see §4.1 for the full argument and the memory numbers):

`ChiPhiEpsFuncPadded` owns all the traced data and all cross-order static
metadata:
* `content: jnp.ndarray`, shape `(total_rows, len_phi_pad)` — every order's
  *natural-size* content concatenated along axis 0 (not padded to a common
  width up front — see §4.1 for why that's real, roughly 2x, waste, and how
  extraction still produces a uniform-shape result on demand).
* No stored `sizes`/`offsets` — per your later clarification, order `n`'s
  chi width is *exactly* `n+1` (modes `-n..n`, never more, never fewer) for
  every container this design actually needs to pad. `size(n)` and
  `offset(n)` are therefore closed-form functions of the index
  (`n+1` and `n(n+1)/2`), computed inline in `__getitem__` rather than
  looked up — see §4.1, which is a further, real simplification over the
  stored-list version I had here originally, not just a storage saving:
  it also removes a `jnp.take` gather from the traced extraction path.
* `nfp: int` — static, single value (matches current `ChiPhiEpsFunc.nfp`,
  which is already scalar, not per-order).
* `special: list[int]` — static, length `n_orders`, per-order sentinel code
  (0 = normal, matches today's `nfp<=0` codes). This one genuinely has to be
  stored, not derived — unlike chi width, which order is "special" isn't a
  function of the index. Lives in Python, not in the traced array, so
  `__getitem__` with a *concrete* index can still do a cheap Python `if` to
  short-circuit special elements, exactly like today.
* `n_max: int` — static, the highest known order (replaces `len(sizes)-1`;
  matches `ChiPhiEpsFunc.get_order()` today).
* `len_phi_pad` — static int.

Scope note on the closed-form chi width: it holds for the containers that
actually need chi-padding — `X_coef_cp`, `Y_coef_cp`, `Z_coef_cp`,
`B_theta_coef_cp`, `B_psi_coef_cp`, `p_perp_coef_cp`, `Delta_coef_cp`,
`B_denom_coef_c` (all constructed with `square_eps_series=False` in
`leading_orders.py`, all following the `n+1`-modes near-axis regularity
docstring). It does *not* apply to `iota_coef`/`B_alpha_coef`
(`square_eps_series=True`): checking `leading_orders.py`, their elements are
often plain scalars (`B_alpha0 = dl_p/jnp.sqrt(B0)`), not chi-expanding
`ChiPhiFunc`s at all — they're flux functions, chi-trivial at every order.
They never hit the padding problem in the first place and shouldn't be
forced through this container type — leave them as plain per-order scalars
(or the existing ragged `ChiPhiEpsFunc` machinery, which is already correct
and cheap for them).

`ChiPhiFuncPadded` owns nothing but a single order's *extracted, padded-to-
`chi_cap`* slice — `chi_cap` is chosen by the caller of `__getitem__`
(typically "current order being solved + 1"), not stored on the container:
* `content: jnp.ndarray`, shape `(chi_cap, len_phi_pad)`.
* `nfp: int` — static.
* `special: int | None` — static (`None` only for elements produced inside a
  traced-index `__getitem__`, where we can't know the code without reading
  traced data — see §4.2).

That's it. It should *not* carry a back-reference to its parent
`ChiPhiEpsFuncPadded` — once sliced out, it's self-contained data, which is
what lets it flow through `+`/`*`/`diff()` in the generated code exactly like
a `ChiPhiFunc` does today, with no coupling to where it came from.

### Q3 — How much can be shared with `ChiPhiFunc`/`ChiPhiEpsFunc` via an abstract superclass?

Less than you'd think at the *class-hierarchy* level, more than you'd think
at the *function* level:

* **Genuinely representation-agnostic numeric kernels** — reuse directly,
  don't reimplement: `dchi_op`, `trig_to_exp_op`, `exp_to_trig_op`,
  `batch_convolve`, `conv_tensor`, `jit_fftfreq_int`, the FFT-based body of
  `dphi`/`integrate_phi_fft`/`filter`. These operate on a 2D content array
  and don't care whether it came from a ragged or padded container. Both
  `ChiPhiFunc` and `ChiPhiFuncPadded` should call the *same* module-level
  functions.
* **Module-level dispatch functions already shared implicitly by duck
  typing** — no change needed: `diff()`, `is_seq()`, `is_integer()` in
  `math_utilities.py` only ever call `.dchi()`/`.dphi()` by name; they work
  on any object with those methods. This is already the mechanism that lets
  `parsed/*.py` and `MHD_parsed/*.py` stay untouched no matter which
  concrete type flows through them — it's the real "shared infrastructure,"
  and it's already in place. Formalize it with a `typing.Protocol` (not a
  real ABC — see below) so type checkers/readers can see the contract:
  `ChiPhiFuncLike` = `{content, nfp, is_special(), dchi(), dphi(), __add__,
  __sub__, __mul__, __truediv__, __neg__, __pow__}`.
* **What should *not* be shared via inheritance:** the arithmetic operator
  bodies themselves. `ChiPhiFunc.__add__`/`__mul__` are ~60% shape/parity
  reconciliation logic that is *structurally impossible* to hit in the
  padded world (see §4.1.2) — inheriting from a common base that implements
  those checks means paying for branches that can provably never trigger.
  Better to have `ChiPhiFuncPadded` implement a deliberately smaller
  `__add__`/`__mul__` from scratch (a few lines each) than to inherit and try
  to short-circuit the ragged version's checks.
* **Non-hot-path stuff** (`display`, `display_content`, `eval`,
  `export_single_nfp`, saving/loading) — don't duplicate at all. Give
  `ChiPhiFuncPadded`/`ChiPhiEpsFuncPadded` a cheap `.to_ragged()` /
  `ChiPhiEpsFunc.from_padded()` conversion and let those call the existing,
  already-correct `ChiPhiFunc`/`ChiPhiEpsFunc` implementations. Nobody plots
  or exports inside the order-by-order hot loop.

Concretely: `Protocol`s for typing/documentation, a handful of shared
free functions for numeric kernels, explicit (not inherited) `__getitem__`-
and conversion methods at the ragged/padded boundary. Skip a real `abc.ABC`
superclass — Python multiple-dispatch of dunder methods plus JAX pytree
registration already gets you everything an ABC would, without forcing
`ChiPhiFuncPadded.__add__` to route through machinery built for the ragged
case.

### Q4 — Ideal architecture (ignoring the original plan)

Four pieces, in priority order:

#### 4.0 Confirmed: this is a trace-time problem, and it's the right thing to attack

Since `jit` is already applied at the outer layer and retraces every order
(§0), there's no "turn jit on" step left to do — the entire remaining lever
is reducing the number of Python-level operator-overload calls executed
*while building each order's jaxpr*, which is exactly what §4.1-§4.3 target.
Worth instrumenting once before investing further, so the rest of this can
be measured against a real baseline rather than assumed to help: split trace
time from run time per order (e.g. `jax.jit(...).lower(...).compile()`
timed separately from the subsequent call, or compare a call with a brand
new `n_eval` against an immediate repeat of the same `n_eval`).

#### 4.1 `ChiPhiEpsFuncPadded` chi storage: ragged/triangular concatenation, not padded-per-slot

Your proposed design is correct, and better than a padded-per-order buffer
would be — including better than what I'd originally sketched here. Worth
walking through *why* it's correct, since the reasoning also simplifies the
arithmetic (§4.1.2).

**Why mode-centered alignment turns out not to be necessary.**
`ChiPhiFunc.content` already stores each order min-mode-first: row 0 is the
most negative present chi mode, rows step by 2 up to the most positive. For
two operands `a` (natural size `n_a+1`, modes `-n_a..n_a`) and `b` (natural
size `n_b+1`), a **raw, unaligned** `batch_convolve(a, b)` already gives the
right answer with no centering step: row `k` of the length-`(n_a+n_b+1)`
output is exactly the coefficient of mode `-(n_a+n_b)+2k`. That's just how
integer convolution works when both inputs start at their own index 0 —
nothing about it requires `a` and `b` to have been made the same size first.
This is, not coincidentally, exactly what the *current, unpadded*
`ChiPhiFunc.__mul__` already does (`batch_convolve(a, b)` with no alignment
step) — the padded case doesn't need new math, it needs to confirm the old
math still works after padding, which it does:

**Trailing-zero padding preserves this.** If `a` and `b` are each padded up
to a common width `chi_cap` by *appending* zeros (no insertion, no
symmetric split, no parity bookkeeping) and convolved, the output for
`k < n_a+n_b+1` is *identical* to the unpadded convolution — the padding
zeros only ever multiply into terms that were already going to be zero.
`full = batch_convolve(a_padded, b_padded)` has length `2*chi_cap-1`;
truncating to `full[:chi_cap]` is correct, and — because every valid
`py_sum` term pairs operands whose natural sizes sum to the target order
(`i` and `n-i` in `sum_arg_38`, for instance) — row `k` of that truncated
result is mode `-n+2k` for *every* valid term in a given sum, so accumulating
across terms in a `scan` adds like-aligned buffers throughout. This is
simpler than the current `padded=True` WIP's centered-truncation math
(`excess = (full.shape[0]-target)//2`): the window here is always
`[:chi_cap]`, no offset computation needed at all.

**Storage: ragged/triangular, not pre-padded, with closed-form offsets.**
Given the above, there's no reason to pad every stored order up to a common
width up front — that IS the ~2x-ish waste you're pointing at. And per your
clarification that order `n`'s chi width is *exactly* `n+1` — never more,
never fewer, for every one of the eight containers this design targets
(see the scope note in Q2) — `size(n)` and `offset(n)` don't need to be
*stored* at all, they're closed-form functions of `n`:

```
size(n)   = n + 1
offset(n) = n * (n + 1) // 2        # triangular number: rows used by orders 0..n-1
```

This is a further simplification beyond just avoiding storage waste: since
these are pure arithmetic, `offset`/`size` are computed identically whether
`n` is a concrete Python int *or* a JAX tracer — no `sizes`/`offsets` lists,
no `jnp.array(...)` constant-building, no `jnp.take` gather in the traced
path at all, just multiply/add/floor-divide. It also makes `.append()`
trivial: appending order `n`'s content is always exactly
`content = jnp.concatenate([content, new_rows])`, because
`offset(n) == total_rows_so_far` automatically falls out of appending in
order — no bookkeeping to keep in sync.

`content: jnp.ndarray`, shape `(total_rows, len_phi_pad)`. For orders
`0..n_max`, `total_rows = (n_max+1)(n_max+2)/2`, versus `(n_max+1)²` for a
naively pre-padded `(n_orders, chi_cap, len_phi)` array — **roughly half
the memory** — and versus the symmetric comb-buffer I originally suggested
(width `2*n_max+1` per order), **roughly a quarter**. Append a small margin
of dummy zero rows at the very end (see gotcha below).

**Extraction (`__getitem__`), both index modes:**

```python
def __getitem__(self, index, chi_cap):
    if isinstance(index, int):                              # static — common case
        if index < 0 or index > self.n_max:
            return ChiPhiFuncPadded.zero(chi_cap, self.len_phi_pad, self.nfp)
        start, size = index * (index + 1) // 2, index + 1
        raw = self.content[start:start + size]                  # static slice
        padded = jnp.pad(raw, ((0, chi_cap - size), (0, 0)))     # trailing zeros
        return ChiPhiFuncPadded(padded, self.nfp, self.special[index])
    # traced — only reachable from inside a scan/fori_loop body
    valid = (index >= 0) & (index <= self.n_max)               # self.n_max is static
    safe_index = jnp.clip(index, 0, self.n_max)
    start = safe_index * (safe_index + 1) // 2
    size = safe_index + 1
    row_indices = start + jnp.arange(chi_cap)
    window = jnp.take(self.content, row_indices, axis=0, mode='fill', fill_value=0.0)
    row_valid = (jnp.arange(chi_cap) < size) & valid
    return ChiPhiFuncPadded(jnp.where(row_valid[:, None], window, 0), self.nfp, None)
```

`chi_cap` is a parameter of the *extraction*, not a fixed property of the
container (see the bonus optimization below for why) — but it must be a
**static** Python int, since it fixes the shape of the returned
`ChiPhiFuncPadded`, which has to be known at trace time for `scan`/
arithmetic to work. Both branches preserve the load-bearing existing
contract ("out-of-bound index → 0"), which several docstrings note is
relied on for correctness (`ChiPhiEpsFunc.zero_append`'s comment: *"out of
range returns a special ChiPhiFunc, rather than 0, [was needed] to check
for logical error. Is now redundant"* — i.e. "out of range ⇒ 0" is already
the production contract, this just needs to keep holding under a traced
index too).

**Why `jnp.take(..., mode='fill')` instead of `lax.dynamic_slice_in_dim`.**
An earlier version of this used `dynamic_slice_in_dim`, which has a sharp
edge worth naming explicitly since it's easy to get wrong silently:
`dynamic_slice` always returns exactly the number of rows you ask for, even
if that runs past the end of the array — it does this by quietly sliding
`start` backward until the window fits. Concretely, with 6 stored rows
(orders 0,1,2: sizes 1,2,3) and a request for order 2 (`start=3`) padded to
`chi_cap=4` (`[3:7]`, one past the end), it wouldn't error — it would
silently return `[2:6]` instead, so row 0 of the "result" is actually order
1's `mode +1` row mislabeled as order 2's `mode -2`. No exception, just a
wrong number in the right-looking shape.

That's fully *fixable* (pad `content` with `max(0, chi_cap - (n_max+1))`
dummy rows so the window can never run past the end), but it's a fix you
have to remember to apply correctly, and the failure mode if you don't is
silent corruption, not a crash. `jnp.take(..., mode='fill', fill_value=0.0)`
sidesteps the whole category: out-of-bounds row indices come back as `0`
directly, unconditionally, with no margin bookkeeping required at all. It
trades a `dynamic_slice` for a `gather`, a marginally heavier primitive, but
given `chi_cap` is small (§4.1's bonus point — it tracks the current order,
not a global max), that cost is very unlikely to matter next to the
trace-time win this design is chasing, and it deletes an entire class of
"did I compute the margin right" risk. Use `mode='fill'` as the default;
only reach for `dynamic_slice` + an explicit margin if profiling later shows
the gather actually costs something.

**Bonus, independent optimization: `chi_cap` should track the call, not a
global max order.** `ChiPhiEpsFuncPadded` gets rebuilt from the ragged
`ChiPhiEpsFunc` once per `iterate_2` call anyway (§4.4's `to_padded`
boundary) — it doesn't need a single global `n_cap` sized for the eventual
final order of the whole solve. Since most `iterate_2` calls happen at
small `n_eval` early in a solve, choosing `chi_cap = n_eval + 1` (matching
*that* call's actual need) avoids paying for a large `chi_cap` on every
early, cheap order. This stacks with the ragged-storage saving rather than
replacing it: ragged storage saves memory *within* one call's container
across its differently-sized orders; per-call `chi_cap` saves memory
*across calls* by not over-sizing early containers for a future order they
don't need yet.

##### 4.1.2 Arithmetic, given ragged storage feeding fixed-`chi_cap` extraction

Because every `ChiPhiFuncPadded` that arithmetic ever touches already has
shape `(chi_cap, len_phi_pad)` by the time it comes out of `__getitem__`,
`__add__`/`__mul__` on `ChiPhiFuncPadded` itself don't need to know
anything about the ragged storage at all — they're exactly as simple as
uniform-shape arithmetic should be: `__add__` is `content_a + content_b`;
`__mul__` is `batch_convolve(content_a, content_b)[:chi_cap]` (fixed slice,
per the derivation above — no centered-truncation offset arithmetic). The
ragged storage is entirely an implementation detail of `ChiPhiEpsFuncPadded`;
`ChiPhiFuncPadded` never sees it, and never branches on shape or parity.

`len_phi_pad`: pad every order to the same φ-grid length up front
(`target_len_phi`, already computed once in `iterate_2` today) rather than
allowing the `shape[1]==1` "broadcast constant" degenerate case that
`__add__`/`__mul__` special-case today via `stretch_phi`. Broadcasting a
constant order to full length once, at `.append()` time, is strictly
cheaper than re-broadcasting it via `+jnp.zeros(...)` on every arithmetic op
that touches it — another branch eliminated for good, not deferred.

#### 4.3 `py_sum`/`py_sum_parallel`: scan-ify leaf sums first, staged

`math_utilities.py` is hand-written (not machine-generated), so all the
smarts belong here — this is the one place allowed to change behavior
without touching `parsed/`/`MHD_parsed/`.

The generated code nests `py_sum` up to 3 deep (`sum_arg_29` →
`sum_arg_28` → `sum_arg_27` in `eval_Xn.py`), with inner bounds depending on
outer loop variables. Fully vectorizing all levels at once means every level
below the vectorized one must also become trace-safe (traced bounds via
`jnp.floor`/`jnp.ceil` instead of Python `floor`/`ceil`, traced indexing
throughout) — doable since `is_seq`/`is_integer` are already
`jnp.where`-based, but it's a bigger surface to get right at once.

Pragmatic staging, in order of payoff/risk:

1. **Leaf sums** (`py_sum`/`py_sum_parallel` calls whose body contains no
   further `py_sum`) — the majority by call count, e.g. `sum_arg_38`
   through `sum_arg_2`, `sum_arg_18`/`17`/`16` in `eval_Xn.py`. These only
   need `ChiPhiEpsFuncPadded.__getitem__`'s traced branch (§4.1). Replace
   with `lax.fori_loop`/`lax.scan`, seeded with an explicit zero
   `ChiPhiFuncPadded` built from the caller's chosen `chi_cap` plus
   `ChiPhiEpsFuncPadded`'s *static* `len_phi_pad`/`nfp` — not, as today's
   fallback does, by evaluating `expr(lower_ceil)` and inferring shape from
   the result, since a `scan`'s carry must have a shape fixed before
   entering the loop. This alone converts the dominant cost driver
   (leaf-sum trace count, which scales with the total number of terms
   across the whole nested tree) from O(n) individually-traced Python calls
   to one traced loop body per leaf `py_sum` call site.
2. **Non-leaf sums** — leave as static Python unroll (today's `py_sum`)
   initially. Even with only leaf sums vectorized, a nested sum like
   `sum_arg_29` goes from "every innermost call is a full
   `ChiPhiFunc.__add__`/`__mul__` dispatch" to "every innermost call reuses
   one already-traced scan body" — the outer two Python loops still run,
   but their bodies get cheap. This is a real asymptotic win (roughly
   O(n²) trace-time Python calls instead of O(n³) for a 3-level nest)
   without touching bound-tracing correctness at all.
3. **Fully recursive vectorization** (traced bounds, `jnp.floor`/`ceil`
   dispatch, nested `scan`) as a phase-2 stretch goal once (1) is measured
   and validated. Flag explicitly as higher risk: getting dynamic
   `lax.fori_loop` bounds right for deeply nested, sign-sensitive
   summation ranges (`ceil(n/2)`, `i53-i54`, etc.) is exactly the kind of
   thing worth a dedicated correctness test pass (compare against the
   existing eager `py_sum` output, order by order, on a known-good
   equilibrium) before trusting it.

Detection of which path to take: keep the same shape as today's
`py_sum_parallel` — probe whether `expr`'s domain type is
`ChiPhiEpsFuncPadded`-backed (e.g. check `isinstance(coefficients, ...)` at
the `eval_*` call site, or thread a boolean through, rather than probing
`first.padded` after evaluating one term as today's WIP does — evaluating a
throwaway first term just to check a flag is itself wasted trace/eager work
that disappears once shape is known statically from the container).

#### 4.4 Class list, and choosing the backend once at `leading_orders`

* `ChiPhiFunc`, `ChiPhiEpsFunc` — **unchanged**. Stay the ragged,
  general-purpose, already-correct implementation.
* `ChiPhiFuncPadded`, `ChiPhiEpsFuncPadded` — new. Ragged/triangular storage
  with dual-mode (`chi_cap`-parameterized) indexing and minimal,
  branch-free arithmetic (§4.1).
* `ChiPhiFuncLike`/`ChiPhiEpsFuncLike` — `typing.Protocol`s documenting the
  contract `math_utilities.py` and `parsed/`/`MHD_parsed/` actually rely on
  (§Q3). Not enforced via inheritance; used for readability/typing only.

**Backend selection: one flag at `leading_orders`, no conversion boundary.**
Since both families satisfy the same duck-typed contract (§Q3), and every
downstream quantity traces back to what `leading_orders.py` constructs, add
one argument there (`padded: bool`, or similar) that picks which concrete
classes the *initial* `X_coef_cp`/`Y_coef_cp`/... containers are built with.
Everything downstream — `iterate_2`, `iterate_looped`, `recursion_relations.py`,
`parsed/*`, `MHD_parsed/*` — is unmodified and works with either family,
exactly as written today, since none of it hardcodes a concrete type in its
logic (only in a handful of constructor call sites, next point). This is a
much stronger validation story than a boundary-conversion function: it lets
you run the *entire* solve twice, once per backend, through unmodified code,
and diff the results order by order — not just check one conversion
round-trips correctly.

**The flag can't stop at `leading_orders.py` — there are other fresh
construction sites.** `equilibrium.py`, `looped_solver.py`, and
`recursion_relations.py` all contain hardcoded `ChiPhiFunc(...)` calls
outside of `leading_orders.py` — e.g. `iterate_2`'s default arguments
(`B_denom_nm1 = ChiPhiFunc(jnp.array([[0],[0]]), equilibrium.nfp)`),
`looped_solver.py`'s `ones = ChiPhiFunc(jnp.ones(...), nfp)`,
`recursion_relations.py`'s `ChiPhiFunc(jnp.zeros(...), X_coef_cp.nfp)`. Left
as-is, these would construct a ragged `ChiPhiFunc` even inside a
padded-backend solve — a hardcoded reference to the wrong family. Fix is
local and mechanical, not a new plumbing mechanism: replace the hardcoded
class name with the type of a same-family object already in scope —
`type(X_coef_cp[0])(...)` instead of `ChiPhiFunc(...)`. The handful of true
defaults with no obvious sibling nearby (`iterate_2`'s `B_denom_nm1`/
`B_denom_n`) can derive it from `equilibrium`'s existing containers instead
(`equilibrium` is already the first argument there). No new "current
backend" flag needs to be threaded through function signatures — the type
is always recoverable locally from whatever correctly-typed data is already
in scope at each site.

**Mixing is a hard error, not something to reconcile — but `other` has three
cases, not two.** `other` in `__add__`/`__mul__`/etc. is legitimately a bare
scalar today, and stays that way: every numeric constant in the generated
code (`dl_p**2`, `tau_p`, plain `2`, ...) reaches these operators as a raw
Python/JAX scalar, not a `ChiPhiFunc`, and `ChiPhiFunc.__add__`/`__mul__`
already special-case it via `jnp.array(other).ndim==0`. A blanket "must be
`isinstance(other, ChiPhiFuncPadded)` or raise" would wrongly reject those
too. The dispatch needs to mirror today's structure — own-family / scalar /
reject — with one addition: give the reject branch a distinguished case for
"you passed the *other* family" so the error is legible instead of an
incidental `jnp.array()` conversion failure several frames removed from the
actual bug:

```python
def __add__(self, other):
    if isinstance(other, ChiPhiFuncPadded):
        ...                                    # normal same-family path
    if isinstance(other, ChiPhiEpsFuncPadded):
        return other + self
    if isinstance(other, (ChiPhiFunc, ChiPhiEpsFunc)):
        raise TypeError(
            "ChiPhiFuncPadded combined with a ragged ChiPhiFunc/ChiPhiEpsFunc "
            "— likely a construction site still hardcodes the ragged class "
            "(see architecture plan §4.4)."
        )
    if jnp.ndim(other) == 0:
        ...                                    # scalar path, unchanged from today
    raise TypeError(f"unsupported operand for ChiPhiFuncPadded.__add__: {type(other)}")
```

Same shape for `ChiPhiEpsFuncPadded.append()`, which today also explicitly
allows appending a bare scalar (e.g. `B_alpha_coef.append(B_alpha_nb2)`
where `B_alpha_nb2` is frequently a raw scalar, per `leading_orders.py`).

This is safe to do with an ordinary `raise` (not a new negative-`nfp`
sentinel code): which family an object belongs to is a static Python-level
fact known the moment it's constructed, not something derived from a traced
value, so it doesn't run into the "JAX can't branch on tracers" constraint
that motivates the sentinel-code convention elsewhere in this codebase.
Because `ChiPhiFuncPadded` doesn't inherit from `ChiPhiFunc` (Protocol, not
ABC — §Q3), the `isinstance(other, (ChiPhiFunc, ChiPhiEpsFunc))` check
cleanly and unambiguously catches the other family, with no risk of an
accidental match through shared inheritance.

Combined with the previous point, this turns the migration from a manual
audit into a self-diagnosing one: run the padded backend once on a small
test equilibrium, let it crash with a `TypeError` at the first missed
hardcoded-`ChiPhiFunc` site, fix that one line, rerun. Repeat until it
completes. The error *is* the audit — no need to grep for every
construction site in advance.

**`to_padded()`/`to_ragged()` become optional interop, not a pipeline
stage.** With `leading_orders` constructing the chosen family directly,
there's no ragged→padded conversion inside the main `iterate_2` path at
all. Keep conversion methods only as an escape hatch for code that must
stay ragged regardless of backend — `display()`, `eval()`, DESC export —
in case someone wants to run those against a padded-backend result.

## 2. Migration plan

1. Instrument the existing manually-`jit`ted call to split trace time from
   run time per order (§4.0), so everything below has a real before/after
   baseline instead of an assumed one.
2. Implement `ChiPhiFuncPadded`/`ChiPhiEpsFuncPadded` per §4.1 (ragged
   storage, dual-mode `__getitem__` via `jnp.take(mode='fill')`, branch-free
   arithmetic that raises `TypeError` on cross-family mixing per §4.4).
3. Add the `padded`/backend flag to `leading_orders.py` (§4.4). Run the
   padded backend on a small known equilibrium; fix hardcoded-`ChiPhiFunc`
   construction sites in `equilibrium.py`/`looped_solver.py`/
   `recursion_relations.py` as the `TypeError`s surface them, one at a time,
   using the local `type(existing_object)(...)` pattern (§4.4) — this *is*
   the correctness pass for wiring, not a separate audit step. Once it runs
   to completion, diff its per-order output against the ragged backend on
   the same equilibrium (order-by-order numerical comparison) as the real
   correctness check — pay particular attention to extraction near the last
   stored order of a container, where an out-of-bounds `__getitem__` request
   is most likely.
4. Rewrite `py_sum_parallel` to do leaf-sum scan-ification (§4.3 stage 1),
   gated behind the container type, with the stage-1 fallback (today's
   `py_sum` unroll) preserved as the default for anything not yet converted.
   No changes required in `parsed/`, `MHD_parsed/`, or a hypothetical
   `looped_coefficients` — they only ever call `py_sum`/`py_sum_parallel`
   by name.
5. Profile again (trace time and execution time, separately, per order).
   Only pursue §4.3 stage 3 (nested scan) if the leaf-only win isn't enough
   and the profile shows non-leaf sums are still the bottleneck.

Nothing in steps 2-4 requires editing `parsed/*.py`, `MHD_parsed/*.py`, or
any other generated file — the whole point of keeping `py_sum`, `diff()`,
`is_seq()`, `is_integer()` as the stable interface is that generated code
only ever calls those four names plus arithmetic operators and
`ChiPhiEpsFuncPadded.__getitem__`, all of which are hand-written and safe to
change underneath it.

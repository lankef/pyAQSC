'''
ChiPhiFuncPadded / ChiPhiEpsFuncPadded: a uniform-chi-shape implementation
of the ChiPhiFunc / ChiPhiEpsFunc duck-typed contract (see chiphifunc.py,
chiphiepsfunc.py), designed for JAX trace-time performance rather than
minimal-per-object storage.

Why this exists -----

parsed/*.py, MHD_parsed/*.py and looped_coefs/*.py (all machine-generated,
never edited by hand) unroll deeply-nested py_sum/py_sum_parallel calls into
thousands of individual Python-level ChiPhiFunc.__add__/__mul__ calls per
solved order. Because every ChiPhiFunc in that recursion naturally has a
different chi width (order n has exactly n+1 chi components), those calls
carry real shape/parity-reconciliation branching, which is what actually
costs JAX trace time -- not the compiled algebra itself. ChiPhiFuncPadded
gives every object flowing through one computation a *statically known,
uniform* chi width, so lax.scan can replace Python-level unrolling in
py_sum_parallel (see math_utilities.py), and __add__/__mul__ themselves
become branch-free.

Chi representation -----

ChiPhiFunc's convention: a content array with L rows represents chi modes
-(L-1)..+(L-1) in steps of 2 (row 0 = most negative mode). L's parity is
tied to the physical order's parity: order n has n+1 rows, so even orders
have odd row-counts (and their modes -- and only theirs -- include 0), odd
orders have even row-counts (modes never include 0).

Adding two ChiPhiFunc-like objects only makes sense mode-by-mode. If two
objects of different natural width are embedded into a shared buffer using
*trailing* zero-padding (row 0 = each object's own natural minimum mode),
their row-to-mode mappings differ whenever their natural widths differ, and
naive row-wise addition silently sums together different chi modes. This is
a real failure mode in the generated code, not a hypothetical one: e.g.
parsed/eval_Ynp1.py's rhs_minus_lhs adds a width-(n+3) term (sum_arg_31) and
a width-(n+1) term (sum_arg_27) directly in its top-level `out=`.

The fix is the same symmetric, *centered* zero-padding ChiPhiFunc.__add__
and .cap_m() already use (see centered_resize_content in chiphifunc.py),
just with a statically-chosen target width instead of one computed from
max(a.shape, b.shape) at every call. Because ChiPhiFunc.__add__ already
rejects mismatched content.shape[0]%2 as an error (cross-parity addition is
already unsupported in the ragged implementation), splitting storage and
target widths by parity -- rather than using one dense buffer wide enough
for both parities -- costs no extra memory over exact packing and loses no
capability versus today's ragged behavior:

- Even orders (0, 2, 4, ...): shared target width chi_cap_even = M_even+1
  (M_even even), row r <-> mode 2r - M_even. Row M_even/2 is mode 0 for
  every even-order object regardless of its own natural width.
- Odd orders (1, 3, 5, ...): shared target width chi_cap_odd = M_odd+1
  (M_odd odd), same formula mode = 2r - M_odd. No single row is mode 0
  (odd orders never contain it), but the two rows nearest the middle are
  always modes -1 and +1 for every odd-order object, which is exactly
  where every odd-order object's own natural content is centered too.

Cross-parity products (e.g. X[i]*X[n-i] where n is odd forces i, n-i to
have opposite parity) work out automatically: convolution is index-additive
regardless of each operand's own centering convention, so the raw output's
row R always represents mode 2R - (M_a+M_b), and M_a+M_b's parity is
exactly the XOR of the operand parities (even+even=even, odd+odd=even,
even+odd=odd) -- matching even*even=even, odd*odd=even, even*odd=odd. No
special-casing needed; centered_resize_content's static-target formula
handles narrowing the raw output down to whichever target buffer is wanted.

Storage -----

ChiPhiEpsFuncPadded stores two parallel *triangular* (ragged) arrays, one
per parity, holding only each order's true natural-width content
concatenated together -- no per-slot padding waste. Order->offset within
one parity is closed-form (sum of an arithmetic sequence), so no per-order
metadata beyond a `special` sentinel-code list needs to be stored. The
*extraction* target width (chi_cap_even/chi_cap_odd) is a property of the
container, not of any one stored order -- chosen once via `mode_cap`
(the largest order magnitude this container's buffers need to represent),
and grown as needed via .append()/.resize(). This has to be a container-
level property, not a per-call parameter, because the machine-generated
code always indexes via plain subscript (X_coef_cp[i]), which can only pass
one argument through Python's __getitem__ protocol.

Backend selection -----

A single flag chosen once in leading_orders.py picks ChiPhiFuncPadded vs.
ChiPhiFunc for an entire solve. The two families are never mixed -- every
operator here checks for the *other* family explicitly and raises
TypeError, rather than attempting to reconcile. This is deliberate: it
turns "did every hand-written construction site get updated for this
backend" from a manual audit into a self-diagnosing one (the first missed
site throws immediately, with a message pointing back here).

Not shared with ChiPhiFunc/ChiPhiEpsFunc: any operator body (the ragged
shape-reconciliation branching is structurally impossible to hit here, so
inheriting it would mean paying for dead branches). What *is* shared:
representation-agnostic free functions operating on plain arrays --
batch_convolve, dchi_op, centered_resize_content, etc. -- imported directly
from chiphifunc.py. No abstract base class: ChiPhiFuncPadded does not
inherit from ChiPhiFunc, specifically so `isinstance(other, ChiPhiFunc)`
unambiguously means "the other family," which is what the mixing check
above relies on. See typing.Protocol definitions below for the documented
duck-typed contract instead.
'''
import jax.numpy as jnp
import jax.lax as lax
import matplotlib.pyplot as plt

from jax import tree_util
from typing import Protocol, runtime_checkable, Union

from .config import *
from .chiphifunc import ChiPhiFunc, ChiPhiFuncSpecial, display_content_shared
from .chiphiepsfunc import ChiPhiEpsFunc
from .math_utilities import (
    batch_convolve,
    dchi_op,
    centered_resize_content,
    jit_fftfreq_int,
    fft_filter,
    dphi_op_pseudospectral,
    trig_to_exp_op,
    exp_to_trig_op,
    get_l2_shared,
)

if double_precision:
    from jax import config as jax_config
    jax_config.update("jax_enable_x64", True)
    _complex_dtype = jnp.complex128
else:
    _complex_dtype = jnp.complex64

''' 0. Duck-typed contracts (documentation/typing only, not enforced) '''

@runtime_checkable
class ChiPhiFuncLike(Protocol):
    content: jnp.ndarray
    nfp: int
    def is_special(self) -> bool: ...
    def dchi(self, order: int = 1): ...
    def dphi(self, order: int = 1, mode: int = 0): ...
    def cap_m(self, m: int): ...
    def __add__(self, other): ...
    def __sub__(self, other): ...
    def __mul__(self, other): ...
    def __truediv__(self, other): ...
    def __neg__(self): ...
    def __pow__(self, other): ...

@runtime_checkable
class ChiPhiEpsFuncLike(Protocol):
    nfp: int
    def __getitem__(self, index): ...
    def append(self, item): ...
    def mask(self, n): ...
    def get_order(self) -> int: ...
    def dchi(self): ...
    def dphi(self): ...


''' I. ChiPhiFuncPadded '''

def _widths_for_mode_cap(mode_cap: int):
    '''
    mode_cap: the largest chi mode magnitude a container's buffers need to
    represent (not necessarily an order that's actually stored -- callers
    size this to whatever order they're about to *compute*, which can be
    ahead of what's currently appended).

    Returns (chi_cap_even, chi_cap_odd): row-counts (M+1) for the even-order
    and odd-order family buffers respectively, each M rounded down to the
    matching parity. Floors at the smallest valid width for each parity
    (1 row for even i.e. m=0; 2 rows for odd i.e. m=1), so a container is
    always safely constructible even at mode_cap=0.
    '''
    m_even = mode_cap if mode_cap % 2 == 0 else mode_cap - 1
    m_even = max(m_even, 0)
    m_odd = mode_cap if mode_cap % 2 == 1 else mode_cap - 1
    m_odd = max(m_odd, 1)
    return m_even + 1, m_odd + 1


class ChiPhiFuncPadded:
    '''
    A ChiPhiFunc-like object with a chi width chosen by its container
    (ChiPhiEpsFuncPadded), not by its own natural order. See module
    docstring for the centering convention.

    Two internal representations, distinguished by the static is_comb flag:

    - Normal (is_comb=False, the default, what every public-facing object
      uses): content.shape[0]'s parity determines which family (even-order
      / odd-order) this object belongs to, exactly like ChiPhiFunc today.
      Row r <-> mode 2r - M (M = content.shape[0]-1).
    - Comb (is_comb=True): a transient, parity-agnostic representation used
      only inside math_utilities.py's py_sum_parallel leaf-sum lax.scan
      machinery, needed because a *traced* summation index's parity is not
      known at trace time (see module docstring's "Backend selection"
      section and the py_sum_parallel implementation for why this is
      unavoidable given machine-generated code calls +/* directly on
      whatever __getitem__ returns). Row r <-> mode r - mode_cap
      (mode_cap = (content.shape[0]-1)//2), spanning both parities in one
      dense buffer. Never mix comb and normal operands -- __add__/__mul__
      raise if is_comb disagrees between two ChiPhiFuncPadded operands.
      Convert back to normal representation via .comb_to_normal(target_m)
      once the target order's parity is known statically (always true by
      the time a leaf sum's accumulated result needs to rejoin the rest of
      an enclosing formula).

    Members mirror ChiPhiFunc: content (traced 2d array, or jnp.nan when
    special), nfp (static int; <=0 encodes a special/error sentinel exactly
    like ChiPhiFunc, see ChiPhiFuncSpecial/ChiPhiFuncPaddedSpecial).
    '''
    def __init__(self, content: jnp.ndarray, nfp: int, is_comb: bool = False, trig_mode: bool = False):
        self.is_comb = is_comb
        if nfp <= 0:
            self.content = jnp.nan
            self.nfp = nfp
        else:
            if content.ndim != 2:
                self.content = jnp.nan
                self.nfp = -7
            elif not isinstance(nfp, int):
                self.content = jnp.nan
                self.nfp = -2
            else:
                self.nfp = nfp
                self.content = jnp.asarray(content).astype(_complex_dtype)
                if trig_mode:
                    n_dim = self.content.shape[0]
                    self.content = trig_to_exp_op(n_dim) @ self.content

    def is_special(self):
        return self.nfp <= 0

    def __str__(self):
        tag = ', comb' if self.is_comb else ''
        if self.is_special():
            msg = 'conditional 0' if self.nfp == 0 else 'error ' + str(self.nfp)
            return 'ChiPhiFuncPadded(' + msg + tag + ')'
        return 'ChiPhiFuncPadded(content.shape=' + str(self.content.shape) + ', nfp=' + str(self.nfp) + tag + ')'

    def _tree_flatten(self):
        return ((self.content,), {'nfp': self.nfp, 'is_comb': self.is_comb})

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    ''' I.1 Operator overloads '''

    def __getitem__(self, index: int):
        '''
        Obtains the m=index chi mode component, mirroring ChiPhiFunc.
        __getitem__ (mode-indexing, not list-indexing). Returns a
        ChiPhiFuncPadded of shape (1, len_phi) (or a bare scalar if
        len_phi==1 too), or ChiPhiFuncPaddedSpecial(-3) for an invalid mode
        number (wrong parity, or out of the represented range).
        '''
        if self.is_special():
            return self
        len_chi = self.content.shape[0]
        if self.is_comb:
            mode_cap = (len_chi - 1) // 2
            if abs(index) > mode_cap:
                return ChiPhiFuncPaddedSpecial(-3)
            row = index + mode_cap
        else:
            if len_chi % 2 == index % 2 or abs(index) > abs(len_chi - 1):
                return ChiPhiFuncPaddedSpecial(-3)
            row = (index + len_chi - 1) // 2
        new_content = jnp.array([self.content[row]])
        if new_content.shape == (1, 1):
            return new_content[0, 0]
        return ChiPhiFuncPadded(new_content, self.nfp, self.is_comb)

    def __neg__(self):
        return ChiPhiFuncPadded(-self.content, self.nfp, self.is_comb) if not self.is_special() else self

    def _check_comb_match(self, other):
        if self.is_comb != other.is_comb:
            raise ValueError(
                f'ChiPhiFuncPadded: combined a comb-representation object (is_comb='
                f'{self.is_comb}) with a normal-representation one (is_comb={other.is_comb}). '
                'This is an internal invariant violation, not a user-facing family mismatch -- '
                'comb representation is only meant to live transiently inside '
                'py_sum_parallel\'s scan body and should never escape it or meet a '
                'normal-representation object.'
            )

    @staticmethod
    def _coerce_operand(other):
        '''
        A genuinely-special (nfp<=0: zero or error sentinel) ragged
        ChiPhiFunc can legitimately meet a ChiPhiFuncPadded operand in a
        well-formed padded-backend computation: iota_coef/B_alpha_coef stay
        on plain ChiPhiEpsFunc/scalars even under the padded backend (see
        ChiPhiEpsFuncPadded's docstring -- they're chi-trivial, padding buys
        nothing), and out-of-bounds access on them
        (ChiPhiEpsFunc.__getitem__) returns ragged ChiPhiFuncSpecial(0),
        which then gets multiplied/added against ChiPhiFuncPadded
        quantities elsewhere in the same generated formula (e.g.
        `iota_coef[n-i]*X_coef_cp_padded[...]`-style expressions are
        pervasive in the census). Special-ness is a type-erased marker by
        design (see ChiPhiFuncSpecial's docstring in chiphifunc.py) -- it
        should propagate regardless of which family produced it.

        A *non-special* ragged ChiPhiFunc meeting a ChiPhiFuncPadded,
        conversely, is never legitimate under this design (every
        chi-expanding, non-chi-trivial quantity is padded consistently
        throughout one backend) -- that case is left alone here and still
        hits the TypeError raised by the caller.
        '''
        if isinstance(other, ChiPhiFunc) and other.is_special():
            return ChiPhiFuncPaddedSpecial(other.nfp)
        return other

    def _reconcile_width(self, other_content):
        '''
        Both operands are guaranteed same is_comb (checked by the caller)
        and, for the normal case, same parity. Their static width can still
        differ if they came from containers/scans with different
        mode_cap -- an exceptional, not routine, case (routine same-sum
        terms always share a container). Reconcile via the same centered
        formula used everywhere else (valid for both representations, since
        both are symmetric about their own center row), at pure trace-time
        (shapes are static), not per-scan-iteration cost.
        '''
        len_a = self.content.shape[0]
        len_b = other_content.shape[0]
        if len_a == len_b:
            return self.content, other_content, len_a
        target = max(len_a, len_b)
        return (
            centered_resize_content(self.content, target),
            centered_resize_content(other_content, target),
            target,
        )

    def __add__(self, other):
        other = self._coerce_operand(other)
        if isinstance(other, ChiPhiFuncPadded):
            if self.nfp == 0:
                return other
            if other.nfp == 0:
                return self
            if self.is_special():
                if other.is_special():
                    return ChiPhiFuncPaddedSpecial(self.nfp)
                return self
            if other.is_special():
                return other
            if self.nfp != other.nfp:
                return ChiPhiFuncPaddedSpecial(-2)
            self._check_comb_match(other)
            if not self.is_comb and self.content.shape[0] % 2 != other.content.shape[0] % 2:
                return ChiPhiFuncPaddedSpecial(-4)
            a, b, _ = self._reconcile_width(other.content)
            return ChiPhiFuncPadded(a + b, self.nfp, self.is_comb)
        if isinstance(other, ChiPhiEpsFuncPadded):
            return other + self
        if isinstance(other, (ChiPhiFunc, ChiPhiEpsFunc)):
            raise TypeError(
                'ChiPhiFuncPadded combined with a ragged ChiPhiFunc/ChiPhiEpsFunc -- '
                'this almost always means a hand-written construction site still '
                'hardcodes the ragged class instead of deriving it from a same-family '
                'sibling in scope (see architecture plan, "fixing the hardcoded '
                'construction sites").'
            )
        if jnp.ndim(other) != 0:
            return ChiPhiFuncPaddedSpecial(-5)
        if self.is_special():
            if self.nfp == 0:
                return other
            return self
        if self.is_comb:
            # Comb buffers always contain a row at mode 0 (mode_cap - mode_cap
            # = 0 is always representable), regardless of what parity the
            # eventual target order will be -- unlike the normal
            # representation, there's no even/oddness to check here.
            center_loc = self.content.shape[0] // 2
            updated_center = self.content[center_loc] + other
            updated_content = self.content.at[center_loc, :].set(updated_center)
            return ChiPhiFuncPadded(updated_content, self.nfp, True)
        if self.content.shape[0] % 2 == 0:
            return ChiPhiFuncPaddedSpecial(-4)
        center_loc = self.content.shape[0] // 2
        updated_center = self.content[center_loc] + other
        updated_content = self.content.at[center_loc, :].set(updated_center)
        return ChiPhiFuncPadded(updated_content, self.nfp)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -(self - other)

    def __mul__(self, other):
        other = self._coerce_operand(other)
        if isinstance(other, ChiPhiFuncPadded):
            if self.nfp == 0:
                return self if other.nfp < 0 else self
            if other.nfp == 0:
                return self if self.nfp < 0 else other
            if self.is_special():
                if other.is_special():
                    return ChiPhiFuncPaddedSpecial(self.nfp)
                return self
            if other.is_special():
                return other
            if self.nfp != other.nfp:
                return ChiPhiFuncPaddedSpecial(-2)
            self._check_comb_match(other)
            stretch_phi = jnp.zeros((1, max(self.content.shape[1], other.content.shape[1])))
            a = self.content + stretch_phi
            b = other.content + stretch_phi
            full = batch_convolve(a, b)
            if self.is_comb:
                # Comb convolution needs no centered re-truncation: row R of
                # the raw output already means mode R - (mode_cap_a +
                # mode_cap_b) directly (index-additive convolution), and
                # full's own width (2*mode_cap_a+1)+(2*mode_cap_b+1)-1 =
                # 2*(mode_cap_a+mode_cap_b)+1 is already exactly a valid comb
                # width for mode_cap_out = mode_cap_a+mode_cap_b.
                return ChiPhiFuncPadded(full, self.nfp, True)
            m_a = self.content.shape[0] - 1
            m_b = other.content.shape[0] - 1
            m_out = m_a + m_b
            target_width = centered_resize_content(full, m_out + 1)
            return ChiPhiFuncPadded(target_width, self.nfp)
        if isinstance(other, ChiPhiEpsFuncPadded):
            return other * self
        if isinstance(other, (ChiPhiFunc, ChiPhiEpsFunc)):
            raise TypeError(
                'ChiPhiFuncPadded combined with a ragged ChiPhiFunc/ChiPhiEpsFunc -- '
                'likely a hardcoded construction site (see __add__ for detail).'
            )
        if jnp.ndim(other) != 0:
            return ChiPhiFuncPaddedSpecial(-5)
        if self.is_special():
            return self
        return ChiPhiFuncPadded(other * self.content, self.nfp, self.is_comb)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = self._coerce_operand(other)
        if isinstance(other, ChiPhiFuncPadded):
            if other.is_special():
                if other.nfp == 0:
                    return ChiPhiFuncPaddedSpecial(-8)
                if self.is_special():
                    return ChiPhiFuncPaddedSpecial(self.nfp)
                return other
            if self.nfp == 0:
                return self
            if self.nfp != other.nfp:
                return ChiPhiFuncPaddedSpecial(-2)
            self._check_comb_match(other)
            # Ragged ChiPhiFunc.__truediv__ requires other.content.shape[0]
            # ==1 (chi-independent divisor) as a validity check -- but under
            # the padded backend, a genuinely chi-independent quantity
            # (e.g. B_denom_coef_c[0]) still comes back padded to its
            # container's chi_cap, so shape[0]==1 is never true even when
            # the divisor really is chi-independent. shape can no longer
            # serve as a validity proxy here; narrow to the center (chi-
            # independent) component directly instead. Per ChiPhiFunc.
            # __truediv__'s own docstring, plain / in the generated code is
            # only ever used against a genuinely chi-independent divisor
            # (the one chi-dependent "division" case goes through
            # get_O_O_einv_from_A_B instead, never through /) -- same trust
            # assumption this codebase already makes about formula
            # correctness elsewhere (e.g. is_integer()'s docstring).
            other_center = centered_resize_content(other.content, 1)
            return ChiPhiFuncPadded(self.content / other_center, self.nfp, self.is_comb)
        if isinstance(other, (ChiPhiFunc, ChiPhiEpsFunc)):
            raise TypeError(
                'ChiPhiFuncPadded combined with a ragged ChiPhiFunc/ChiPhiEpsFunc -- '
                'likely a hardcoded construction site (see __add__ for detail).'
            )
        if self.nfp == 0:
            return self
        if jnp.ndim(other) != 0:
            return ChiPhiFuncPaddedSpecial(-5)
        return ChiPhiFuncPadded(self.content / other, self.nfp, self.is_comb)

    def __rtruediv__(self, other):
        if self.is_special():
            if self.nfp == 0:
                return ChiPhiFuncPaddedSpecial(-8)
            return self
        # self is the divisor here -- same shape-can't-validate-anymore
        # reasoning as __truediv__ above: narrow to the center component
        # instead of checking shape[0]==1.
        self_center = centered_resize_content(self.content, 1)
        other = self._coerce_operand(other)
        if isinstance(other, ChiPhiFuncPadded):
            if other.is_special():
                return other
            self._check_comb_match(other)
            return ChiPhiFuncPadded(other.content / self_center, self.nfp, self.is_comb)
        if isinstance(other, (ChiPhiFunc, ChiPhiEpsFunc)):
            raise TypeError(
                'ChiPhiFuncPadded combined with a ragged ChiPhiFunc/ChiPhiEpsFunc -- '
                'likely a hardcoded construction site (see __add__ for detail).'
            )
        if jnp.ndim(other) != 0:
            return ChiPhiFuncPaddedSpecial(-5)
        return ChiPhiFuncPadded(other / self_center, self.nfp, self.is_comb)

    def __pow__(self, other):
        if self.is_special():
            return self
        if other % 1 != 0:
            return ChiPhiFuncPaddedSpecial(-9)
        if other == 0:
            return 1
        out = self
        for _ in range(int(other) - 1):
            out = out * self
        return out

    ''' I.2 Derivatives '''

    def dchi(self, order=1):
        if self.is_special():
            return self
        len_chi = self.content.shape[0]
        if self.is_comb:
            # Comb rows are step-1 (mode = row - mode_cap), unlike the
            # normal representation's step-2 (mode = 2*row - M) -- the mode
            # multiplier formula must match.
            mode_cap = (len_chi - 1) // 2
            mode_i = (1j * (jnp.arange(len_chi) - mode_cap)[:, None]) ** order
        else:
            mode_i = (1j * jnp.arange(-len_chi + 1, len_chi + 1, 2)[:, None]) ** order
        return ChiPhiFuncPadded(mode_i * self.content, self.nfp, self.is_comb)

    def dphi(self, order: int = 1, mode=0):
        if self.is_special():
            return self
        if mode == 0:
            mode = diff_mode
        if mode == 1:
            len_phi = self.content.shape[1]
            content_fft = jnp.fft.fft(self.content, axis=1)
            fftfreq_temp = jit_fftfreq_int(len_phi) * 1j
            out_content_fft = content_fft * fftfreq_temp[None, :] ** order
            out = jnp.fft.ifft(out_content_fft, axis=1)
        elif mode == 2:
            out = self.content
            for _ in range(order):
                out = (dphi_op_pseudospectral(self.content.shape[1]) @ out.T).T
        else:
            raise ValueError(f'ChiPhiFuncPadded.dphi: unsupported diff mode {mode}')
        # dphi only ever touches the phi axis, so it's representation-
        # agnostic -- just carry is_comb through unchanged.
        return ChiPhiFuncPadded(out * self.nfp ** order, self.nfp, self.is_comb)

    def antid_chi(self):
        '''
        Anti-derivative in chi. Ignores the m=0 component, if any. Mirrors
        ChiPhiFunc.antid_chi, with the same comb-vs-normal row<->mode
        distinction as dchi (see dchi's comment).
        '''
        if self.is_special():
            return self
        len_chi = self.content.shape[0]
        if self.is_comb:
            mode_cap = (len_chi - 1) // 2
            temp = (jnp.arange(len_chi) - mode_cap).astype(jnp.float32)[:, None]
            temp = temp.at[mode_cap].set(jnp.inf)
        else:
            temp = jnp.arange(-len_chi + 1, len_chi + 1, 2, dtype=jnp.float32)[:, None]
            if len_chi % 2 == 1:
                temp = temp.at[len_chi // 2].set(jnp.inf)
        return ChiPhiFuncPadded(-1j * self.content / temp, self.nfp, self.is_comb)

    def fft(self):
        return ChiPhiFuncPadded(jnp.fft.fft(self.content, axis=1), self.nfp, self.is_comb)

    def ifft(self):
        return ChiPhiFuncPadded(jnp.fft.ifft(self.content, axis=1), self.nfp, self.is_comb)

    def exp(self):
        '''
        e**(self). Only mathematically valid for chi-independent objects
        (matches ChiPhiFunc.exp's contract) -- but under the padded
        backend, shape[0]==1 can no longer be used to validate that (see
        __truediv__ for the same issue and full reasoning); narrow to the
        center component instead of rejecting on shape.
        '''
        if self.is_special():
            return self
        center = centered_resize_content(self.content, 1)
        return ChiPhiFuncPadded(jnp.exp(center), self.nfp, self.is_comb)

    def integrate_phi_fft(self, zero_avg):
        '''Phi-integrates over 0..2pi or 0..phi. Representation-agnostic (phi axis only).'''
        if self.is_special():
            return self
        len_phi = self.content.shape[1]
        phis = jnp.linspace(0, 2 * jnp.pi * (1 - 1 / len_phi), len_phi)
        content_fft = jnp.fft.fft(self.content, axis=1)
        fftfreq_temp = jit_fftfreq_int(len_phi) * 1j
        fftfreq_temp = fftfreq_temp.at[0].set(jnp.inf)
        out_content_fft = content_fft / fftfreq_temp[None, :] / self.nfp
        out_content = jnp.fft.ifft(out_content_fft, axis=1)
        if not zero_avg:
            out_content -= out_content[:, 0][:, None]
            out_content += phis[None, :] * content_fft[:, 0][:, None] / self.nfp / len_phi
        return ChiPhiFuncPadded(out_content, self.nfp, self.is_comb)

    ''' I.2.1 Phi filters '''

    def filter(self, arg: float, mode: int = 0):
        if self.is_special():
            return self
        if mode != 0:
            return ChiPhiFuncPaddedSpecial(-16)
        arg = jnp.where(arg < 0, jnp.inf, arg)
        len_phi = self.content.shape[1]
        W = jnp.abs(jit_fftfreq_int(len_phi))
        f_signal = jnp.fft.fft(self.content, axis=1)
        cut_f_signal = jnp.where(W[None, :] > arg, 0, f_signal)
        return ChiPhiFuncPadded(jnp.fft.ifft(cut_f_signal, axis=1), self.nfp, self.is_comb)

    def filter_reduced_length(self, arg: int):
        if self.is_special():
            return self
        fft_content = jnp.fft.fft(self.content, axis=1)
        short_fft_content = fft_filter(fft_content, arg * 2, axis=1)
        short_content = jnp.fft.ifft(short_fft_content, axis=1)
        return ChiPhiFuncPadded(short_content, self.nfp, self.is_comb)

    ''' I.3 Shape adjustment '''

    def cap_m(self, m: int):
        '''
        Trims/pads (symmetric, centered) to exactly m+1 chi rows. Unlike
        ChiPhiFunc.cap_m, this always uses centered_resize_content and
        handles both narrowing and widening uniformly (no dead pad_m() call
        whose result gets discarded).

        Only valid on normal-representation objects -- cap_m's m+1 target
        means something different for comb (which should be resized in
        mode_cap terms, not m+1 terms); comb objects are transient and
        should go through comb_to_normal() instead of cap_m() before ever
        reaching code that calls cap_m (e.g. recursion_relations.py).
        '''
        if self.is_special():
            return self
        if self.is_comb:
            raise ValueError(
                'ChiPhiFuncPadded.cap_m() called on a comb-representation object -- '
                'convert via comb_to_normal(target_m) first.'
            )
        return ChiPhiFuncPadded(centered_resize_content(self.content, m + 1), self.nfp)

    def pad_m(self, m: int):
        ''' Alias for pad_chi(m+1), mirroring ChiPhiFunc.pad_m. '''
        return self.pad_chi(m + 1)

    def pad_chi(self, target_chi: int):
        '''
        Mirrors ChiPhiFunc.pad_chi's contract exactly (same error codes for
        parity mismatch / shrinking), implemented via centered_resize_content
        instead of a self+zeros trick.
        '''
        if self.is_special():
            return self
        if self.is_comb:
            raise ValueError(
                'ChiPhiFuncPadded.pad_chi() called on a comb-representation object.'
            )
        len_chi = self.content.shape[0]
        if len_chi % 2 != target_chi % 2:
            return ChiPhiFuncPaddedSpecial(-4)
        if target_chi < len_chi:
            return ChiPhiFuncPaddedSpecial(-12)
        return ChiPhiFuncPadded(centered_resize_content(self.content, target_chi), self.nfp)

    def comb_to_normal(self, target_m: int):
        '''
        Converts a comb-representation (parity-agnostic, step-1 mode
        indexing) object into a normal-representation one at chi width
        target_m+1. Used only by py_sum_parallel's leaf-sum scan machinery,
        once the target order's parity is known statically.

        Derivation: comb row r holds mode r - mode_cap (mode_cap =
        (content.shape[0]-1)//2). Extracting every row whose mode matches
        target_m's parity, in order, gives a dense same-parity (step-2)
        array; centered_resize_content then trims/pads that down to exactly
        target_m+1 rows. The starting offset r0 for "every row matching
        parity p" is the smaller of {0, 1} such that (r0 - mode_cap) % 2
        == p, i.e. r0 = (p + mode_cap) % 2.
        '''
        if self.is_special():
            return self
        if not self.is_comb:
            raise ValueError('comb_to_normal() called on a non-comb ChiPhiFuncPadded.')
        mode_cap = (self.content.shape[0] - 1) // 2
        p = target_m % 2
        r0 = (p + mode_cap) % 2
        same_parity = self.content[r0::2]
        resized = centered_resize_content(same_parity, target_m + 1)
        return ChiPhiFuncPadded(resized, self.nfp, False)

    def get_l2(self):
        if self.nfp==0:
            return(0.)
        elif self.nfp<0:
            return(jnp.inf)
        return get_l2_shared(self.content)
        
    def to_ragged(self):
        '''
        Optional interop escape hatch: converts to an ordinary ChiPhiFunc
        (same content, same nfp). Not used in the main iterate_2 pipeline --
        only for code that must stay ragged regardless of backend, e.g.
        display()/eval()/DESC export. Only valid on normal representation.
        '''
        if self.is_special():
            return ChiPhiFuncSpecial(self.nfp)
        if self.is_comb:
            raise ValueError('to_ragged() called on a comb-representation object.')
        return ChiPhiFunc(self.content, self.nfp)

    
    ''' I.1.5 Output and plotting '''
    def display_content(self, trig_mode=False, colormap_mode=False):
        '''
        Plot the content of a ChiPhiFunc.

        Input: -----

        trig_mode: bool. When True, plot the trig Chi fourier coefficients,
        rather than exponential

        colormap_mode: bool. When True, make colormaps. Otherwise makes line plots.
        '''

        plt.rcParams['figure.figsize'] = [8,3]
        content = self.content
        display_content_shared(content, self.nfp, trig_mode=trig_mode, colormap_mode=colormap_mode)


ChiPhiFuncPaddedSpecial_originals = [ChiPhiFuncPadded(jnp.nan, -i) for i in range(18)]

def ChiPhiFuncPaddedSpecial(error_code: int):
    '''Mirrors ChiPhiFuncSpecial exactly -- see chiphifunc.py.'''
    if error_code > 0 or error_code <= -len(ChiPhiFuncPaddedSpecial_originals):
        return ChiPhiFuncPaddedSpecial_originals[2]
    return ChiPhiFuncPaddedSpecial_originals[-error_code]


tree_util.register_pytree_node(
    ChiPhiFuncPadded, ChiPhiFuncPadded._tree_flatten, ChiPhiFuncPadded._tree_unflatten
)



# Bottom-of-file import to break the cycle:
#   ChiPhiFuncPadded.__add__/__mul__ need isinstance(other, ChiPhiEpsFuncPadded);
#   ChiPhiEpsFuncPadded is defined in chiphiepsfunc_padded.py which imports
#   ChiPhiFuncPadded from this file, so neither can go at the top of the other.
from .chiphiepsfunc_padded import ChiPhiEpsFuncPadded

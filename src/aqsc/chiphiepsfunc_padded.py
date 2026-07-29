'''
chiphiepsfunc_padded.py — ChiPhiEpsFuncPadded and its helpers.

ChiPhiEpsFuncPadded is a power-series-in-epsilon container analogous to
ChiPhiEpsFunc, but its elements are ChiPhiFuncPadded rather than ChiPhiFunc.
It stores two parallel triangular arrays (one per chi-mode parity) with a
shared extraction target width (mode_cap), so traced-index __getitem__ always
returns a statically-shaped result and lax.scan can replace Python-level
unrolling.

See chiphifunc_padded.py for the full design rationale.
'''
import jax.numpy as jnp
from jax import tree_util

from .config import double_precision
from .chiphifunc import ChiPhiFunc, ChiPhiFuncSpecial
from .chiphifunc_padded import ChiPhiFuncPadded, ChiPhiFuncPaddedSpecial, _widths_for_mode_cap
from .chiphiepsfunc import ChiPhiEpsFunc
from .math_utilities import centered_resize_content

if double_precision:
    _complex_dtype = jnp.complex128
else:
    _complex_dtype = jnp.complex64


DEFAULT_MODE_CAP_MARGIN = 2
'''
Default extra headroom (beyond the highest stored order) a freshly
constructed/appended container's mode_cap gets, if the caller doesn't
request a specific mode_cap explicitly. The machine-generated recursion
relations for order-(n+1) quantities (parsed/eval_Ynp1.py, eval_Znp1.py,
MHD_parsed/eval_loop.py, ...) systematically use bounds like "0, n+2" --
i.e. they read up to two orders past what's nominally masked/stored -- so a
margin of 2 covers the patterns observed across the whole generated-code
census. Hand-written call sites that need more should call .resize(...)
explicitly rather than relying on this default; if verification ever shows
this default is insufficient somewhere, it is the one place to raise it.
'''


def _offset_even(n: int, chi_shift: int = 0) -> int:
    '''
    Row offset of order n (even) within the even-order triangular store.

    With chi_shift, every stored order's natural width is widened uniformly
    by chi_shift (order m holds m+1+chi_shift rows), so the offset of even
    order n picks up an extra chi_shift per even order below it (there are
    n//2 of them).
    '''
    k = n // 2
    return k * k + k * chi_shift


def _offset_odd(n: int, chi_shift: int = 0) -> int:
    ''' Row offset of order n (odd) within the odd-order triangular store. '''
    k = (n - 1) // 2
    return k * (k + 1) + k * chi_shift


def _total_even_rows_through(n_even: int, chi_shift: int = 0) -> int:
    ''' Total rows used by even orders 0..n_even inclusive (n_even even, or <0 for none). '''
    if n_even < 0:
        return 0
    k = n_even // 2
    return (k + 1) ** 2 + (k + 1) * chi_shift


def _total_odd_rows_through(n_odd: int, chi_shift: int = 0) -> int:
    ''' Total rows used by odd orders 1..n_odd inclusive (n_odd odd, or <1 for none). '''
    if n_odd < 1:
        return 0
    k = (n_odd - 1) // 2
    return (k + 1) * (k + 2) + (k + 1) * chi_shift


class ChiPhiEpsFuncPadded:
    '''
    ChiPhiEpsFunc-like container specialized for chi-expanding coefficient
    series (X_coef_cp, Y_coef_cp, Z_coef_cp, B_theta_coef_cp, B_psi_coef_cp,
    p_perp_coef_cp, Delta_coef_cp, B_denom_coef_c). Do NOT use this for
    iota_coef/B_alpha_coef (square_eps_series=True in leading_orders.py) --
    those are chi-trivial flux functions (often literal Python/JAX scalars)
    that never hit the chi-padding problem this class exists to solve; keep
    them on plain ChiPhiEpsFunc/scalars even in "padded" backend mode.

    Storage: two parallel triangular arrays (even_content, odd_content),
    each holding only the true natural-width content of its orders,
    concatenated -- see _offset_even/_offset_odd for the closed-form
    order->row mapping. Extraction target width (chi_cap_even/chi_cap_odd)
    is a property of the container (mode_cap), not of any stored order.
    '''

    def __init__(self, chiphifunc_list: list, nfp: int, mode_cap: int = None, len_phi: int = None, chi_shift: int = 0):
        # chi_shift: uniform widening of every order's natural chi width
        # (order n holds n+1+chi_shift rows instead of n+1). 0 for ordinary
        # chi-expanding coefficient series (the default, near-axis
        # regularity). deps() produces chi_shift=+1 because it relabels
        # order-(i+1) content -- whose chi width is one wider and of the
        # opposite parity -- as order i; truncating that back to n+1 (the
        # old behavior) silently dropped a physically-nonzero outermost chi
        # mode. See deps() for details.
        self.nfp = nfp
        self.square_eps_series = False
        self.chi_shift = chi_shift
        n_max = len(chiphifunc_list) - 1
        if mode_cap is None:
            mode_cap = max(n_max, 0) + DEFAULT_MODE_CAP_MARGIN
        mode_cap = max(mode_cap, n_max + chi_shift)

        if len_phi is None:
            len_phi = None
            for item in chiphifunc_list:
                if isinstance(item, (ChiPhiFunc, ChiPhiFuncPadded)) and not item.is_special():
                    len_phi = max(len_phi or 1, item.content.shape[1])
            if len_phi is None:
                # No non-special ChiPhiFunc-like item to infer the phi grid
                # size from (e.g. an empty list, or all-scalar/all-special
                # entries) -- silently defaulting to 1 here would corrupt
                # later .append()s of real data (broadcast_to can't go from
                # N columns down to 1). Caller must pass len_phi explicitly
                # in this case (e.g. leading_orders.py's B_psi_coef_cp =
                # ChiPhiEpsFuncPadded([], nfp, len_phi=len_phi)).
                raise ValueError(
                    'ChiPhiEpsFuncPadded: could not infer len_phi from chiphifunc_list '
                    '(empty, or no non-special ChiPhiFunc-like item present) -- pass '
                    'len_phi explicitly.'
                )

        special = []
        even_pieces = []
        odd_pieces = []
        for n, item in enumerate(chiphifunc_list):
            code, raw = _classify_item(item, n, len_phi, chi_shift)
            special.append(code)
            (even_pieces if n % 2 == 0 else odd_pieces).append(raw)

        even_content = (
            jnp.concatenate(even_pieces, axis=0) if even_pieces
            else jnp.zeros((0, len_phi), dtype=_complex_dtype)
        )
        odd_content = (
            jnp.concatenate(odd_pieces, axis=0) if odd_pieces
            else jnp.zeros((0, len_phi), dtype=_complex_dtype)
        )
        self._init_raw(even_content, odd_content, nfp, n_max, mode_cap, len_phi, special, chi_shift)

    def _init_raw(self, even_content, odd_content, nfp, n_max, mode_cap, len_phi, special, chi_shift=0):
        self.nfp = nfp
        self.square_eps_series = False
        self.chi_shift = chi_shift
        self.n_max = n_max
        self.mode_cap = mode_cap
        self.chi_cap_even, self.chi_cap_odd = _widths_for_mode_cap(mode_cap)
        self.len_phi = len_phi
        self.special = special
        self.even_content = even_content
        self.odd_content = odd_content

    @classmethod
    def _from_raw(cls, even_content, odd_content, nfp, n_max, mode_cap, len_phi, special, chi_shift=0):
        self = object.__new__(cls)
        self._init_raw(even_content, odd_content, nfp, n_max, mode_cap, len_phi, list(special), chi_shift)
        return self

    def _tree_flatten(self):
        children = (self.even_content, self.odd_content)
        aux_data = {
            'nfp': self.nfp, 'n_max': self.n_max, 'mode_cap': self.mode_cap,
            'len_phi': self.len_phi, 'special': tuple(self.special),
            'chi_shift': self.chi_shift,
        }
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        even_content, odd_content = children
        return cls._from_raw(
            even_content, odd_content, nfp=aux_data['nfp'], n_max=aux_data['n_max'],
            mode_cap=aux_data['mode_cap'], len_phi=aux_data['len_phi'],
            special=list(aux_data['special']), chi_shift=aux_data['chi_shift'],
        )

    def __str__(self):
        return (
            f'ChiPhiEpsFuncPadded(n_max={self.n_max}, mode_cap={self.mode_cap}, '
            f'chi_shift={self.chi_shift}, nfp={self.nfp}, special={self.special})'
        )

    ''' Resizing '''

    def resize(self, new_mode_cap: int):
        '''
        Returns a new container with mode_cap widened to at least
        new_mode_cap (never shrinks). O(1) -- mode_cap only affects
        extraction-time target width, not the stored triangular data, so no
        re-embedding of existing content is needed.
        '''
        new_mode_cap = max(new_mode_cap, self.mode_cap)
        return ChiPhiEpsFuncPadded._from_raw(
            self.even_content, self.odd_content, self.nfp, self.n_max,
            new_mode_cap, self.len_phi, self.special, self.chi_shift,
        )

    ''' Indexing '''

    def __getitem__(self, index):
        if isinstance(index, int):
            # Concrete (static Python int) index: the overwhelming majority
            # of extractions in the machine-generated code, since n is
            # always a static Python int and most inner loop variables in
            # non-vectorized (static-unrolled) sums are static too. Returns
            # the *natural* width (index+1), matching ragged ChiPhiEpsFunc
            # exactly -- NOT padded to chi_cap_even/chi_cap_odd. Padding to
            # a container-wide constant here was an earlier design mistake:
            # concrete indices never need JAX shape-uniformity (that's only
            # a lax.scan requirement), and padding anyway meant every
            # extraction was wider than its true mathematical content, and
            # __mul__'s width-additive convolution compounds that inflation
            # through any chain of operations outside a vectorized leaf sum
            # (which is most of the machine-generated code -- only the
            # specific py_sum_parallel-vectorized leaf sums use chi_cap, via
            # the traced path below). ChiPhiFuncPadded's __add__/__mul__
            # already reconcile mismatched widths dynamically (same
            # centered-resize-to-the-max logic ragged ChiPhiFunc uses), so
            # returning natural width here makes padded arithmetic track
            # ragged's own width bookkeeping term-by-term, not just after a
            # final narrowing step.
            if index < 0 or index > self.n_max:
                return ChiPhiFuncPaddedSpecial(0)
            code = self.special[index]
            if code != 0:
                return ChiPhiFuncPaddedSpecial(code)
            even = (index % 2 == 0)
            store = self.even_content if even else self.odd_content
            offset = (
                _offset_even(index, self.chi_shift) if even
                else _offset_odd(index, self.chi_shift)
            )
            size = index + 1 + self.chi_shift
            raw = store[offset: offset + size]
            return ChiPhiFuncPadded(raw, self.nfp)

        # Traced index -- only reachable inside a lax.scan/fori_loop body
        # (see math_utilities.py's leaf-sum vectorization, and the module
        # docstring's "Backend selection" section for why this needs a
        # different representation from the concrete-index case). A traced
        # index's parity is not known at trace time, and centered_resize_
        # content-based reconciliation between chi_cap_even and chi_cap_odd
        # doesn't work (they differ in width *parity* by construction --
        # one is odd-width, one is even-width -- so symmetric padding
        # between them always shifts one side by half a row, silently
        # misaligning modes). The only correct fix is a parity-agnostic
        # (comb, step-1 mode indexing) result -- see ChiPhiFuncPadded's
        # is_comb representation. Returns a comb ChiPhiFuncPadded with
        # mode_cap = self.mode_cap (comb width 2*mode_cap+1); the caller
        # (py_sum_parallel) is responsible for converting the final
        # accumulated result back to normal representation via
        # .comb_to_normal(target_m) once the target parity is known.
        if self.chi_shift != 0:
            # The traced-index (comb) path is only exercised inside
            # py_sum_parallel's lax.scan over ordinary chi-expanding
            # coefficient series, which always have chi_shift==0. deps()-style
            # shifted-width series (chi_shift!=0) only ever flow through
            # concrete-index geometry code (contravariant_basis_eps etc.), so
            # this branch should never be reached for them; the comb row<->mode
            # mapping below assumes width index+1 and would misalign otherwise.
            raise NotImplementedError(
                'ChiPhiEpsFuncPadded: traced-index __getitem__ is not supported '
                'for chi_shift != 0 (shifted-width series only support '
                'concrete-index access).'
            )
        n_max = self.n_max  # static
        mode_cap = self.mode_cap  # static
        comb_width = 2 * mode_cap + 1
        n_even_rows = self.even_content.shape[0]  # static

        valid = (index >= 0) & (index <= n_max)
        safe_index = jnp.clip(index, 0, max(n_max, 0))
        is_even = (safe_index % 2 == 0)

        offset_if_even = (safe_index // 2) ** 2
        k_odd = (safe_index - 1) // 2
        offset_if_odd = k_odd * (k_odd + 1)
        # Row offset into the concatenation [even_content; odd_content].
        combined_offset = jnp.where(is_even, offset_if_even, n_even_rows + offset_if_odd)
        combined_store = jnp.concatenate([self.even_content, self.odd_content], axis=0)

        rel_modes = jnp.arange(comb_width) - mode_cap  # static: -mode_cap..mode_cap
        # natural_row_local = (mode + index) // 2 is order `index`'s own row
        # for this mode, valid only where the mode has index's parity and
        # lies within [-index, index]; row_valid masks out everything else,
        # so it doesn't matter what garbage physical_row happens to compute
        # to (or what jnp.take fetches there) outside that window.
        parity_ok = (rel_modes + safe_index) % 2 == 0
        in_range = jnp.abs(rel_modes) <= safe_index
        row_valid = parity_ok & in_range & valid
        natural_row_local = (rel_modes + safe_index) // 2
        physical_row = combined_offset + natural_row_local
        gathered = jnp.take(combined_store, physical_row, axis=0, mode='fill', fill_value=0.0)
        result_content = jnp.where(row_valid[:, None], gathered, 0.0)
        return ChiPhiFuncPadded(result_content, self.nfp, True)

    ''' List-like operations, mirroring ChiPhiEpsFunc '''

    def append(self, item):
        # _classify_item handles the family check (including the
        # special-ragged-placeholder exception -- see its docstring); no
        # need to duplicate/pre-empt that logic here.
        n = self.n_max + 1
        code, raw = _classify_item(item, n, self.len_phi, self.chi_shift)
        even = (n % 2 == 0)
        even_content = self.even_content
        odd_content = self.odd_content
        if even:
            even_content = jnp.concatenate([even_content, raw], axis=0)
        else:
            odd_content = jnp.concatenate([odd_content, raw], axis=0)
        new_mode_cap = max(self.mode_cap, n + self.chi_shift)
        return ChiPhiEpsFuncPadded._from_raw(
            even_content, odd_content, self.nfp, n, new_mode_cap, self.len_phi,
            self.special + [code], self.chi_shift,
        )

    def zero_append(self, n=1):
        ''' No-op, matching ChiPhiEpsFunc.zero_append (see its docstring: historical). '''
        return self

    def mask(self, n):
        if n == float('inf') or n >= self.n_max:
            return self
        n = max(n, -1)
        if n < 0:
            return ChiPhiEpsFuncPadded._from_raw(
                jnp.zeros((0, self.len_phi), dtype=_complex_dtype),
                jnp.zeros((0, self.len_phi), dtype=_complex_dtype),
                self.nfp, -1, self.mode_cap, self.len_phi, [], self.chi_shift,
            )
        largest_even = n if n % 2 == 0 else n - 1
        largest_odd = n if n % 2 == 1 else n - 1
        even_rows = _total_even_rows_through(largest_even, self.chi_shift)
        odd_rows = _total_odd_rows_through(largest_odd, self.chi_shift)
        return ChiPhiEpsFuncPadded._from_raw(
            self.even_content[:even_rows], self.odd_content[:odd_rows],
            self.nfp, n, self.mode_cap, self.len_phi, self.special[:n + 1],
            self.chi_shift,
        )

    def get_order(self):
        return self.n_max

    ''' Derivatives '''

    def prepend_zero(self):
        '''
        Shift the series up by one eps-order by prepending a zero term.
        Inverse of deps (up to the order-number factors in deps).

        result[0] = 0, result[i+1] = self[i] for i = 0..n_max.

        Like deps, this swaps parity stores: result order i+1 (which lives
        in the opposite parity store from order i) comes from self[i].
        The source natural width i+1 is widened to target width i+2 via
        centered_resize_content (zero-padded on both sides).
        '''
        new_n_max = self.n_max + 1
        even_pieces, odd_pieces, special = [], [], []

        # order 0 is always zero
        special.append(0)
        even_pieces.append(jnp.zeros((1, self.len_phi), dtype=_complex_dtype))

        for i in range(self.n_max + 1):
            dst = i + 1           # destination order in result
            src_width = i + 1     # natural width of source at order i
            target_width = dst + 1  # = i + 2
            src_code = self.special[i]
            special.append(src_code)
            if src_code != 0:
                raw = jnp.zeros((target_width, self.len_phi), dtype=_complex_dtype)
            else:
                src_even = (i % 2 == 0)
                store = self.even_content if src_even else self.odd_content
                offset = _offset_even(i) if src_even else _offset_odd(i)
                raw_src = store[offset: offset + src_width]
                # Widen i+1 → i+2 rows (zero-pads the outermost chi mode).
                raw = centered_resize_content(raw_src, target_width)
            (even_pieces if dst % 2 == 0 else odd_pieces).append(raw)

        even_content = (
            jnp.concatenate(even_pieces, axis=0) if even_pieces
            else jnp.zeros((0, self.len_phi), dtype=_complex_dtype)
        )
        odd_content = (
            jnp.concatenate(odd_pieces, axis=0) if odd_pieces
            else jnp.zeros((0, self.len_phi), dtype=_complex_dtype)
        )
        return ChiPhiEpsFuncPadded._from_raw(
            even_content, odd_content, self.nfp, new_n_max,
            self.mode_cap, self.len_phi, special,
        )

    def deps(self):
        '''
        Derivative with respect to eps (the near-axis expansion parameter).

        For a series f = sum_n f_n * eps^n, deps(f)_i = (i+1) * f_{i+1}.

        This shifts the epsilon order by -1. Result order i takes source
        order (i+1)'s content verbatim (scaled by i+1). That content has
        chi width (i+2)+self.chi_shift and the chi parity of order i+1 --
        one wider, and opposite parity, relative to what a natural
        chi_shift=0 slot for order i would hold. Rather than truncate it
        back to i+1 (the old behavior, which silently dropped the
        physically-nonzero outermost chi mode ±(i+1) and misaligned the
        rest by a half-row parity flip), the result simply carries
        chi_shift + 1: every order is stored at its true, one-wider width,
        so no chi information is lost.
        '''
        new_n_max = self.n_max - 1
        new_shift = self.chi_shift + 1
        even_pieces, odd_pieces, special = [], [], []
        for i in range(new_n_max + 1):
            src = i + 1
            # Width of source order (i+1), which becomes result order i's
            # width verbatim -- exactly the natural width of order i under
            # the incremented chi_shift, so no resize is needed.
            width = src + 1 + self.chi_shift  # = (i + 2) + self.chi_shift
            src_code = self.special[src]
            special.append(src_code)
            if src_code != 0:
                raw = jnp.zeros((width, self.len_phi), dtype=_complex_dtype)
            else:
                src_even = (src % 2 == 0)
                store = self.even_content if src_even else self.odd_content
                offset = (
                    _offset_even(src, self.chi_shift) if src_even
                    else _offset_odd(src, self.chi_shift)
                )
                raw = store[offset: offset + width] * (i + 1)
            (even_pieces if i % 2 == 0 else odd_pieces).append(raw)
        even_content = (
            jnp.concatenate(even_pieces, axis=0) if even_pieces
            else jnp.zeros((0, self.len_phi), dtype=_complex_dtype)
        )
        odd_content = (
            jnp.concatenate(odd_pieces, axis=0) if odd_pieces
            else jnp.zeros((0, self.len_phi), dtype=_complex_dtype)
        )
        return ChiPhiEpsFuncPadded._from_raw(
            even_content, odd_content, self.nfp, new_n_max,
            max(self.mode_cap, new_n_max + new_shift), self.len_phi, special,
            new_shift,
        )

    def dchi(self):
        return self._map_orders(lambda f: f.dchi())

    def dphi(self):
        return self._map_orders(lambda f: f.dphi())

    def _map_orders(self, fn):
        ''' Shared helper for dchi/dphi: apply fn order-by-order, rebuild triangular stores.
        Preserves chi_shift (fn is width-preserving: dchi/dphi/negate/scalar-mul). '''
        even_pieces, odd_pieces, special = [], [], []
        for n in range(self.n_max + 1):
            width = n + 1 + self.chi_shift
            code = self.special[n]
            special.append(code)
            if code != 0:
                natural = jnp.zeros((width, self.len_phi), dtype=_complex_dtype)
            else:
                even = (n % 2 == 0)
                store = self.even_content if even else self.odd_content
                offset = (
                    _offset_even(n, self.chi_shift) if even
                    else _offset_odd(n, self.chi_shift)
                )
                natural_in = store[offset: offset + width]
                if self.chi_shift == 0:
                    # Original behavior: work at the container's chi_cap
                    # (parity matches the natural width when chi_shift==0),
                    # then trim back.
                    chi_cap = self.chi_cap_even if even else self.chi_cap_odd
                    work_in = centered_resize_content(natural_in, chi_cap)
                else:
                    # Shifted widths can differ in parity from chi_cap_*;
                    # dchi/dphi are width-preserving, so just work at the
                    # order's own (parity-correct) natural width.
                    work_in = natural_in
                out = fn(ChiPhiFuncPadded(work_in, self.nfp))
                natural = centered_resize_content(out.content, width) if not out.is_special() else jnp.zeros((width, self.len_phi), dtype=_complex_dtype)
            (even_pieces if n % 2 == 0 else odd_pieces).append(natural)
        even_content = jnp.concatenate(even_pieces, axis=0) if even_pieces else jnp.zeros((0, self.len_phi), dtype=_complex_dtype)
        odd_content = jnp.concatenate(odd_pieces, axis=0) if odd_pieces else jnp.zeros((0, self.len_phi), dtype=_complex_dtype)
        return ChiPhiEpsFuncPadded._from_raw(
            even_content, odd_content, self.nfp, self.n_max, self.mode_cap, self.len_phi, special,
            self.chi_shift,
        )

    ''' Arithmetic (operating on the whole series, mirroring ChiPhiEpsFunc) '''

    def __neg__(self):
        return self._map_orders(lambda f: -f)

    def __add__(self, other):
        if isinstance(other, ChiPhiEpsFuncPadded):
            n_max = max(self.n_max, other.n_max)
            items = [self[i] + other[i] for i in range(n_max + 1)]
            mode_cap = max(self.mode_cap, other.mode_cap)
            # Same-width addition preserves the shift; a mismatch (only
            # possible if the two operands carry different chi_shift) results
            # in the wider width, and same-parity reconciliation is handled
            # per-item by ChiPhiFuncPadded.__add__ (a parity mismatch there
            # surfaces as a special error, never a silent misalignment).
            chi_shift = max(self.chi_shift, other.chi_shift)
            return ChiPhiEpsFuncPadded(items, self.nfp, mode_cap=mode_cap, len_phi=self.len_phi, chi_shift=chi_shift)
        if isinstance(other, (ChiPhiFunc, ChiPhiEpsFunc)):
            raise TypeError(
                'ChiPhiEpsFuncPadded combined with a ragged ChiPhiFunc/ChiPhiEpsFunc -- '
                'likely a hardcoded construction site producing the wrong family.'
            )
        # Add to leading (order-0) term only, matching ChiPhiEpsFunc.__add__.
        leading = self[0] + other
        return self._replace_order(0, leading)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -(self - other)

    def __mul__(self, other):
        if isinstance(other, ChiPhiEpsFuncPadded):
            n_max = self.n_max + other.n_max
            items = []
            for k in range(n_max + 1):
                acc = ChiPhiFuncPaddedSpecial(0)
                for i in range(max(0, k - other.n_max), min(self.n_max, k) + 1):
                    acc = acc + self[i] * other[k - i]
                items.append(acc)
            # Convolving widths adds them: order-k product has width
            # (i+1+sa)+(k-i+1+sb)-1 = k+1+(sa+sb), i.e. chi_shift = sa+sb.
            chi_shift = self.chi_shift + other.chi_shift
            return ChiPhiEpsFuncPadded(items, self.nfp, len_phi=self.len_phi, chi_shift=chi_shift)
        if isinstance(other, (ChiPhiFunc, ChiPhiEpsFunc)):
            raise TypeError(
                'ChiPhiEpsFuncPadded combined with a ragged ChiPhiFunc/ChiPhiEpsFunc -- '
                'likely a hardcoded construction site producing the wrong family.'
            )
        return self._map_orders(lambda f: f * other)

    def __rmul__(self, other):
        return self * other

    def _replace_order(self, index, item):
        items = [self[i] for i in range(self.n_max + 1)]
        items[index] = item
        return ChiPhiEpsFuncPadded(items, self.nfp, mode_cap=self.mode_cap, len_phi=self.len_phi, chi_shift=self.chi_shift)

    ''' Interop '''

    def to_ragged(self):
        ''' Optional escape hatch, see ChiPhiFuncPadded.to_ragged. '''
        items = []
        for n in range(self.n_max + 1):
            code = self.special[n]
            items.append(ChiPhiFuncSpecial(code) if code != 0 else self[n].to_ragged())
        return ChiPhiEpsFunc(items, self.nfp, False)


    def eval(self, psi, chi=None, phi=None, n_max=float('inf')):
        return self.to_ragged().eval(psi, chi, phi, n_max=n_max)

    def eval_eps(self, eps, chi=None, phi=None, n_max=float('inf')):
        '''
        Evaluate the power series at (eps, chi, phi). Non-hot-path: converts
        to ragged and reuses ChiPhiEpsFunc.eval_eps (see ARCHITECTURE_PLAN).
        '''
        return self.to_ragged().eval_eps(eps, chi, phi, n_max=n_max)

    def get_l2_order_by_order(self):
        return self.to_ragged().get_l2_order_by_order()

    def get_max_order_by_order(self, len_chi:int=100, len_phi:int=100):
        return self.to_ragged().get_max_order_by_order(len_chi=len_chi, len_phi=len_phi)

    def get_norm_order_by_order(self, **kwargs):
        '''
        Calculate the L2 norm of the term at each order.
        (squared integral over chi and phi.)

        :param kwargs: Parameters of jnp.linalg.norm
        '''
        return self.to_ragged().get_norm_order_by_order(**kwargs)

    @classmethod
    def zeros_like(cls, other):
        return cls(
            [ChiPhiFuncPaddedSpecial(0)] * (other.get_order() + 1),
            other.nfp, mode_cap=other.mode_cap if hasattr(other, 'mode_cap') else None,
        )


def _classify_item(item, n: int, len_phi: int, chi_shift: int = 0):
    '''
    Given a raw list element (as passed to ChiPhiEpsFuncPadded's constructor
    or .append()) destined for order n, returns (special_code, content)
    where content is always a (n+1+chi_shift, len_phi) array (zeros for
    special/error items -- specialness is tracked via special_code, not via
    the stored array, so it never contaminates arithmetic with NaN).

    chi_shift widens every order's natural width uniformly (see
    ChiPhiEpsFuncPadded.__init__ / deps): order n holds n+1+chi_shift rows.
    '''
    natural_width = n + 1 + chi_shift
    natural_shape = (natural_width, len_phi)
    if isinstance(item, ChiPhiFuncPadded):
        if item.is_special():
            return item.nfp, jnp.zeros(natural_shape, dtype=_complex_dtype)
        width = item.content.shape[0]
        if width > natural_width:
            raise ValueError(
                f'ChiPhiEpsFuncPadded: item at order {n} has content.shape[0]='
                f'{width}, wider than the maximum of {natural_width} '
                f'(n+1+chi_shift, chi_shift={chi_shift}).'
            )
        if width % 2 != natural_width % 2:
            raise ValueError(
                f'ChiPhiEpsFuncPadded: item at order {n} has content.shape[0]='
                f'{width}, wrong parity for order {n} (expected '
                f'{natural_width%2}-parity row count, i.e. matching '
                f'{natural_width} mod 2, chi_shift={chi_shift}).'
            )
        content = item.content
        if width < natural_width:
            # Narrower-than-natural-but-same-parity is a real, intentional
            # pattern in this codebase, not an error: e.g. leading_orders.py
            # appends B_theta_coef_cp[2] as just its chi-independent (width
            # 1) average, to be widened by later arithmetic reconciliation
            # or replaced by a full-width value from a subsequent iterate_2
            # call (equilibrium.py's mask-then-reappend dance). Ragged
            # ChiPhiEpsFunc.append() never validates width at all, so this
            # already works silently there -- widening here via the same
            # centered formula ordinary arithmetic reconciliation would use
            # anyway keeps padded behavior identical, just done eagerly
            # instead of lazily.
            content = centered_resize_content(content, natural_width)
        if content.shape[1] != len_phi:
            content = jnp.broadcast_to(content, (content.shape[0], len_phi))
        return 0, content
    if isinstance(item, ChiPhiFunc):
        # A genuinely-special ragged placeholder (e.g. leading_orders.py's
        # ChiPhiEpsFunc([ChiPhiFuncSpecial(0), ChiPhiFuncSpecial(0)], ...)
        # idiom, reused verbatim under the padded backend) is legitimate --
        # same reasoning as ChiPhiFuncPadded._coerce_operand. A non-special
        # ragged ChiPhiFunc is always a real construction-site bug.
        if item.is_special():
            return item.nfp, jnp.zeros(natural_shape, dtype=_complex_dtype)
        raise TypeError(
            f'ChiPhiEpsFuncPadded: item at order {n} is a ragged ChiPhiFunc, not '
            'ChiPhiFuncPadded -- construction site producing the wrong family.'
        )
    if jnp.ndim(item) == 0:
        if natural_width % 2 == 0:
            raise ValueError(
                f'ChiPhiEpsFuncPadded: a bare scalar was given for order {n} '
                f'(natural width {natural_width}, chi_shift={chi_shift}), which '
                'has no mode-0 (chi-independent) row -- a scalar entry only '
                'makes sense where the natural width is odd.'
            )
        content = jnp.zeros(natural_shape, dtype=_complex_dtype)
        # Mode-0 row is the center of an odd-width buffer.
        content = content.at[(natural_width - 1) // 2, :].set(item)
        return 0, content
    raise TypeError(
        f'ChiPhiEpsFuncPadded: item at order {n} is neither ChiPhiFuncPadded, '
        f'ChiPhiFunc, nor a scalar: {type(item)}.'
    )


tree_util.register_pytree_node(
    ChiPhiEpsFuncPadded,
    ChiPhiEpsFuncPadded._tree_flatten,
    ChiPhiEpsFuncPadded._tree_unflatten,
)

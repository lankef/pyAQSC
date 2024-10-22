# ChiPhiEpsFunc API

As introduced in [data structure](data-structure.md), `ChiPhiEpsFunc` is a list-like class containing scalars and `ChiPhiFunc`'s. It is the primary data structure pyAQSC stores results in. 

`ChiPhiEpsFunc` manages the power-Fourier series:
$$
F(\psi, \chi, \phi)\\
=\sum_{n=0}^{n_{max}}\epsilon^nF_n(\chi,\phi) (\epsilon\equiv\sqrt{\psi})\\
=\sum_{n=0}^{n_{max}}\epsilon^n\sum_{m=0|1}^n e^{im} F_{n,m}(\phi) + e^{-im} F_{n,-m}(\phi).
$$

## Class attributes
### `self.chiphifunc_list : int` (list of traced)
The list containing all power coefficients. A power coefficient can be a scalar or a `ChiPhiFunc.`
### `self.nfp : int` (static)
Required number of field period of the `ChiPhiFunc`'s in `self.chiphifunc_list`.
### `self.square_eps_series : bool` (static)
Whether the object represents a series containing only even powers of $\epsilon$. One example is:
$$
B_\alpha(\psi) = \sum_{n=0}^\infty\epsilon^{2n}B_{\alpha n}
$$

`ChiPhiEpsFunc` supports `__getitem__` and `append`, but **not** `__setitem__`. `ChiPhiEpsFunc[n]` Extracts the $n$-th order power coefficient, which may be a scalar or a `ChiPhiFunc`.

## Constructor
`aqsc.ChiPhiEpsFunc(self, list:list, nfp:int, check_consistency:bool=False)`
Parameters:

- `chiphifunc_list : list` (list of traced) - The list containing all power coefficients. A power coefficient can be a scalar or a `ChiPhiFunc.`
- `nfp : int` (static) - Required number of field period of the `ChiPhiFunc`'s in `self.chiphifunc_list`.
- `check_consistency : bool` - Runs `aqsc.ChiPhiEpsFunc.check_nfp_consistency()` if set to `True`. The method checks for items with inconsistent `nfp` and replaes them with `ChiPhiFunc(nfp=-14)`.

## Functions

### `aqsc.ChiPhiEpsFunc.__getitem__(self, index)`

Implements `ChiPhiEpsFunc[n]`. Finds the $n$-th order power coefficient. Returns `ChiPhiFunc(nfp=0)` for negative or out-of-bound `index`.

Parameters:

- `index : int` (static) - index of item. 

Returns:

- A `ChiPhiFunc` or a scalar.

### `aqsc.ChiPhiEpsFunc.append(item)`
Returns a new ChiPhiEpsFunc with a new item appended to the end. Checks consistency. Does not modify `self`. Returns a new `ChiPhiEpsFunc`.

Parameters:

- `item` (traced) - Power coefficient to append.

Returns: 
- A new `ChiPhiEpsFunc`.

### `aqsc.ChiPhiEpsFunc.eval(psi, chi=0, phi=0, n_max=float('inf'))`
A vectorized function that evaluates the value of the `ChiPhiEpsFunc`, $f(\psi, \chi, \phi)$, at given `psi`, `chi` and `phi`. The $\phi$ interpolation is performed by `jax.numpy.interp`. When `sq_eps_series` is set to `True`, the series is treated as an even power series, which $\bar\iota$ and $B_\alpha$ are expanded as.
$$
F(\psi, \chi, \phi) = \sum_{n=0}^{n_{max}}F_n(\chi, \phi)\epsilon^{2n}.
$$

Parameters:

- `psi, chi, phi : array or scalar` (traced) - $\psi$, $\chi$, $\phi$'s to evaluate at.
- `n_max=float('inf')` (static) - The order to evaluate to. If larger than highest available $n$, evaluates to highest available $n$.

Returns: 
- An `jnp.array` evaluation result.

### `aqsc.ChiPhiEpsFunc.deps()`, `aqsc.ChiPhiEpsFunc.dchi()`, `aqsc.ChiPhiEpsFunc.dphi()`
Calculates the $\epsilon$, $\chi$ or $\phi$ derivative of a `ChiPhiEpsFunc`.

Returns: 
- A new `aqsc.ChiPhiEpsFunc`.

### `aqsc.ChiPhiEpsFunc.eval_eps(eps, chi=0, phi=0, n_max=float('inf'))`
A vectorized function that evaluates the value of the `ChiPhiEpsFunc`, $f(\epsilon, \chi, \phi)$, at given `psi`, `chi` and `phi`. The $\phi$ interpolation is performed by `jax.numpy.interp`. When `sq_eps_series` is set to `True`, the series is treated as an even power series, which $\bar\iota$ and $B_\alpha$ are expanded as.
$$
F(\psi, \chi, \phi) = \sum_{n=0}^{n_{max}}F_n(\chi, \phi)\epsilon^{2n}.
$$

Parameters:

- `eps, chi, phi : array or scalar` (traced) - $\psi$, $\chi$, $\phi$'s to evaluate at.
- `n_max=float('inf')` (static) - The order to evaluate to. If larger than highest available $n$, evaluates to highest available $n$.

Returns: 
- An `jnp.array` evaluation result.

### `aqsc.ChiPhiEpsFunc.deps()`, `aqsc.ChiPhiEpsFunc.dchi()`, `aqsc.ChiPhiEpsFunc.dphi()`
Calculates the $\epsilon$, $\chi$ or $\phi$ derivative of a `ChiPhiEpsFunc`.

Returns: 
- A new `aqsc.ChiPhiEpsFunc`.

        
<!-- ### `aqsc.ChiPhiEpsFunc.zero_append(n=1)` **DEPRECIATED**
Does nothing. Previously appends one or more `ChiPhiFunc(nfp=0)` at the end of `self`. Does not modify `self`. Returns the original `ChiPhiEpsFunc`. Was created to for use with `mask` to evaluate expressions setting an unknown, highest order term in an `ChiPhiEpsFunc` to zero, back when out-of-index returns a unique `ChiPhiFuncSpecial`, rather than 0 to check for mistakes in governing equations. This is no longer needed because now out-of-index items are `ChiPhiFunc(nfp=0)`.

Parameters:

- `n : int` (static) - Does nothing

Returns: 
- The original `ChiPhiEpsFunc`. -->


### `aqsc.ChiPhiEpsFunc.mask(n)`
Produces a `ChiPhiEpsFunc` from self containing power coefficients up to order $n$. When $n$ is lower than the highest currently known order, returns a `ChiPhiEpsFunc` containing a sublist of `self.chiphifunc_list`. When $n$ is higher than the highest currently known order, fill in the unknown elements with `ChiPhiFunc(nfp=0)`. 

Does not modify `self`. Returns a new `ChiPhiEpsFunc`.

Parameters:

- `n : int` (static) - Maximum order to extract.

Returns: 
- A new `ChiPhiEpsFunc`.

### `aqsc.ChiPhiEpsFunc.get_order()`
Returns the highest known order $n$ for a `ChiPhiEpsFunc`. Equivalent to `len(self.chiphifunc_list)`.

Returns:

- `n : int` - The highest known order $n$.

### `aqsc.ChiPhiEpsFunc.get_max_order_by_order(n)`
Calculating the maximum of the absolute value of the function represented by the Fourier coefficient of each order.

Parameters: 

- `n_chi, n_phi : int` - The grid size for evaluation.

Returns:

- A list of real scalars.

### `aqsc.ChiPhiEpsFunc.zeros_like(other)`
Produces a `ChiPhiFunc(nfp=0)`-filled ChiPhiEpsFunc with the same $n$ as another `ChiPhiEpsFunc`.

Returns:

- A `ChiPhiEpsFunc`.

### `aqsc.ChiPhiEpsFunc.__str__(self)`
Overrides str(aqsc.ChiPhiEpsFunc). Produces a string summarizing all elements in `self.chiphifunc_list`.

### `aqsc.ChiPhiEpsFunc.to_content_list()`
Converts a ChiPhiEpsFunc to a list of `int`'s, `array`'s and `string`'s that can be saved and loaded by `np.save()`. For `aqsc.Equilibrium.save(file_name)`.

### `aqsc.ChiPhiEpsFunc.from_content_list(content_list, nfp)` 
Loads a ChiPhiEpsFunc from a list of `int`'s, `array`'s and `string`'s generated by `aqsc.ChiPhiEpsFunc.to_content_list()`. For `aqsc.Equilibrium.load(file_name)`.
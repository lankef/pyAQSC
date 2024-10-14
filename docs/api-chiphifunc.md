# ChiPhiFunc API

As introduced in [data structure](data-structure.md), `ChiPhiFunc` represents
$$
F_n(\chi, \phi) = \sum_{m=0|1}^n e^{im} F_{n,m}(\phi) + e^{-im} F_{n,-m}(\phi)
$$
and handles numerical operations on/between such series. It is a traced class to support auto-differentiation by JAX.

**Numerical operations between ChiPhiFunc's with different grid number or nfp are not supported.**

## Class Attributes
### `self.content : jax.numpy.array` (traced)
  
  A `complex128` or `complex64` array storing the Fourier series coefficients.
   
  `axis=0` corresponds to represents even/odd $\phi$ mode number $m$ due to regularization. Its length is n+1 for an order $n$ term in a power-Fourier series. For more detailed discussions on regularization, see [Ref. 5](https://doi.org/10.1017/S0022377818001289)

  `axis=1` stores $\phi$-dependent coefficients $F_{n,m}$ on $n$ uniformly spaced grid points over one field period, $\phi = 0, \frac{2\pi}{n_{fp}}\frac{1}{n}, ..., \frac{2\pi}{n_{fp}}\frac{(n-1)}{n}$, where $n_{fp}$ is the number of field period.

      self.content = \
        [                           # [
            [Chi_coeff_-n(phi)],    #     [Chi_coeff_-n(phi)],
            ...                     #     ...
            [Chi_coeff_-2(phi)],    #     [Chi_coeff_-1(phi)],
            [const(phi)],           #     [Chi_coeff_1(phi)],
            [Chi_coeff_2(phi)],     #     ...
            ...                     #     [Chi_coeff_n(phi)]
            [Chi_coeff_n(phi)]      # ] for odd n
        ] for even n
  
### `self.nfp : int` (static)
Number of field period when >0, and error code when <=0. See [Data structure](data-structure.md) for the list of all implemented error codes.

## Constructor 
### `aqsc.ChiPhiFunc(content:jax.numpy.ndarray, nfp:int, trig_mode:bool=False)`
Parameters:

- `content : jax.numpy.array` (traced)
- `nfp : int` (static)
- `trig_mode : bool`: When set to `True`, treat provided `content` as trigonometric Fourier coefficients:
  
      [
          [Chi_sin_coeff_{m}(phi)],
          [Chi_sin_coeff_{m-1}(phi)],
          ...,
          [Chi_cos_coeff__{m-1}(phi)],
          [Chi_cos_coeff__{m}(phi)],
      ]
  
  and converts to esponential coefficients as part of the initialization:

### `aqsc.ChiPhiFuncSpecial(error_code:int)`
Creates a `ChiPhiFunc` with non-positive `self.nfp`. Its `self.content` will be `np.nan`.

Parameters:

- `int error_code`: the nfp of said `ChiPhiFunc`. If `error_code`>0, it will be defaulted to -2.

## Arithmetic operators
`+, -` can be performed between 
 - two even `ChiPhiFunc`'s
 - two odd `ChiPhiFunc`'s, 
 - an even `ChiPhiFunc` and a scalar
 - any `ChiPhiFunc` with a `ChiPhiFunc(nfp<=0)` (see note on error handling)

`*` can be performed between any combination of even/odd `ChiPhiFunc`'s and scalars. Internally, it is a convolution along `axis=0`.

`**` can only be performed to a static integer power. Very high powers are not advised as it's internally a series of convolutions.

`/` can only be performed when the divisor is scalar or $\chi$-independent (`divisor_ChiPhiFunc.content.shape[0]==1`).

`@` can only be performed between a `(n,len_chi)` array and a `ChiPhiFunc` with `len_chi` $\chi$ modes. This performs a matrix operation on the $\chi$ components of the `ChiPhiFunc`.

`my_ChiPhiFunc[m]` returns the $m$-th $\chi$ harmonic of `my_ChiPhiFunc` as a new `ChiPhiFunc`.

### `aqsc.ChiPhiFunc.exp()` 
Calculates $e^{F_n(\phi)}$. Only supports `ChiPhiFunc` with no $\chi$ dependence.

## Functions for derivatives, integrals and FFT 

### `aqsc.ChiPhiFunc.dchi(order:int=1)` 
Takes the $\chi$ derivative and returns a new `ChiPhiFunc.`

Parameters: 
- `order : int` (static) - Order of derivative.

### `aqsc.ChiPhiFunc.antid_chi()` 
Takes the $\chi$ anti-derivative and returns a new `ChiPhiFunc.`

### `aqsc.ChiPhiFunc.dphi(order:int=1, mode=0)` 
Takes the $\phi$ derivative and returns a new `ChiPhiFunc.` 

Parameters:

- `order : int` (static) - Order of derivative.
- `mode : int` (static) - Numerical method. Supports:
  - `mode=0` -  default mode specified in `config.py`
  - `mode=1` -  spectral method
  - `mode=2` -  pseudo-spectral method (slow to compile and in practice has similar accuracy to spectral method)

### `aqsc.ChiPhiFunc.integrate_phi_fft(zero_avg:bool)` 
Integrates the `ChiPhiFunc` in $\phi$ with spectral method. 

Parameters: 
- `zero_avg : bool` (static) - When `zero_avg==True`, the integral will have zero average. When `zero_avg==False`, the integral will be $0$ at $\phi=0$

### `aqsc.ChiPhiFunc.fft()`
FFT the `axis=1` of content and returns as a ChiPhiFunc.


### `aqsc.ChiPhiFunc.ifft()`
IFFT the `axis=1` of content and returns as a ChiPhiFunc.

## Functions for filtering

### `aqsc.ChiPhiFunc.filter(self, arg:floar, mode:int=0)`
An expandable filter. Now only low-pass is available.

Parameters:

- `mode : int` (static) - Filtering mode. Available modes are:
  - `mode=0` - Low_pass.

- `arg : float` (traced) - Filter  argument. 
  - Under low-pass mode, this is the cutoff mode number.

Returns:

- A new `ChiPhiFunc`.

### `aqsc.ChiPhiFunc.filter_reduced_length(self, arg)`
A Low pass filter that reduces the $\phi$ grid number of a ChiPhiFunc.
        
Parameters: 
- `arg : int` (static) - Cut-off frequency.

Returns: 
- A new `ChiPhiFunc`.

## Functions for indexing

### `aqsc.ChiPhiEpsFunc.__getitem__(self, index)`

Implements `ChiPhiEpsFunc[m]`. Finds the $m$-th mode coefficient.   
Parameters:

- `index : int` (static) - $\chi$ mode number $m$. Must be even or odd (depends on `self.content.shape[0]%2`) and 

Returns:

- A `ChiPhiFunc`.

### `aqsc.ChiPhiFunc.cap_m(m)` 

Removes all $\chi$ modes with mode number larger than `m`. Takes the center m+1 rows of `self.content`. If the ChiPhiFunc contain less $\chi$ modes than $m+1$, It will be zero-padded.

Parameters: 
- `m : int` (static) - number of m in the output.

Returns
- A `ChiPhiFunc` with maximum $\chi$ mode number $m$ (and $m+1$ $\chi$ components).

<!-- pad_m and pad_chi are not used much and omitted here.  -->
## Functions for output and plotting
An overview of a `ChiPhiFunc` can be printed with the built-in `str()` and `print()` statements.

### `aqsc.ChiPhiFunc.eval(chi, phi)`
A vectorized function that evaluates the value of the `ChiPhiFunc`, $f(\chi, \phi)$, at given `chi` and `phi`. The $\phi$ interpolation is performed by `jax.numpy.interp`.

Parameters:

- `chi, phi : array or scalar` (traced) - $\chi$, $\phi$'s to evaluate at.

Returns: 
- An `jnp.array` evaluation result.

### `aqsc.ChiPhiFunc.display_content(self, trig_mode=False, colormap_mode=False)`
Plot the `ChiPhiFunc`'s $\chi$ coefficients as line plots or colormaps.

Parameters: 
- `trig_mode : bool` - When True, plot the trigonometric (instead of exponential) Fourier coefficients, $F_{n,m}^s$ and $F_{n,m}^c$.

- `colormap_mode : bool` - `False` by default. When `True`, make colormaps where each row corresponds to a different $\chi$ component.
  
### `aqsc.ChiPhiFunc.display(self, complex = False, size=(100,100), avg_clim = False)`

Plot the value of the function represented by a `ChiPhiFun` as a colormap w.r.t $\chi$ and $\phi$.

Parameters: 

- `complex : bool` - Set to `True` to plot both the real and imaginary components. By default, we assume the function represented by `ChiPhiFunc` is real and plot only the real component.
- `size : (int, int)` - Number of $\chi$, $\phi$ grids in evaluation
- `avg_clim : bool`: When set to `True`, limits the color range to `+-np.avg(np.abs(self.content))`.

### `aqsc.ChiPhiFunc.export_single_nfp()`
Tiles a `ChiPhiFunc` by `self.nfp` and export a new `ChiPhiFunc` with `nfp==1`.

### `aqsc.ChiPhiFunc.trig_to_exp()`
Converts a ChiPhiFunc from trig to exp fourier series.

Returns:

- A `ChiPhiFunc`.

### `aqsc.ChiPhiFunc.exp_to_trig()`
Converts a ChiPhiFunc from exp to trig fourier series.

Returns:

- A `ChiPhiFunc`.

### `aqsc.ChiPhiFunc.get_max(n_chi, n_phi)`
Calculating the maximum of the absolute value of the function of $\chi$ and $\phi$ represented by this object.

Parameters: 

- `n_chi, n_phi : int` - The grid size for evaluation.

Returns:

- A real scalar.

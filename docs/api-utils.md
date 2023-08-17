# Math utilities and solvers
## Operator generators (`chiphifunc.py`)
### `aqsc.dchi_op(n_dim:int)`
Generates a differential operator that performs derivative in $\chi$. The operator is applied using `diff_matrix@f.content = dchi(f).content`.

Parameters:
- `ndim : int` (static) - Total number of $\chi$ harmonics. (NOT maximum $m$).

Returns: 
- (n_dim, n_dim) a jax.numpy array.


### `aqsc.trig_to_exp_op(n_dim:int)`
Generates an operator that converts a ChiPhiFunc from trigonometric to exponental Fourier series. The operator is applied using `diff_matrix@f.content = dchi(f).content`.

Parameters:
- `n_dim : int` (static) - Total number of $\chi$ harmonics. (NOT maximum $m$).

Returns: 
- (n_dim, n_dim) a jax.numpy array.

### `aqsc.exp_to_trig_op(n_dim:int)`
Generates an operator that converts a ChiPhiFunc from exponential to trigonometric Fourier series. The operator is applied using `diff_matrix@f.content = dchi(f).content`.

Parameters:
- `n_dim : int` (static) - Total number of $\chi$ harmonics. (NOT maximum $m$).

Returns: 
- (n_dim, n_dim) jax.numpy array.

### `aqsc.max_log10(input)`
Calculates the maximum amplitude's order of magnitude

Parameters:
- `input : jax.numpy.array` (traced) - Input data

Returns: 

- The log 10 of the maximum element in input.

### `aqsc.phi_avg(in_quant)`
A type-insensitive phi-averaging function that:
- Averages along phi and output a ChiPhiFunc if the input is a ChiPhiFunc.
- Does nothing if the input is a scalar.
  
Parameters:
- `in_quant` (traced) - Can be scalar or `ChiPhiFunc`.

Returns:
- A `jax.numpy.complex128` average.
    
## PDE and ODE solver
### `get_O_O_einv_from_A_B(chiphifunc_A:ChiPhiFunc, chiphifunc_B:ChiPhiFunc, rank_rhs:int, Y1c_mode:bool)`
(NEED DOUBLE-CHECK)
Get O, O_einv and vector_free_coef that solves
$$    
\left[
    -\frac{1}{2}
    B_{\alpha 0}\frac{\partial X_1}{\partial \chi}
    +\frac{1}{2}
    B_{\alpha 0}X_1 
    \frac{\partial}{\partial \chi}
\right]Y_n = (C^{b,\text{rhs}}_{n-1} - C^{b,\text{lhs}}_{n-1})|_{Y_n=0}\\
\Leftrightarrow (A+B\frac{\partial}{\partial\chi})Y_n = RHS.
$$
Where the operator acting on $Y$ is known to be $(m_y+2, m_y+1)$ of rank $m_y-1$. This equation's solution is known to be
$$
    Y_{\text{odd }n} = \hat{\textbf{L}}^{-1}
        (C^{b,\text{rhs}}_{n-1} - C^{b,\text{lhs}}_{n-1})|_{Y_n=0} 
        + Y_{n,1}K_m
$$
or
$$
    Y_{\text{even }n} = \hat{\textbf{L}}^{-1}
        (C^{b,\text{rhs}}_{n-1} - C^{b,\text{lhs}}_{n-1})|_{Y_n=0} 
        + Y_{n,0}K_m,
$$
Where $\hat{\textbf{L}}^{-1}$ is a stack of $(n+1, n+2)$ operators acting on the $\chi$ components of $RHS$, $Y_{n,0}$ or $Y_{n,1}$ is a function of $\phi$, and $K_m$ is a function of $\chi, \phi$

Inputs: 
- `A, B : ChiPhiFunc` (traced) - coefficients
- `rank_rhs : int` (static) - The number of rows in the RHS.

Outputs:
`O_matrices, O_einv, vector_free_coef, Y_nfp`
- `O_matrices : jax.numpy.array` (traced) - An `(n+2, n+1, len_phi)` operator equivalent to $\left[
    -\frac{1}{2}
    B_{\alpha 0}\frac{\partial X_1}{\partial \chi}
    +\frac{1}{2}
    B_{\alpha 0}X_1 
    \frac{\partial}{\partial \chi}
\right]$
- `O_einv : jax.numpy.array` (traced) - The above-mentioned$\hat{\textbf{L}}^{-1}$ operator. Has shape `(n+1, n+2, len_phi)`.
- `vector_free_coef : jax.numpy.array` (traced) - The above-mentioned $K_m$. Has shape `(n+1,len_phi)`.
- `Y_nfp : int` (static) - The `nfp` of $Y$. For internal use.

### `aqsc.solve_1d_asym(p_eff, f_eff)` (Depreciated)
Solves a linear ODE of form $y' + p_{eff}*y = f_{eff}$ using asymptotic series. It only works well when the minimum amplitude of p (?) is large, and in these cases FFT works well too. The maximum truncation order is set in `ChiPhiFunc.py`.

Parameters: 

- `p_eff : jnp.numpy.array` (traced) - Coefficient of y. 1d array.
- `f_eff : jnp.numpy.array` (traced) - The RHS. 1d array.

Returns:

- A `jnp.numpy.array` containing the solution to the equation. When `p_eff` is 0, it will be defaulted to the anti-derivative of `f_eff` with zero average.

### `aqsc.solve_1d_fft(p_eff, f_eff, static_max_freq:int=None)`
Solves a linear ODE of form $y' + p_{eff}*y = f_{eff}$ using spectral method. The maximum truncation order is set in `ChiPhiFunc.py`.

Parameters: 

- `p_eff : jnp.numpy.array` (traced) - Coefficient of y. 1d array.
- `f_eff : jnp.numpy.array` (traced) - The RHS. 1d array.
- `static_max_freq : int` (static) - Maximum number of Fourier harmonics used. Should be set as low as posssible while trying not losing too much accuracy to prevent high-frequency noise from blowing up. Need to be find empirically.

Returns:

- A `jnp.numpy.array` containing the solution to the equation. When `p_eff` is 0, it will be defaulted to the anti-derivative of `f_eff` with zero average.

### `aqsc.solve_ODE(coeff_arr, coeff_dp_arr, f_arr, static_max_freq: int=None)`
Solves a list of linear first order ODE systems in of form
$(\text{coeff} + \text{coeff}_\phi \frac{d}{d\phi}) y = f$ (equivalent to $y' + p_{eff}*y = f_{eff}$) using spectral method.

NOTE:
Does not work well for p>10 with zeros or resonant p without low-pass filtering.

Parameters:
- `coeff_arr, coeff_dp_arr, f_arr : jax.numpy.array` (traced) - Components of the equations as 2d matrices. Its `axis=0` is equation indices and `axis=1` is $\phi$ dependence on grid points. All quantities are assumed periodic.
- `static_max_freq : int` (static): Maximum number of Fourier harmonics used. Should be set as low as posssible while trying not losing too much accuracy to prevent high-frequency noise from blowing up. Need to be find empirically.

Returns:

- A `jax.numpy.array` containing solutions to the equations.

### `aqsc.solve_ODE_chi(coeff, coeff_dp, coeff_dc, f, static_max_freq:int)`
Solves the periodic linear 1st order PDE $(\text{coeff} + \text{coeff}_\phi\frac{d}{d\phi} + \text{coeff}_\chi\frac{d}{d\chi}) y = f(\phi, \chi)$ using spectral method.

Parameters: 

- `coeff, coeff_dp, coeff_dc` (traced) - The coefficients. Can be 2D arrays with the same format as `ChiPhiFunc.content` or scalars. 
- `f : jax.numpy.array` (traced) - The RHS. Must be a 2D array with the same format as `ChiPhiFunc.content`.
- `static_max_freq : int` (static) - Maximum number of Fourier harmonics used. Should be set as low as posssible while trying not losing too much accuracy to prevent high-frequency noise from blowing up. Need to be find empirically.

Returns: 
- A `jax.numpy.array` containing the solution. Has the same format as `ChiPhiFunc.content`.

### `aqsc.solve_dphi_iota_dchi(iota, f, static_max_freq: int)`
Solves the periodic linear 1st order PDE $(\frac{d}{d\phi} + \bar{\iota}\frac{d}{d\chi}) y = f(\phi, \chi)$ using spectral method.

Parameters: 
- `iota` (traced) - A scalar coefficient.
- `f : jax.numpy.array` (traced) - The RHS. Must be a 2D array with the same format as `ChiPhiFunc.content`.
- `static_max_freq : int` (static) - Maximum number of Fourier harmonics used. Should be set as low as posssible while trying not losing too much accuracy to prevent high-frequency noise from blowing up. Need to be find empirically.

Returns: 
- A `jax.numpy.array` containing the solution. Has the same format as `ChiPhiFunc.content`.

## Utilities

### `aqsc.fft_filter(fft_in:jax.numpy.ndarray, target_length:int, axis:int)`
Shorten an array in frequency domain to leave only target_length elements. (equivalent to a low-pass filter) by removing the highest frequency modes, but used in solvers to reduce array sizes. The result will still be in frequency domain.with IFFT.

Parameters: 
- `fft_in : jax.numpy.array` (traced) - `ndarray` to filter. Must already be in frequency domain produced by `np.fft.fft()`.
- `target_length : int` (static) - Length of the output array
- `axis : int` (static) - Axis to filter along

Returns: 
- A filtered `jax.numpy.array`.

### `aqsc.fft_pad(fft_in:jax.numpy.array, target_length:int, axis:int)`
Pads an `array` in frequency domain to `target_length` elements by adding zeroes for high frequency modes coefficients. Used in solvers to match required array sizes. The result will still be in frequency domain.

Parameters: 
- `fft_in : jax.numpy.array` (traced) - `array` to pad. Must already be in frequency domain produced by `numpy.fft.fft`.
- `target_length : int` (static) - Length of the output array.
- `axis : int` (static) - axis to filter along

Returns: 
- The padded `jax.numpy.array`.

## Tensor construction functions for `looped_solver.py` (Not very useful otherwise)
(NEED DOUBLE-CHECK)

Theses functions are for constructing differential/convolution tensors (equivalent to coupled PDE's) in frequency space for `looped_solver.py`

### `aqsc.conv_tensor(content:jax.numpy.ndarray, n_dim:int)`
Generate a 3D array containing a stack of `content.shape[1]` convolution matrices. The first two axes represent the row and column of a single convolution matrix. It convolves the $\chi$ Fourier components of a `ChiPhiFunc` to another along axis 0. For multiplication in FFT space during ODE solves. To apply the operator, use `x2_conv_y2 = jax.numpy.einsum('ijk,jk->ik',conv_x2, y2.content)`

Parameters: 
- `content : jax.numpy.ndarray` (traced) - A content matrix of the `ChiPhiFunc` to convolve with

Returns: 
- A `jax.numpy.array` with shape `(content.shape[0]+n_dim-1, n_dim, content.shape[1])` containing a stack of content.shape[1] convolution matrices. The first two axes represent the row and column of a single convolution matrix.

### `aqsc.fft_conv_tensor_batch(source:jax.numpy.array)`
Generates a 4D convolution operator in the phi axis from a 3D stack of $\chi$ operators. (see comments in looped_solver for explanation).

Parameters: 
- `source : jax.numpy.ndarray` (traced) - An `(a, b, len_phi)` array, where a and b represents a matrix acting on the $\chi$ components of a `ChiPhiFunc`

Returns: 
- An `(a, b, len_phi, len_phi)` array that acts on a content by transposing axis 1 and 2 and then using `jax.numpy.tensordot()`. (see explanation for a "tensor operator" in comments in `looped_solver.py`) 

### `aqsc.to_tensor_fft_op(ChiPhiFunc_in:ChiPhiFunc, len_tensor:int)`
Reduce the grid number of a `ChiPhiFunc` in frequency domain with low-pass filter, and then create a 4D convolution operator in $\phi$ frequency domain of given length that performs point-wise multiplication with a `ChiPhiFunc` to a **$\chi$-independent** `ChiPhiFunc.fft().content`.

Parameters:
- `ChiPhiFunc_in : ChiPhiFunc` (traced) - The kernel. Has `a` $\chi$ components.
- `len_tensor : int` (static) - The dimension of the output.

Returns:
- An `(a, 1, len_tensor, len_tensor)` array that acts on a content by transposing axis 1 and 2 and then using `jax.numpy.tensordot()`. (see explanation for a "tensor operator" in comments in `looped_solver.py`) 

### `aqsc.to_tensor_fft_op_multi_dim(ChiPhiFunc_in:ChiPhiFunc, dphi:int, dchi:int, num_mode:int, cap_axis0:int, len_tensor: int, nfp: int)`
Reduce the grid number of a `ChiPhiFunc` in frequency domain with low-pass filter, and then create a 4D convolution operator in $\phi$ frequency domain of given length that performs point-wise multiplication with a `ChiPhiFunc` to a **$\chi$-dependent** `ChiPhiFunc.fft().content`, or optionally its derivative(s) in $\chi$ and/or $\phi$:

$$F(\chi,\phi)\left(\frac{d^a}{d\phi^a}\frac{d^b}{d\phi^b}\right)$$
    
Parameters: 
- `ChiPhiFunc_in:ChiPhiFunc` (traced) - A 
- `num_mode : int` (static) - The number of columns of the resulting tensor. Corresponds to the total number of $\chi$ component in the `ChiPhiFunc.fft().content` the tensor will acting on.
- `cap_axis0 : int` (static) - The length of axis=0 for the resulting tensor,used to remove outer components that are known to cancel. Must have the same even/oddness and smaller than the row number of the convolution tensor generated from `ChiPhiFunc_in` and `num_mode`.

Returns: 
- An `jax.numpy.array` with shape `(len_chi+num_mode-1, num_mode, len_tensor, len_tensor)`. Acts on a `ChiPhiFunc.fft().content` by `np.tensordot(operator, content, 2)`.

### `aqsc.linear_least_sq_2d_svd(A, b)`
Solves the linear least square problem minimizing $|Ax-b|^2$. Used to solve over-determined linear systems.

Parameters: 
- `A, b : jax.numpy.array` (traced) - Input arrays of shape `(n,m)` and `(n)`

Returns: 
- A `jax.numpy.array` with shape `(m)`.




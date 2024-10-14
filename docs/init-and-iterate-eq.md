# New QS equilibria

PyAQSC can solve 
- The full governing equation set for a QS equilibrium.
- Or only the magnetic equations for a QS field with no force balance. 

Each configuration is managed by a object of the traced class `Equilibrium`. An example is the return from `aqsc.circular_axis()`. See [background on free parameters](background-free-params.md) for a list of free parameters, their physics definition, dependence, and expansion forms.

This part shows how to create a new QS equilibrium from scratch and iterate to higher orders. For creating a QS magnetic field without force balance, see the [next part](init-and-iterate-mag.md).

All grid values must be provided at $\phi=0, \frac{1}{n_\text{grid}}\frac{2\pi}{n_\text{field period}}, ..., \frac{n_\text{grid}-1}{n_\text{grid}}\frac{2\pi}{n_\text{field period}}$.

## Creating new configrations
`aqsc.leading_orders()` creates a QS equilibrium and performs leading order ($n=2$) calculations:

    new_equilibrium = aqsc.leading_orders(
        nfp,
        Rc, Rs, Zc, Zs,
        p0
        Delta_0_avg,
        B_theta_20_avg,
        B_alpha_1, 
        B0, B11c, B22s, B20, B22c,
        len_phi,
        static_max_freq,
        traced_max_freq,
        riccati_secant_n_iter
    )

**Note**: `B11c` must be **nonzero** due to the inner working of the expansion.

Parameters:

- `nfp : int` (static) - The number of field period.
- `Rc, Rs, Zc, Zs : list(float)` (traced) - $sin(i\Phi)$, $cos(i\Phi)$ coefficients of axis shape $R_0$ and $Z_0$ w.r.t. cylindrical $\Phi$. The $i$-th component is the $i$-th mode coefficient.
- `p0 : array` (traced) - The on-axis pressure as a function of general Boozer coordinate angle $\phi$ in one field period on grids. 
- `Delta_0_avg : float` (traced) - The average on-axis anisotropy.
- `B_theta_20_avg : float` (traced) - The average $\chi$-independent component of $\bar{B}_{\theta 2}$.
- `B_alpha_1 : float` (traced) - The 1st order component of flux funtion $B_\alpha$
- `B0, B11c, B22s, B20, B22c : float` (Traced) - Leading components of the magnetic field magnitude $B^-$.
- `len_phi` (static) - The $\phi$ grid number
- `static_max_freq : int` (static) - The cut-off frequency for the low-pass filter on the results. Tied to array sizes during spectral solve, and lower value drastically increases solving speeds. Changing will result in recompiling.
- `traced_max_freq` (traced) - The cut-off frequency for the low-pass filter on the results. Doesn't impact speed and doesn't require recompiling.
- `riccati_secant_n_iter` (static) - The number of Secant iterations for calculating the leading order Riccati equation.
  
Returns: 
- A new `Equilibrium`.

## Iterating to higher orders
To calculate the next two orders, use `aqsc.Iterate_2()`:

    new_equilibrium = iterate_2(
        equilibrium,
        B_alpha_nb2,
        B_denom_nm1, B_denom_n,
        iota_new, 
        B_theta_n0_avg,
        static_max_freq=(-1,-1),
        traced_max_freq=(-1,-1),
        max_k_diff_pre_inv=(-1,-1),
        n_eval=None,
    )

Each call creates a new `Equilubrium`.

Parameters:
- `equilibrium : Equilibrium` (traced) - The `Equilibrium` to iterate.
- `B_alpha_nb2 : float` (traced) - $B_{\alpha n/2}$
- `B_denom_nm1 : ChiPhiFunc` (traced) - $B^-_{n-1}$
- `B_denom_n : ChiPhiFunc` (traced) - $B^-_n$
- `iota_new : float` (traced) - $\bar{\iota}$
- `B_theta_n0_avg: float`(traced) - The toroidal average of $B_{\theta n0}$, $\int d\phi B_{\theta n0}$
- `static_max_freq : (int,int)` (static) - The cut-off frequency for the low-pass filter on the results.
- `traced_max_freq : (scalar,scalar)` (traced) - The cut-off frequency for the low-pass filter on the results.
- `max_k_diff_pre_inv : (scalar,scalar)` (traced) - Cut-off frequency for the off-diagonal filter when solving the looped equations.
- `n_eval : int` (static) - Order to evaluate to. By default evaluates the next two orders, but can also re-evaluate 2 orders from lower $n$. Must be even and no greater than $n+2$.

Returns
- A new `Equilibrium`

This function is defined in `equilibrium.py`.


# New QS equilibria

PyAQSC can solve 
- The full governing equation set for a QS equilibrium.
- Or only the magnetic equations for a QS field with no force balance. 

Each configuration is managed by a object of the traced class `Equilibrium`. An example is the return from `aqsc.circular_axis()`. See [background on free parameters](background-free-params.md) for a list of free parameters, their physics definition, dependence, and expansion forms.

This part shows how to create a new QS magnetic field from scratch and iterate to higher orders.

## Creating new configrations
`aqsc.leading_orders_magnetic()` creates a QS equilibrium and performs leading order ($n=2$) calculations:

    new_equilibrium = aqsc.leading_orders(
        nfp,
        Rc, Rs, Zc, Zs, 
        iota_0, 
        B_theta_20, 
        B_psi_00,
        Y20,
        B_alpha_1,
        B0, B11c, B2, 
        len_phi,
        static_max_freq,
        traced_max_freq,
    )

Parameters:

- `nfp : int` (static) - The number of field period.
- `Rc, Rs, Zc, Zs : 1d array` (traced) - $sin(i\Phi)$, $cos(i\Phi)$ coefficients of axis shape $R_0$ and $Z_0$ w.r.t. cylindrical $\Phi$. The $i$-th component is the $i$-th mode coefficient.
- `iota_0 : float` (traced) - The 0th order rotational transform.
- `B_theta_20 : 1d array` (traced) - $B_{\theta 2,0}$.
- `B_psi_00 : 1d array` (traced) - $B_{\psi 0,0}$.
- `Y20 : 1d array` (traced) - $B_{\psi 0,0}$.
- `B_alpha_1 : float` (traced) - The 1st order component of flux funtion $B_\alpha$
- `B0, B11c, B2 : float, float, ChiPhiFunc` (Traced) - Leading components of the magnetic field magnitude $B^-$.
- `len_phi` (static) - The $\phi$ grid number
- `static_max_freq : int` (static) - The cut-off frequency for the low-pass filter on the results. Tied to array sizes during spectral solve, and lower value drastically increases solving speeds. Changing will result in recompiling.
- `traced_max_freq` (traced) - The cut-off frequency for the low-pass filter on the results. Doesn't impact speed and doesn't require recompiling.

Returns: 
- A new `Equilibrium`.

This function is defined in `leading_orders.py`.

## Iterating to higher orders
To iterate the next two orders, use `aqsc.iterate_2_magnetic_only()`:

    new_equilibrium = iterate_2_magnetic_only(
        equilibrium,
        B_theta_nm1, B_theta_n,
        Yn0,
        B_psi_nm20,
        B_alpha_nb2,
        B_denom_nm1, B_denom_n,
        iota_nm2b2,
        static_max_freq=(-1,-1),
        traced_max_freq=(-1,-1),
        n_eval=None,
    )

Parameters:
`B_theta_nm1 : ChiPhiFunc` (traced) - $B_{\theta n-1}$
`B_theta_n : ChiPhiFunc` (traced) - $B_{\theta n}$
`Yn0 : 1d array` (traced) - $Y_{n,0}$
`B_psi_nm20 : 1d array` (traced) - $B_{\psi n-2, 0}$
`B_alpha_nb2 : float` (traced) - $B_{\alpha n/2}$
`B_denom_nm1 : ChiPhiFunc` (traced) - $B^-_{n-1}$
`B_denom_n : ChiPhiFunc` (traced) - $B^-_{n}$
`iota_nm2b2 : float` (traced) - $\bar{\iota}^-_{n-1}$
`static_max_freq : (int, int)` (static) - The cut-off frequency for the low-pass filter on the results.
`traced_max_freq : (scalar, scalar)` (traced) - The cut-off frequency for the low-pass filter on the results.
`n_eval : ChiPhiFunc` (static) - `n_eval : int` (static) - Order to evaluate to. By default evaluates the next two orders, but can also re-evaluate 2 orders from lower $n$. Must be even and no greater than $n+2$.

Returns
- A new `Equilibrium`
  
This function is defined in `equilibrium.py`.

## Loading from pySQC
PyAQSC also supports loading magnetic fields from a `qsc.Qsc` object in pyQSC. Simply use:

    new_equilibrium = import_from_stel(stel, len_phi=1000, nfp_enabled=False)

Where `stel` is the `qsc.Qsc` object and `len_phi` is the desired $\phi$ grid number. `qsc.Qsc` stores all field periods on its grid. When `nfp_enabled` is set to `True`, pyAQSC converts the data to store only one field period. When set to `False`, all field periods will be stored. This method cannot be JIT compiled.
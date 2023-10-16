# Equilibrium API
As its name suggests, the Equilibrium class manages all information for a QS equilibrium or magnetic field. For the definition of quantities, see the sections on [unknowns](background-solves-for.md) and [free parameters](background-free-params.md).

## Class attributes
### `self.unknown : {string:ChiPhiEpsFunc}` (dict of traced) 
All unknowns solved by pyAQSC. Contains keys:

- `'X_coef_cp' : ChiPhiEpsFunc` - $X$. At even order $n$ at all time.
- `'Y_coef_cp' : ChiPhiEpsFunc` - $Y$. At even order $n$ at all time.
- `'Z_coef_cp' : ChiPhiEpsFunc` - $Z$. At even order $n$ at all time.
- `'B_psi_coef_cp' : ChiPhiEpsFunc` - $B_{\psi}$. At even order $n-2$ at all time.
- `'B_theta_coef_cp' : ChiPhiEpsFunc` - $B_{\theta}$. At even order $n$ at all time.
- `'iota_coef' : ChiPhiEpsFunc` - $\bar{\iota}$. At order $(n-2)/2$ at all time.
- `'p_perp_coef_cp' : ChiPhiEpsFunc` - $p_\perp$. At even order $n$ at all time.
- `'Delta_coef_cp' : ChiPhiEpsFunc` - $\Delta$. At even order $n$ at all time.

### `self.constant : {string:ChiPhiEpsFunc or ChiPhiFunc}` (dict of traced) 
All constants (inputs or calculated from inputs). Contains keys:

- `'B_denom_coef_c' : ChiPhiEpsFunc` - $B^-$. At even order $n$ at all time.
- `'B_alpha_coef' : ChiPhiEpsFunc` - $B_{\alpha}$. At order $n/2$ at all time.
- `'kap_p' : ChiPhiFunc` - $\kappa$.
- `'dl_p' : float` - $dl/d\phi$.
- `'tau_p' : ChiPhiFunc` - $\tau$.

### `self.axis_info : {string:jax.numpy.array}` (dict of traced) 
Axis information, not used in further iteration. The naming convention is the same as in [pyQSC](https://landreman.github.io/pyQSC/outputs.html#outputs). Contains keys:

- `'varphi' : array` - The Boozer toroidal angle $\phi$ on a uniformly spaced grid of cylindrical toroidal angle $\Phi=0, \frac{2\pi}{n_{fp}}\frac{1}{n_{grid}}, ..., \frac{2\pi}{n_{fp}}\frac{n_{grid}}{n_{grid}}$.
- `'phi' : array` - A uniformly spaced grid of cylindrical toroidal angle $\Phi=0, \frac{2\pi}{n_{fp}}\frac{1}{n_{grid}}, ..., \frac{2\pi}{n_{fp}}\frac{n_{grid}}{n_{grid}}$.
- `'d_phi' : float` - $\Phi$ grid spacing
- `'R0' : array` - $R_0$, axis shape in cylindrical coordinate.
- `'Z0' : array` - $Z_0$, axis shape in cylindrical coordinate.
- `'R0p' : array` - $dR_0/d\Phi$.
- `'Z0p' : array` - $dZ_0/d\Phi$.
- `'R0pp' : array` - $d^2Z_0/d\Phi^2$.
- `'Z0pp' : array` - $d^2Z_0/d\Phi^2$.
- `'R0ppp' : array` - $d^3R_0/d\Phi^3$.
- `'Z0ppp' : array` - $d^3R_0/d\Phi^3$.
- `'d_l_d_phi' : array` - $dl/d\Phi$, rate of change of axis length $l$ w.r.t. the cylindrical toroidal angle $\Phi$.
- `'axis_length' : array` - Total axis length.
- `'curvature' : array` - Curvature in cylindrical coordinate.
- `'torsion' : array` - Torsion in cylindrical coordinate.
- `'tangent_cylindrical' : array` - Components of the unit tangent vector $\hat{\boldsymbol{b}}_0$ in cylindrical coordinate.
- `'normal_cylindrical' : array` - Components of the unit normal vector $\hat{\boldsymbol{\kappa}}_0$ in cylindrical coordinate.
- `'binormal_cylindrical' : array` - Components of the unit binormal vector $\hat{\boldsymbol{\tau}}_0$ in cylindrical coordinate.

### `self.nfp : int` (static) 
The number of field period of an equilibrium
### `self.magnetic_only : bool` (static) 
Iteration mode. When set to `False` (default), is a QS equilibrium with anisotropic pressure. When set to `True`, is a QS magnetic field without force balance.

## Constructor
Instead of the constructor, we recommend creating equilibria using `aqsc.leading_orders()` (see [iteration API](api-iteration.md)). 
### `aqsc.Equilibrium(unknown, constant, nfp, magnetic_only, axis_info={})`

Parameters:

- `self.unknown : {string:ChiPhiEpsFunc}` (dict of traced) .
- `self.constant : {string:ChiPhiEpsFunc or ChiPhiFunc}` (dict of traced) 
- `self.axis_info : {string:jax.numpy.array}` (dict of traced) 
- `self.nfp : int` (static) 
- `self.magnetic_only : bool` (static) 

### `from_known(X_coef_cp, Y_coef_cp, Z_coef_cp, B_psi_coef_cp, B_theta_coef_cp, B_denom_coef_c, B_alpha_coef, kap_p, dl_p, tau_p, iota_coef, p_perp_coef_cp, Delta_coef_cp, axis_info={}, magnetic_only=False )`
Constructs an equilibrium from known quantities. **Does not** check order consistency, nfp, grid number or whether the known quantities obey the ordered governing equations. We recommend creating equilibria using `aqsc.leading_orders()` (see [Quick start](api-iteration.md)). 

Parameters:

- `X_coef_cp : ChiPhiEpsFunc` - $X$. Must be even order $n$.
- `Y_coef_cp : ChiPhiEpsFunc` - $Y$. Must be even order $n$.
- `Z_coef_cp : ChiPhiEpsFunc` - $Z$. Must be even order $n$.
- `B_psi_coef_cp : ChiPhiEpsFunc` - $B_{\psi}$. Must be even order $n-2$.
- `B_theta_coef_cp : ChiPhiEpsFunc` - $B_{\theta}$. Must be even order $n$.
- `B_denom_coef_c : ChiPhiEpsFunc` - $B^-$. Must be even order $n$.
- `B_alpha_coef : ChiPhiEpsFunc` - $B_{\alpha}$. Must be order $n/2$.
- `kap_p : ChiPhiFunc` - $\kappa$.
- `dl_p : float` - $dl/d\phi$.
- `tau_p : ` - $\tau$.
- `iota_coef : ChiPhiEpsFunc` - $\bar{\iota}$. Must be order $(n-2)/2$.
- `p_perp_coef_cp : ChiPhiEpsFunc` - $p_\perp$. Must be even order $n$.
- `Delta_coef_cp : ChiPhiEpsFunc` - $\Delta$. Must be even order $n$.
- `axis_info : ChiPhiEpsFunc` - Axis information calculated by `aqsc.leading_orders()`. Not used in further iteration. Can be left blank.
- `magnetic_only : ChiPhiEpsFunc` - Whether the iteration is magnetic-only. `False` by default.

## Functions for saving and loading
### `aqsc.Equilibrium.save(file_name)`
Saves `self` as a `.npy` file.

Parameters:

- `file_name : str` - The name of the save file (without extension).
  
### `aqsc.Equilibrium.load(file_name)`
Loads from a `.npy` file created by `aqsc.Equilibrium.save()`.

Parameters:

- `file_name : str` - The name of the save file (with extension).
Returns: 
- An `aqsc.Equilibrium`.

### `aqsc.Equilibrium.save_plain(self, file_name)`
Saves `self` as a `.npy` file. This `.npy` file contains a dict of `list`'s from `aqsc.ChiPhiEpsFunc.to_content_list()` and does not require pyAQSC to load.

Parameters:

- `file_name : str` : The name of the save file (without extension).
  
### `aqsc.Equilibrium.load_plain(filename)`
Loads from a `.npy` file created by `aqsc.Equilibrium.save_plain()`.

Parameters:

- `file_name : str` - The name of the save file (without extension).
Returns: 
- An `aqsc.Equilibrium`.

## Functions for output

### `aqsc.Equilibrium.get_eps_crit(n_max=float('inf'), n_grid_chi=100, n_grid_phi=100, eps_cap = 2.0, n_newton_iter = 10)`
### `aqsc.Equilibrium.get_psi_crit(n_max=float('inf'), n_grid_chi=100, n_grid_phi=100, eps_cap = 2.0, n_newton_iter = 10)`
Estimates the critical $\epsilon=\sqrt{\psi}$ or $\psi$ where flux surface self-intersects
by numerically finding the zero of $\sqrt{g}(\epsilon, \chi, \phi)$ with the smallest $\epsilon$ or $\psi$ using Newton's method. At each search step, $\sqrt{g}(\epsilon_i, \chi, \phi)$ is evaluated on a `n_grid_chi`x`n_grid_phi` grid.

This function can be slow to JIT compile. When good accuracy is not required, reducing `n_newton_iter`, `n_grid_chi` and `n_grid_phi` can substantially increase the compile speed.

JIT compiling the function setting `eps_cap` as traced will not cause errors, but it increases compile time and is not recommended.

Parameters:

- `n_max` (static) - maximum order to evaluate coordinate transformation to.

- `n_grid_chi, n_grid_phi : int` (static) - Grid size to evaluate Jacobian $\sqrt{g}$ on. 
The critical point occurs when $min(\sqrt{g}\leq0)$.

- `eps_cap : float` (static) - An initial guess of an epsilon>epsilon_crit used in Newton's 
method. Need to be beyond t

- `n_newton_iter : int` (static) - Maximum number of steps in Newton's method.
higher number gives better acuracy but is slower to jit.

Returns: 

- `(eps_crit, jacobian_residue)`: $\epsilon_{crit}$ and the flux surface min of the Jacobian at eps_crit.

### `aqsc.Equilibrium.get_order()`
Gets the highest known order $n$ of `self`.

Returns:

- An int $n$.

### `aqsc.Equilibrium.get_helicity()`
Gets the helicity of `self`, same as the number of rotation of the normal
basis vector $\kappa$ in the Frenet basis.

Returns:

- An int $n$.

### `aqsc.Equilibrium.flux_to_frenet(psi, chi, phi, n_max=float('inf'))`
### `aqsc.Equilibrium.flux_to_cylindrical(psi, chi, phi, n_max=float('inf'))`
### `aqsc.Equilibrium.flux_to_xyz(psi, chi, phi, n_max=float('inf'))`
Vectorized functions that transforms points in the flux coordinte $(\psi, \chi, \phi)$ into points in the Frenet coordinate $(\textit{curvature}, \textit{binormal}, \textit{tangent})$, cylindrical coordinate $(R, \Phi, Z)$, or Cartesian coordinate $(x, y, z)$ using the self-consistently solved coordinate transformations $X, Y, Z(\psi, \chi, \phi)$. ($Z$ here is different from $Z$ in the cylindrical coordinate. See [background](background-solves-for.md) for its definition.)

Parameters:

- `psi, chi, phi : array or scalar` (traced) - Points in the flux coordinate.
- `n_max : scalar` (static) - Max order $n$ of the coordinate transform $X, Y, Z$ to use. If larger than the highest known order, uses all known orders. By default uses all known orders.

Returns:

- 3 `jnp.arrays`. `(curvature, binormal, tangent)`, `(R, Phi, Z)` or `(x, y, z)`

### `aqsc.Equilibrium.frenet_basis_phi(phi)`
A vectorized function that evaluates the axis shape $\textbf{r}_0[l(\phi)]$ and Frenet basis $(\hat{\boldsymbol{\kappa}}_0, \hat{\boldsymbol{\tau}}_0, \hat{\boldsymbol{b}}_0)[l(\phi)]$ in the cylindrical coordinate $(R, \Phi, Z)$ at a given $\phi$.

Parameters:

- `phi : array or scalar` (traced) - The toroidal angle $\phi$ on axis. 
  
Returns:

- `axis_r0_phi_R : jnp.array`
- `axis_r0_phi_Phi : jnp.array`
- `axis_r0_phi_Z : jnp.array`
- `tangent_phi_R : jnp.array`
- `tangent_phi_Phi : jnp.array`
- `tangent_phi_Z : jnp.array`
- `normal_phi_R : jnp.array`
- `normal_phi_Phi : jnp.array`
- `normal_phi_Z : jnp.array`
- `binormal_phi_R : jnp.array`
- `binormal_phi_Phi : jnp.array`
- `binormal_phi_Z : jnp.array`

### `aqsc.Equilibrium.check_order_consistency()`
Checks whether all items in `self.unknown` and `self.constant` has consistent highest known order. If not, throws `AttributeError`'s. **Cannot be JIT compiled.**

### `aqsc.Equilibrium.check_governing_equations(n_unknown:int, magnetic:bool=False)`
Evaluates the residual (LHS-RHS) of all governing equations at a given order. The results should be as close to 0 as possible.

Parameters:

- `n_unknown : int` (static) - Order to evaluate residuals at.

Returns:

- `J : ChiPhiFunc` (traced) - The residual of the Jacobian equation
- `Cb : ChiPhiFunc` (traced) - The residual of the co/contravariant equation's tangent component.
- `Ck : ChiPhiFunc` (traced) - The residual of the co/contravariant equation's normal component. 
- `Ct : ChiPhiFunc` (traced) - The residual of the co/contravariant equation's binormal component
- `I : ChiPhiFunc` (traced) - The residual of the force balance equation I. (See [Ref.2](https://doi.org/10.1063/5.0027574))
- `II : ChiPhiFunc` (traced) - The residual of the force balance equation II.
- `II : ChiPhiFunc` (traced) - The residual of the force balance equation III.

### `aqsc.Equilibrium.display_order(n:int)`
Plots all quantities at order $n$.

Parameters:

- `n : int` - Order to plot.

### `aqsc.Equilibrium.display(psi_max:float=0.03)`
Plots a boundary with given $\psi$ and some flux surfaces.

Parameters:

- `psi_max : float` - Max $\psi$ to plot to. 


Parameters:

- `n_unknown : int` (static) - Order to plot.


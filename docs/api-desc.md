# DESC conversion tools

PyAQSC comes with 2 functions that generates DESC equilibria from the near-axis solutions. 

### `aqsc_to_desc_near_axis(na_eq, psi_max, n_max=float('inf'), M=6, N=8, stellsym=True, solve_force_balance=True, maxiter=100)`

Generates a DESC equilibrium using near-axis constraints. 

Parameters:

- `na_eq : aqsc.equilibrium` (traced) - The AQSC equilibrium to convert.
- `psi_max : float` (traced) - The maximum $\psi$ fit the equilibrium to.
- `n_max : int or float('inf)` (static) - The order to fit to.
- `M, N : int` (static) - The M and N of the Fourier-Zernike.
- `stellsym : bool = True` (static) - Stellarator symmetry.
- `solve_force_balance : bool = True` (static) - Whether to solve for force balance, or only fit the equilibrium.
- `maxiter : int = 100` (static) - The number of iteration when solving for force balance.

Returns:

- `eq : desc.Equilibrium` - The fitted DESC equilibrium.
- `eq_force_balanced : desc.Equilibrium` - (only when `solve_force_balance == True`) The fitted DESC equilibrium after optimizing for force balance
- `aux_dict` - A dictionary containing the following entries:
    ```
    {
        'rho': rho, 
        'thetaBs': thetaBs, 
        'phiBs': phiBs, 
        'psi': psi,
        'chis': chis
    }
    ```

### `aqsc_to_desc_boundary(na_eq, psi_max, n_max=float('inf'), M=6, N=8, stellsym=True, solve_force_balance=True, maxiter=100)`

Generates a DESC equilibrium using near-axis constraints. 

Parameters:

- `na_eq : aqsc.equilibrium` (traced) - The AQSC equilibrium to convert.
- `psi_max : float` (traced) - The maximum $\psi$ fit the equilibrium to.
- `n_max : int or float('inf)` (static) - The order to fit to.
- `M, N : int` (static) - The M and N of the Fourier-Zernike.
- `stellsym : bool = True` (static) - Stellarator symmetry.
- `solve_force_balance : bool = True` (static) - Whether to solve for force balance, or only fit the equilibrium.
- `maxiter : int = 100` (static) - The number of iteration when solving for force balance.

Returns:

- `eq : desc.Equilibrium` - The fitted DESC equilibrium.
- `eq_force_balanced : desc.Equilibrium` - (only when `solve_force_balance == True`) The fitted DESC equilibrium after optimizing for force balance
- `aux_dict` - A dictionary containing the following entries:
    ```
    {
        'rho': rho, 
        'thetaBs': thetaBs, 
        'phiBs': phiBs, 
        'psi': psi,
        'chis': chis
    }
    ```
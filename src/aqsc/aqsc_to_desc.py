import jax.numpy as jnp
from scipy.constants import mu_0
from scipy import special

def aqsc_to_desc_near_axis(
    na_eq, psi_max, n_max=float('inf'), 
    M=6, N=8, solve_force_balance=True,
    stellsym=True,
    **kwargs):
    """Convert an equilibrium from AQSC to DESC.
    Parameters
    ----------
    na_eq : aqsc.Equilibrium
        aqsc equilibrium to convert
    r : float
        Value of Psi/2pi defining the LCFS of the near axis equilibrium
    M : int
        Maximum poloidal mode number of DESC equilibrium
    N : int
        Maximum toroidal mode number of DESC equilibrium
    Returns
    -------
    desc_eq : desc.equilibrium.Equilibrium
        near axis equilibrium fit to desc basis.
    """
    try:
        from desc.equilibrium import Equilibrium
        from desc.grid import Grid
        from desc.basis import FourierZernikeBasis
        from desc.transform import Transform
        from desc.profiles import FourierZernikeProfile, PowerSeriesProfile
        from desc.objectives import ForceBalanceAnisotropic, ObjectiveFunction, get_NAE_constraints
        # Code leveraging desc-opt
    except ImportError:
        raise ImportError("This feature requires DESC. Install it with `pip install desc-opt`.")

    r=float(psi_max) # Psi_aqsc = Psi/2pi
    L=M
    spectral_indexing="ansi"
    
    if N is None:
        N = M
    if N == jnp.inf:
        N = int((na_eq.axis_info['phi'].shape - 1) / 2)
    
    ntheta = 4 * M + 1
    nzeta = 4 * N + 1
    nrho = 2 * L
        
    rho, _ = special.js_roots(nrho, 2, 2)
    thetaBs = jnp.linspace(0, 2*jnp.pi, ntheta)
    phiBs = jnp.linspace(0,2*jnp.pi/na_eq.nfp, nzeta)
    
    rho = jnp.broadcast_to(rho[:, None, None], (nrho, ntheta, nzeta))
    thetaBs = jnp.broadcast_to(thetaBs[None, :,None], (nrho, ntheta, nzeta))
    phiBs = jnp.broadcast_to(phiBs[None, None, :], (nrho, ntheta, nzeta))
    chis = thetaBs - na_eq.get_helicity()*phiBs
    R, phiC, Z = na_eq.flux_to_cylindrical(r*rho**2, chis, phiBs, n_max=n_max)
    R, phiC, Z = R.real, phiC.real, Z.real
    grid = Grid(jnp.array([rho.flatten(), thetaBs.flatten(), phiC.flatten()]).T, sort=False)

    if not stellsym:
        raise NotImplementedError('Fitting for non-stellarator-symmetric equilibria is not implemented yet.')
    else:
        print('Assuming stellarator symmetry.')
    basis_cos = FourierZernikeBasis(
        L=L,
        M=M,
        N=N,
        NFP=na_eq.nfp,
        sym="cos", #maybe change this
        spectral_indexing=spectral_indexing,
    )
    basis_sin = FourierZernikeBasis(
        L=L,
        M=M,
        N=N,
        NFP=na_eq.nfp,
        sym="sin",
        spectral_indexing=spectral_indexing,
    )
    
    transform_cos = Transform(grid, basis_cos, method="direct1")
    transform_sin = Transform(grid, basis_sin, method="direct1")
    Acos = transform_cos.matrices['direct1'][0][0][0]
    Asin = transform_sin.matrices['direct1'][0][0][0]
    p_perp = jnp.real(na_eq.unknown['p_perp_coef_cp'].eval(rho**2*r, chis, phiBs, n_max=n_max).real / mu_0)
    delta = jnp.real(na_eq.unknown['Delta_coef_cp'].eval(rho**2*r, chis, phiBs, n_max=n_max).real)
    iota = jnp.real(na_eq.constant['iota_coef'].eval(rho[:,0,0,]**2*r, 0, 0, n_max=(n_max-1)//2) + na_eq.get_helicity())
    
    nu_B = phiBs - phiC
    lmbda = nu_B * iota[:,None,None]
    p_perp = FourierZernikeProfile.from_values(grid.nodes[:,0], grid.nodes[:,1], grid.nodes[:,2], p_perp.flatten(), L=L, M=M, N=N, NFP=na_eq.nfp, w=1/grid.nodes[:,0]**2)
    delta = FourierZernikeProfile.from_values(grid.nodes[:,0], grid.nodes[:,1], grid.nodes[:,2], delta.flatten(), L=L, M=M, N=N, NFP=na_eq.nfp, w=1/grid.nodes[:,0]**2)
    iota = PowerSeriesProfile.from_values(rho[:,0,0], iota, order=L, w=1/rho[:,0,0]**2)
    W = jnp.sqrt(jnp.diag(1/rho.flatten()**2))
    Acosw = jnp.dot(W,Acos)
    Asinw = jnp.dot(W,Asin)
    R_lmn = jnp.linalg.lstsq(Acosw, R.flatten()@W, rcond=None)[0]
    Z_lmn = jnp.linalg.lstsq(Asinw, Z.flatten()@W, rcond=None)[0]
    L_lmn = jnp.linalg.lstsq(Asinw, lmbda.flatten()@W, rcond=None)[0]
    inputs = {
        "Psi": r*2*jnp.pi,
        "NFP": na_eq.nfp,
        "L": L,
        "M": M,
        "N": N,
        "sym": True, #maybe change
        # "spectral_indexing ": spectral_indexing,
        "pressure" : p_perp,
        "iota" : iota,
        "anisotropy": delta,
        "R_lmn": R_lmn,
        "Z_lmn": Z_lmn,
        "L_lmn": L_lmn,
    }
    eq_fit = Equilibrium(**inputs)
    eq_fit.surface = eq_fit.get_surface_at(rho=1)
    eq_fit.axis = eq_fit.get_axis()
    aux_dict = {
        'rho': rho, 
        'thetaBs': thetaBs, 
        'phiBs': phiBs, 
        'psi': r*2*jnp.pi,
        'chis': chis
    }
    if solve_force_balance:    
        cons = get_NAE_constraints(eq_fit, qsc_eq=None, profiles=True, fix_lambda=True, N=eq_fit.N)
        eq_force_balance, _ = eq_fit.solve(objective=ObjectiveFunction(ForceBalanceAnisotropic(eq_fit)), constraints=cons, copy=True, **kwargs)
        return(eq_fit, eq_force_balance, aux_dict)
    print('Force balance is disabled. Please solve with a call similar to this:')
    print('eq_force_balance, _ = eq_fit.solve(objective=ObjectiveFunction(ForceBalanceAnisotropic(eq_fit)), verbose=3, maxiter=25, copy=True)')
    return(eq_fit, aux_dict)
  

def aqsc_to_desc_boundary(
    na_eq, psi_max, n_max=float('inf'), 
    M=6, N=8, solve_force_balance=True,
    stellsym=True,
    **kwargs):
    """Convert an equilibrium from AQSC to DESC.
    Parameters
    ----------
    na_eq : aqsc.Equilibrium
        aqsc equilibrium to convert
    r : float
        Value of Psi/2pi defining the LCFS of the near axis equilibrium
    M : int
        Maximum poloidal mode number of DESC equilibrium
    N : int
        Maximum toroidal mode number of DESC equilibrium
    Returns
    -------
    desc_eq : desc.equilibrium.Equilibrium
        near axis equilibrium fit to desc basis.
    """
    try:
        from desc.equilibrium import Equilibrium
        from desc.grid import Grid
        from desc.basis import FourierZernikeBasis
        from desc.transform import Transform
        from desc.profiles import FourierZernikeProfile, PowerSeriesProfile
        from desc.objectives import ForceBalanceAnisotropic, ObjectiveFunction
        # Code leveraging desc-opt
    except ImportError:
        raise ImportError("This feature requires DESC. Install it with `pip install desc-opt`.")

    r=float(psi_max) # Psi_aqsc = Psi/2pi
    L=M
    # spectral_indexing="ansi"
    
    if N is None:
        N = M
    if N == jnp.inf:
        N = int((na_eq.axis_info['phi'].shape - 1) / 2)
    
    ntheta = 4 * M + 1
    nzeta = 4 * N + 1
    nrho = 2 * L
        
    rho, _ = special.js_roots(nrho, 2, 2)
    thetaBs = jnp.linspace(0, 2*jnp.pi, ntheta)
    phiBs = jnp.linspace(0,2*jnp.pi/na_eq.nfp, nzeta)
    
    rho = jnp.broadcast_to(rho[:, None, None], (nrho, ntheta, nzeta))
    thetaBs = jnp.broadcast_to(thetaBs[None, :,None], (nrho, ntheta, nzeta))
    phiBs = jnp.broadcast_to(phiBs[None, None, :], (nrho, ntheta, nzeta))
    chis = thetaBs - na_eq.get_helicity()*phiBs
    R, phiC, Z = na_eq.flux_to_cylindrical(r*rho**2, chis, phiBs, n_max=n_max)
    R, phiC, Z = R.real, phiC.real, Z.real
    grid = Grid(jnp.array([rho.flatten(), thetaBs.flatten(), phiC.flatten()]).T, sort=False)

    if not stellsym:
        raise NotImplementedError('Fitting for non-stellarator-symmetric equilibria is not implemented yet.')
    else:
        print('Assuming stellarator symmetry.')
    
    basis_cos = FourierZernikeBasis(
        L=L,
        M=M,
        N=N,
        NFP=na_eq.nfp,
        sym="cos", #maybe change this
        # spectral_indexing=spectral_indexing,
    )
    basis_sin = FourierZernikeBasis(
        L=L,
        M=M,
        N=N,
        NFP=na_eq.nfp,
        sym="sin",
        # spectral_indexing=spectral_indexing,
    )
    
    transform_cos = Transform(grid, basis_cos, method="direct1")
    transform_sin = Transform(grid, basis_sin, method="direct1")
    Acos = transform_cos.matrices['direct1'][0][0][0]
    Asin = transform_sin.matrices['direct1'][0][0][0]
    p_perp = jnp.real(na_eq.unknown['p_perp_coef_cp'].eval(rho**2*r, chis, phiBs, n_max=n_max).real / mu_0)
    delta = jnp.real(na_eq.unknown['Delta_coef_cp'].eval(rho**2*r, chis, phiBs, n_max=n_max).real)
    iota = jnp.real(na_eq.constant['iota_coef'].eval(rho[:,0,0,]**2*r, 0, 0, n_max=(n_max-1)//2) + na_eq.get_helicity())
    
    nu_B = phiBs - phiC
    lmbda = nu_B * iota[:,None,None]
    p_perp = FourierZernikeProfile.from_values(grid.nodes[:,0], grid.nodes[:,1], grid.nodes[:,2], p_perp.flatten(), L=L, M=M, N=N, NFP=na_eq.nfp, w=1/grid.nodes[:,0]**2)
    delta = FourierZernikeProfile.from_values(grid.nodes[:,0], grid.nodes[:,1], grid.nodes[:,2], delta.flatten(), L=L, M=M, N=N, NFP=na_eq.nfp, w=1/grid.nodes[:,0]**2)
    iota = PowerSeriesProfile.from_values(rho[:,0,0], iota, order=L, w=1/rho[:,0,0]**2)
    W = jnp.sqrt(jnp.diag(1/rho.flatten()**2))
    Acosw = jnp.dot(W,Acos)
    Asinw = jnp.dot(W,Asin)
    R_lmn = jnp.linalg.lstsq(Acosw, R.flatten()@W, rcond=None)[0]
    Z_lmn = jnp.linalg.lstsq(Asinw, Z.flatten()@W, rcond=None)[0]
    L_lmn = jnp.linalg.lstsq(Asinw, lmbda.flatten()@W, rcond=None)[0]
    inputs = {
        "Psi": r * 2 * jnp.pi,
        "NFP": na_eq.nfp,
        "L": L,
        "M": M,
        "N": N,
        "sym": True, #maybe change
        # "spectral_indexing ": spectral_indexing,
        "pressure" : p_perp,
        "iota" : iota,
        "anisotropy": delta,
        "R_lmn" : R_lmn,
        "Z_lmn": Z_lmn,
        "L_lmn": L_lmn,
    }
    eq_fit = Equilibrium(**inputs)
    eq_fit.surface = eq_fit.get_surface_at(rho=1)
    eq_fit.axis = eq_fit.get_axis()
    aux_dict = {
        'rho': rho, 
        'thetaBs': thetaBs, 
        'phiBs': phiBs, 
        'psi': r * 2 * jnp.pi,
        'chis': chis
    }
    if solve_force_balance:
        eq_force_balance, _ = eq_fit.solve(objective=ObjectiveFunction(ForceBalanceAnisotropic(eq_fit)), copy=True, **kwargs)
        return(eq_fit, eq_force_balance, aux_dict)
    print('Force balance is disabled. Please solve with a call similar to this:')
    print('eq_force_balance, _ = eq_fit.solve(objective=ObjectiveFunction(ForceBalanceAnisotropic(eq_fit)), verbose=3, maxiter=25, copy=True)')
    return(eq_fit, aux_dict)
  
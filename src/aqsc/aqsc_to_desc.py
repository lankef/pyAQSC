import jax.numpy as jnp
from desc.equilibrium import Equilibrium
from desc.grid import Grid
from desc.basis import FourierZernikeBasis
from desc.transform import Transform
from desc.profiles import FourierZernikeProfile, PowerSeriesProfile
from desc.objectives import ForceBalanceAnisotropic, ObjectiveFunction
from qsc import Qsc # currently get_NAE_constraints needs a qsc equilibria so we just create a dummy one
from scipy.constants import mu_0
from scipy import special

def aqsc_to_desc_near_axis(na_eq, psi_max, n_max=float('inf'), M=6, N=8, stellsym=True, solve_force_balance=True, maxiter=100):
    '''
    M is 
    N is 
    '''

    r=float(psi_max) # Psi_aqsc = Psi/2pi
    L=None # leave this alone
    ntheta=None # leave this alone
    nzeta=None # leave this alone
    # spectral_indexing="ansi" # leave this alone

    # default resolution parameters
    if L is None:
        # if spectral_indexing == "ansi":
        L = M
        # elif spectral_indexing == "fringe":
        #     L = 2 * M
    if N is None:
        N = M
    if N == jnp.inf:
        N = int((na_eq.axis_info['phi'].shape - 1) / 2)

    if ntheta is None:
        ntheta = 4 * M + 1
    if nzeta is None:
        nzeta = 4 * N + 1

    nrho = 2*L
        
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

    psi = rho**2*r
    p_perp = na_eq.unknown['p_perp_coef_cp'].eval(psi, chis, phiBs, n_max=n_max).real / mu_0
    delta = na_eq.unknown['Delta_coef_cp'].eval(psi, chis, phiBs, n_max=n_max).real
    iota = na_eq.constant['iota_coef'].eval(rho[:,0,0,]**2*r, 0, 0, n_max=n_max) + na_eq.get_helicity()

    nu_B = phiBs - phiC
    lmbda = nu_B * iota[:,None,None]


    p_perp = FourierZernikeProfile.from_values(grid.nodes[:,0], grid.nodes[:,1], grid.nodes[:,2], p_perp.flatten(), L=L, M=M, N=N, NFP=na_eq.nfp, w=1/grid.nodes[:,0]**2)
    delta = FourierZernikeProfile.from_values(grid.nodes[:,0], grid.nodes[:,1], grid.nodes[:,2], delta.flatten(), L=L, M=M, N=N, NFP=na_eq.nfp, w=1/grid.nodes[:,0]**2)
    iota = PowerSeriesProfile.from_values(rho[:,0,0], iota, order=L, w=1/rho[:,0,0]**2)

    W = jnp.sqrt(jnp.diag(1/rho.flatten()**2))
    Acosw = jnp.dot(W,Acos)
    Asinw = jnp.dot(W,Asin)
    R_lmn = jnp.linalg.lstsq(Acosw, R.flatten()@W)[0]
    Z_lmn = jnp.linalg.lstsq(Asinw, Z.flatten()@W)[0]
    L_lmn = jnp.linalg.lstsq(Asinw, lmbda.flatten()@W)[0]

    inputs = {
        "Psi": r*2*jnp.pi,
        "NFP": na_eq.nfp,
        "L": L, # Resolution
        "M": M,
        "N": N,
        "sym": stellsym, # Stellarator symmetry
        # "spectral_indexing ": spectral_indexing,
        "pressure" : p_perp,
        "iota" : iota,
        "anisotropy": delta,
        "R_lmn" : R_lmn,
        "Z_lmn": Z_lmn,
        "L_lmn": L_lmn,
    }

    eq = Equilibrium(**inputs)
    # eq.surface = eq.get_surface_at(rho=1)
    # eq.axis = eq.get_axis()

    aux_dict = {
        'rho': rho, 
        'thetaBs': thetaBs, 
        'phiBs': phiBs, 
        'psi': psi,
        'chis': chis
    }
    if solve_force_balance:
        cons = get_NAE_constraints(eq, Qsc.from_paper("precise QA"), profiles=True)
        for con in cons:
            if hasattr(con, "_target_from_user"):
                # override the old target to target the current equilibrium
                con._target_from_user = None
            con.build()
        eq2, _ = eq.solve(objective=ObjectiveFunction(ForceBalanceAnisotropic(eq)), constraints=cons, verbose=1, maxiter=maxiter, copy=True)
    return(eq, aux_dict)



def aqsc_to_desc_boundary(na_eq, psi_max, n_max=float('inf'), M=6, N=8, solve_force_balance=True, maxiter=25):
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

    r=float(psi_max) # Psi_aqsc = Psi/2pi
    L=M
    # spectral_indexing="ansi"
    
    if N is None:
        N = M
    if N == jnp.inf:
        N = int((na_eq.axis_info['phi'].shape - 1) / 2)
    
    ntheta = 4 * M + 1
    nzeta = 4 * N + 1
    nrho = 2*L
        
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
    p_perp = na_eq.unknown['p_perp_coef_cp'].eval(rho**2*r, chis, phiBs, n_max=n_max).real / mu_0
    delta = na_eq.unknown['Delta_coef_cp'].eval(rho**2*r, chis, phiBs, n_max=n_max).real
    iota = na_eq.constant['iota_coef'].eval(rho[:,0,0,]**2*r, 0, 0, n_max=n_max) + na_eq.get_helicity()
    
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
        'psi': r*2*jnp.pi,
        'chis': chis
    }
    if solve_force_balance:
        eq_force_balance, _ = eq_fit.solve(objective=ObjectiveFunction(ForceBalanceAnisotropic(eq_fit)), verbose=3, maxiter=maxiter, copy=True)
        return(eq_fit, eq_force_balance, aux_dict)
    print('Force balance is disabled. Please solve with a call similar to this:')
    print('eq_force_balance, _ = eq_fit.solve(objective=ObjectiveFunction(ForceBalanceAnisotropic(eq_fit)), verbose=3, maxiter=25, copy=True)')
    return(eq_fit, aux_dict)
  
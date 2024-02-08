import numpy as np

def aqsc_to_desc(na_eq, psi_max, M=6, N=8):
    '''
    M is 
    N is 
    '''
    from desc.equilibrium import Equilibrium
    from desc.grid import Grid
    from desc.basis import FourierZernikeBasis
    from desc.transform import Transform
    from desc.profiles import FourierZernikeProfile, PowerSeriesProfile
    from scipy import special
    from scipy.constants import mu_0
    r=float(psi_max) # Psi_aqsc = Psi/2pi
    L=None # leave this alone
    ntheta=None # leave this alone
    nzeta=None # leave this alone
    spectral_indexing="ansi" # leave this alone

    # default resolution parameters
    if L is None:
        if spectral_indexing == "ansi":
            L = M
        elif spectral_indexing == "fringe":
            L = 2 * M
    if N is None:
        N = M
    if N == np.inf:
        N = int((na_eq.axis_info['phi'].shape - 1) / 2)

    if ntheta is None:
        ntheta = 4 * M + 1
    if nzeta is None:
        nzeta = 4 * N + 1

    nrho = 2*L
        
    rho, _ = special.js_roots(nrho, 2, 2) 
    thetaBs = np.linspace(0, 2*np.pi, ntheta)
    phiBs = np.linspace(0,2*np.pi/na_eq.nfp, nzeta)

    rho = np.broadcast_to(rho[:, None, None], (nrho, ntheta, nzeta))
    thetaBs = np.broadcast_to(thetaBs[None, :,None], (nrho, ntheta, nzeta))
    phiBs = np.broadcast_to(phiBs[None, None, :], (nrho, ntheta, nzeta))

    chis = thetaBs - na_eq.get_helicity()*phiBs
    R, phiC, Z = na_eq.flux_to_cylindrical(r*rho**2, chis, phiBs)
    R, phiC, Z = R.real, phiC.real, Z.real
    grid = Grid(np.array([rho.flatten(), thetaBs.flatten(), phiC.flatten()]).T, sort=False)

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

    psi = rho**2*r
    p_perp = na_eq.unknown['p_perp_coef_cp'].eval(psi, chis, phiBs).real / mu_0
    delta = na_eq.unknown['Delta_coef_cp'].eval(psi, chis, phiBs).real
    iota = na_eq.constant['iota_coef'].eval(rho[:,0,0,]**2*r, 0, 0, sq_eps_series=True) + na_eq.get_helicity()

    nu_B = phiBs - phiC
    lmbda = nu_B * iota[:,None,None]


    p_perp = FourierZernikeProfile.from_values(grid.nodes[:,0], grid.nodes[:,1], grid.nodes[:,2], p_perp.flatten(), L=L, M=M, N=N, NFP=na_eq.nfp, w=1/grid.nodes[:,0]**2)
    delta = FourierZernikeProfile.from_values(grid.nodes[:,0], grid.nodes[:,1], grid.nodes[:,2], delta.flatten(), L=L, M=M, N=N, NFP=na_eq.nfp, w=1/grid.nodes[:,0]**2)
    iota = PowerSeriesProfile.from_values(rho[:,0,0], iota, order=L, w=1/rho[:,0,0]**2)

    W = np.sqrt(np.diag(1/rho.flatten()**2))
    Acosw = np.dot(W,Acos)
    Asinw = np.dot(W,Asin)
    R_lmn = np.linalg.lstsq(Acosw, R.flatten()@W)[0]
    Z_lmn = np.linalg.lstsq(Asinw, Z.flatten()@W)[0]
    L_lmn = np.linalg.lstsq(Asinw, lmbda.flatten()@W)[0]

    inputs = {
        "Psi": r*2*np.pi,
        "NFP": na_eq.nfp,
        "L": L,
        "M": M,
        "N": N,
        "sym": True, #maybe change
        "spectral_indexing ": spectral_indexing,
        "pressure" : p_perp,
        "iota" : iota,
        "anisotropy": delta,
        "R_lmn" : R_lmn,
        "Z_lmn": Z_lmn,
        "L_lmn": L_lmn,
    }

    eq = Equilibrium(**inputs)
    eq.surface = eq.get_surface_at(rho=1)
    eq.axis = eq.get_axis()
    print()

    print('rho', rho.shape)
    print('thetaBs', thetaBs.shape)
    print('phiBs', phiBs.shape)
    print('psi', psi.shape)
    print('chis', chis.shape)
    aux_dict = {
        'rho': rho, 
        'thetaBs': thetaBs, 
        'phiBs': phiBs, 
        'psi': psi,
        'chis': chis
    }
    return(eq, aux_dict)
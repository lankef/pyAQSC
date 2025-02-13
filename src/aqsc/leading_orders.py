import jax.numpy as jnp
from jax.lax import scan
from jax import grad
from interpax import interp1d, fft_interp1d
from .chiphifunc import *
from .chiphiepsfunc import *
from .math_utilities import diff, newton_solver_scalar
from .looped_solver import iterate_looped
from .equilibrium import Equilibrium
from .recursion_relations import iterate_p_perp_n, iterate_delta_n_0_offset, \
    iterate_dc_B_psi_nm2, iterate_Zn_cp, iterate_Xn_cp, iterate_Yn_cp_magnetic

from .config import interp1d_method

def circular_axis():
    Rc, Rs = ([1, 0, 0.0001], [0, 0, 0])
    Zc, Zs = ([0, 0, 0], [0, 0, 0.001])
    phis_2pi = jnp.linspace(0,2*jnp.pi*0.999,1000)
    return(leading_orders(
        nfp=1,
        Rc=Rc,
        Rs=Rs,
        Zc=Zc,
        Zs=Zs,
        p0=1+0.1*jnp.cos(phis_2pi)+0.1*jnp.cos(2*phis_2pi),
        Delta_0_avg=0,
        B_theta_20_avg=1.5125089,
        # iota_0 should be 0.52564852,
        B_alpha_1=0.1,
        B0=1,
        B11c=-1.8,
        B22c=0.01, B20=0.01, B22s=0.01,
        sigma_0=0,
        len_phi=1000,
        len_phi_axis=1000,
        static_max_freq=(30, 30),
        traced_max_freq=(30, 30),
        tol_riccati=1e-8,
        max_iter_riccati=50,
        n_shooting_riccati=5000
    ))

def get_axis_info(Rc, Rs, Zc, Zs, nfp, len_phi):
    '''
    Axis length, tau and kappa
    This section is rewritten from pyQSC.qsc.init_axis for JAX. The sign of tau
    in Rodriguez 2021 is also different.
    '''
    # First, we pad zero at the end of Rc, Rs, Zc, Zs to
    # make their lengths equal
    RZ_max_len = max(
        len(Rc),
        len(Rs),
        len(Zc),
        len(Zs),
    )
    Rc_arr = jnp.zeros(RZ_max_len)
    Rs_arr = jnp.zeros(RZ_max_len)
    Zc_arr = jnp.zeros(RZ_max_len)
    Zs_arr = jnp.zeros(RZ_max_len)
    Rc_arr = Rc_arr.at[:len(Rc)].set(Rc)
    Rs_arr = Rs_arr.at[:len(Rs)].set(Rs)
    Zc_arr = Zc_arr.at[:len(Zc)].set(Zc)
    Zs_arr = Zs_arr.at[:len(Zs)].set(Zs)
    # make an array like:
    # [
    #     [0],
    #     [1*phi],
    #     [2*phi],
    #     ...
    # ]
    # where phi is the cartesian toroidal angle. Contains 2pi/nfp,
    # TODO: need to be made static.
    mode_num = jnp.arange(RZ_max_len)*nfp
    Phi0 = jnp.linspace(0,2*jnp.pi/nfp, len_phi, endpoint=False)
    d_phi = Phi0[1]-Phi0[0]
    phi_times_mode = mode_num[:, None]*Phi0[None, :]
    cos_arr = jnp.cos(phi_times_mode)
    sin_arr = jnp.sin(phi_times_mode)
    # Calculate r and z on Cartesian phi grid
    # each row of Rc_arr is a trigonometry component with a different mode number.
    R0 = jnp.sum(Rc_arr[:, None]*cos_arr+Rs_arr[:, None]*sin_arr, axis=0)[:, None]
    Z0 = jnp.sum(Zc_arr[:, None]*cos_arr+Zs_arr[:, None]*sin_arr, axis=0)[:, None]
    R0p = jnp.sum(mode_num[:,None]*(-Rc_arr[:, None]*sin_arr+Rs_arr[:, None]*cos_arr), axis=0)[:, None]
    Z0p = jnp.sum(mode_num[:,None]*(-Zc_arr[:, None]*sin_arr+Zs_arr[:, None]*cos_arr), axis=0)[:, None]
    R0pp = jnp.sum(mode_num[:,None]**2*(-Rc_arr[:, None]*cos_arr-Rs_arr[:, None]*sin_arr), axis=0)[:, None]
    Z0pp = jnp.sum(mode_num[:,None]**2*(-Zc_arr[:, None]*cos_arr-Zs_arr[:, None]*sin_arr), axis=0)[:, None]
    R0ppp = jnp.sum(mode_num[:,None]**3*(Rc_arr[:, None]*sin_arr-Rs_arr[:, None]*cos_arr), axis=0)[:, None]
    Z0ppp = jnp.sum(mode_num[:,None]**3*(Zc_arr[:, None]*sin_arr-Zs_arr[:, None]*cos_arr), axis=0)[:, None]
    # dl/dphi in cylindrical phi
    d_l_d_phi = jnp.sqrt(R0**2 + R0p**2 + Z0p**2)
    d2_l_d_phi2 = (R0*R0p + R0p*R0pp + Z0p*Z0pp)/d_l_d_phi
    # dl/dphi in Boozer coordinate
    axis_length = jnp.sum(d_l_d_phi) * d_phi * nfp
    dl_p = axis_length/jnp.pi/2
    # l on cylindrical phi grid
    # Integrating arclength with FFT.
    l_phi = jnp.real(ChiPhiFunc(d_l_d_phi.T, nfp).integrate_phi_fft(zero_avg=False).content[0])
    # The Boozer phi on cylindrical phi grids.
    phi_gbc = l_phi/dl_p
    # These are cylindrical vectors in R, phi, Z frame
    d_r_d_phi_cylindrical = jnp.concatenate([
        R0p,
        R0,
        Z0p
    ], axis=1)
    d2_r_d_phi2_cylindrical = jnp.concatenate([
        R0pp - R0,
        2 * R0p,
        Z0pp
    ], axis=1)
    d3_r_d_phi3_cylindrical = jnp.concatenate([
        R0ppp - 3 * R0p,
        3 * R0pp - R0,
        Z0ppp
    ], axis=1)
    # (db0/dl on cylindrical phi grid)
    d_tangent_d_l_cylindrical = (
        -d_r_d_phi_cylindrical * d2_l_d_phi2 / d_l_d_phi \
        + d2_r_d_phi2_cylindrical
    ) / (d_l_d_phi * d_l_d_phi)

    ''' Calculating axis quantities in cylindrical coordinate '''
    curvature = jnp.sqrt(jnp.sum(d_tangent_d_l_cylindrical**2, axis = 1))
    d_r_d_phi_cylindrical_x_d2_r_d_phi2 = jnp.cross(
        d_r_d_phi_cylindrical,
        d2_r_d_phi2_cylindrical
    )
    torsion_numerator = jnp.sum(
        d3_r_d_phi3_cylindrical*d_r_d_phi_cylindrical_x_d2_r_d_phi2,
        axis = 1
    )
    torsion_denominator = jnp.sum(d_r_d_phi_cylindrical_x_d2_r_d_phi2**2, axis=1)
    torsion = torsion_numerator / torsion_denominator

    ''' Calculating basis '''
    # tangent unit vector b0
    tangent_cylindrical = (d_r_d_phi_cylindrical/d_l_d_phi)
    normal_cylindrical = (d_tangent_d_l_cylindrical / curvature[:, None])
    binormal_cylindrical = jnp.cross(tangent_cylindrical, normal_cylindrical)

    ''' Calculating axis quantities in Boozer coordinate '''
    # Although Phi0 will be output as the cylindrical phi,
    # it can be reused as the Boozer phi grid because both 
    # uses the same uniformly spaced endpoint grids.
    # kap_p_content = jnp.interp(Phi0, phi_gbc, curvature, period = 2*jnp.pi/nfp)[None, :]
    print('Default interpolation method:', interp1d_method)
    kap_p_content = interp1d(Phi0, phi_gbc, curvature, period=2*jnp.pi/nfp, method=interp1d_method)[None, :]
    kap_p = ChiPhiFunc(kap_p_content, nfp)
    # Note: Rodriguez's paper uses an opposite sign for tau compared to Landreman's.
    # tau_p_content = -jnp.interp(Phi0, phi_gbc, torsion, period = 2*jnp.pi/nfp)[None, :]
    tau_p_content = -interp1d(Phi0, phi_gbc, torsion, period=2*jnp.pi/nfp, method=interp1d_method)[None, :]
    tau_p = ChiPhiFunc(tau_p_content, nfp)
    # Storing axis info. All quantities are identically defined to pyQSC.
    axis_info = {
        'dl_p': dl_p, # Checked
        'kap_p': kap_p, # Done
        'tau_p': tau_p, # Done
        'phi_gbc': phi_gbc, # Boozer phi
        'Phi0': Phi0, # Cylindrical Phi
        'R0': R0[:, 0], # Checked
        'Z0': Z0[:, 0], # Checked
        # Derivatives of R and Z in term of Phi
        'R0p': R0p[:, 0], # Checked
        'Z0p': Z0p[:, 0], # Checked
        'R0pp': R0pp[:, 0], # Checked
        'Z0pp': Z0pp[:, 0], # Checked
        'R0ppp': R0ppp[:, 0], # Checked
        'Z0ppp': Z0ppp[:, 0], # Checked
        'd_l_d_Phi_cylindrical': d_l_d_phi[:, 0], # Checked
        'axis_length': axis_length, # Checked
        'curvature': curvature, # Checked
        'torsion': torsion, # Checked
        'tangent_cylindrical': tangent_cylindrical, # axis=1 is R, phi, Z, Checked
        'normal_cylindrical': normal_cylindrical, # axis=1 is R, phi, Z, Checked
        'binormal_cylindrical': binormal_cylindrical, # axis=1 is R, phi, Z, Checked
                # 'd_phi': d_phi, # Grid spacing. Checked.
    }
    return(axis_info)

def leading_orders_magnetic(
    nfp, # Field period
    Rc, Rs, Zc, Zs, # Axis shape
    iota_0, # On-axis rotational transform
    B_theta_20, # Average B_theta[2,0]
    B_psi_00,
    Y20,
    B_alpha_1,  # B_alpha
    B0, B11c, B22s, B20, B22c,
    len_phi,
    static_max_freq,
    traced_max_freq):
    axis_info = get_axis_info(Rc, Rs, Zc, Zs, nfp, len_phi)
    return(leading_orders_magnetic_from_axis(
        nfp=nfp, # Field period
        axis_info=axis_info,
        iota_0=iota_0, # On-axis rotational transform
        B_theta_20=B_theta_20, # Average B_theta[2,0]
        B_psi_00=B_psi_00,
        Y20=Y20,
        B_alpha_1=B_alpha_1,  # B_alpha
        B0=B0, B11c=B11c, B22s=B22s, B20=B20, B22c=B22c,
        len_phi=len_phi,
        static_max_freq=static_max_freq,
        traced_max_freq=traced_max_freq,
    ))

def leading_orders_magnetic_from_axis(
    nfp, # Field period
    axis_info,
    iota_0, # On-axis rotational transform
    B_theta_20, # Average B_theta[2,0]
    B_psi_00,
    Y20,
    B_alpha_1,  # B_alpha
    B0, B11c, B22s, B20, B22c,
    len_phi,
    static_max_freq,
    traced_max_freq):
    dl_p = axis_info['dl_p'] 
    kap_p = axis_info['kap_p'] 
    tau_p = axis_info['tau_p'] 
    # The following variables will not be included in a pyAQSC equilibrium.
    # self.G0 = G0 # NA. GBC is different from Boozer Coordinate.
    # self.Bbar = self.spsi * self.B0 # NA
    # self.abs_G0_over_B0 = abs_G0_over_B0 # NA
    # self.X11s = np.zeros(nphi) # will be provided in other formats
    # self.X11c = self.etabar / curvature # will be provided in other formats
    # self.min_R0 = fourier_minimum(self.R0)
    ''' 0th order quantities '''
    B_alpha0 = dl_p/jnp.sqrt(B0) # (Rodriguez 2021, J0)
    B1 = ChiPhiFunc(
        jnp.array([
            [0], # Choice of angular coordinate. See eq II.
            [B11c]
        ]), nfp, trig_mode=True
    )
    B2 = ChiPhiFunc(
        jnp.array([
            [B22s],
            [B20],
            [B22c]
        ]), nfp, trig_mode=True
    )
    eta = -B11c/(2*B0) # Defined for simple notation. (Rodriguez 2021, eq. 14)

    ''' 1st order quantities '''
    iota_coef = ChiPhiEpsFunc([iota_0], nfp, True)
    B_denom_coef_c = ChiPhiEpsFunc([B0, B1, B2], nfp, False)
    B_alpha_coef = ChiPhiEpsFunc([B_alpha0, B_alpha_1], nfp, True)
    B_theta_coef_cp = ChiPhiEpsFunc([ChiPhiFuncSpecial(0), ChiPhiFuncSpecial(0)], nfp, False)
    B_psi_coef_cp = ChiPhiEpsFunc([], nfp, False)
    Y_coef_cp = ChiPhiEpsFunc([ChiPhiFuncSpecial(0)], nfp, False)
    Z_coef_cp = ChiPhiEpsFunc([ChiPhiFuncSpecial(0), ChiPhiFuncSpecial(0)], nfp, False)
    # X1 (Rodriguez 2021, eq. 14)
    X11c = eta/kap_p
    X1 = ChiPhiFunc(jnp.array([
        jnp.zeros_like(X11c.content[0]), # sin coeff is zero
        X11c.content[0],
    ]), nfp, trig_mode = True).filter(traced_max_freq[0])
    X_coef_cp = ChiPhiEpsFunc([ChiPhiFuncSpecial(0), X1], nfp, False)
    # p_1 and Delta1 has the same formula as higher orders.

    '''
    Leading order 'looped' equations.
    The looped equations at higher even orders is composed of
    (Rodriguez 2021, eq. tilde II)'s m!=0 components,
    (Rodriguez 2021, eq. II)'s m=0 component,
    (Rodriguez 2021, eq. D3)'s m=0 component,
    At the leading order, (Rodriguez 2021, eq. tilde II, m!=0)
    vanish, (Rodriguez 2021, eq. II, m=0) is a 1st order, inhomogeneous linear
    ODE containing only B_theta[2,0] with no unique sln.
    (Rodriguez 2021, eq. D3, m=0) is a Riccati equation (Rodriguez 2021, eq. 26)
    of Yc[1,1] and contains B_theta[2,0] in the inhomogeneity. The following
    section solves II, m=0 with spectral method given average B_theta[2,0], and
    then solves the linear 2nd order homogenous form of D3 for Yc[1,1].
    '''
    ''' II m = 0 '''
    short_length = static_max_freq[0]*2
    # RHS of II[1][0]
    # Creating differential operator and convolution operator
    # as in solve_ODE
    diff_matrix = fft_dphi_op(short_length)*nfp
    B_theta_coef_cp = B_theta_coef_cp.append(ChiPhiFunc(B_theta_20[None, :], nfp))

    ''' D3 m = 0 '''
    Y11s = 2*jnp.sqrt(B0)/eta*kap_p
    # D3 can be written as y' = q0 + q1y + q2y^2
    q0 = -iota_0*(
        2*jnp.sqrt(B0)/eta*kap_p
        +eta**3/(2*jnp.sqrt(B0)*kap_p**3)
    )+dl_p*(2*tau_p+B_theta_coef_cp[2])*eta/kap_p
    q1 = kap_p.dphi()/kap_p
    q2 = -iota_0*eta/(2*jnp.sqrt(B0)*kap_p)
    # This equation is equivalent to the 2nd order linear ODE:
    # u''-R(x)u'+S(x)u=0, where y =
    S_lin = q0*q2
    R_lin = q1+q2.dphi()/q2
    u_avg = 1 # Doesn't actually impact Y! That's crazy.
    # The differential operator is:
    R_fft = fft_filter(jnp.fft.fft(R_lin.content[0]), short_length, axis=0)
    S_fft = fft_filter(jnp.fft.fft(S_lin.content[0]), short_length, axis=0)
    R_conv_matrix = fft_conv_op(R_fft)
    S_conv_matrix = fft_conv_op(S_fft)
    riccati_matrix = diff_matrix**2 - R_conv_matrix@diff_matrix + S_conv_matrix
    # BC
    riccati_matrix = riccati_matrix.at[:, 0].set(riccati_matrix[:, 0]+1)
    riccati_RHS = jnp.ones(short_length)*u_avg*short_length
    # Solution
    riccati_sln_fft = jnp.linalg.solve(riccati_matrix, riccati_RHS)
    riccati_u = ChiPhiFunc(jnp.fft.ifft(fft_pad(riccati_sln_fft, len_phi, axis=0), axis=0)[None, :], nfp)
    Y11c = (-riccati_u.dphi()/(q2*riccati_u))
    Y1 = ChiPhiFunc(jnp.array([
        Y11s.content[0], # sin coeff is zero
        Y11c.content[0],
    ]), nfp, trig_mode = True).filter(traced_max_freq[0])
    Y_coef_cp = Y_coef_cp.append(Y1)

    ''' 2nd order quantities '''
    B_psi_nm2 = iterate_dc_B_psi_nm2(n_eval=2,
        X_coef_cp=X_coef_cp,
        Y_coef_cp=Y_coef_cp,
        Z_coef_cp=Z_coef_cp,
        B_theta_coef_cp=B_theta_coef_cp,
        B_psi_coef_cp=B_psi_coef_cp,
        B_alpha_coef=B_alpha_coef,
        B_denom_coef_c=B_denom_coef_c,
        kap_p=kap_p,
        dl_p=dl_p,
        tau_p=tau_p,
        iota_coef=iota_coef
        ).antid_chi()
    B_psi_nm2_content_new = B_psi_nm2.content.at[B_psi_nm2.content.shape[0]//2].set(B_psi_00)
    B_psi_nm2 = ChiPhiFunc(B_psi_nm2_content_new, B_psi_nm2.nfp)
    B_psi_coef_cp = B_psi_coef_cp.append(B_psi_nm2.filter(traced_max_freq[1]))

    Zn = iterate_Zn_cp(n_eval=2,
        X_coef_cp=X_coef_cp,
        Y_coef_cp=Y_coef_cp,
        Z_coef_cp=Z_coef_cp,
        B_theta_coef_cp=B_theta_coef_cp,
        B_psi_coef_cp=B_psi_coef_cp,
        B_alpha_coef=B_alpha_coef,
        kap_p=kap_p,
        dl_p=dl_p,
        tau_p=tau_p,
        iota_coef=iota_coef
        )
    Z_coef_cp = Z_coef_cp.append(Zn.filter(traced_max_freq[1]))

    Xn = iterate_Xn_cp(n_eval=2,
        X_coef_cp=X_coef_cp,
        Y_coef_cp=Y_coef_cp,
        Z_coef_cp=Z_coef_cp,
        B_denom_coef_c=B_denom_coef_c,
        B_alpha_coef=B_alpha_coef,
        kap_p=kap_p,
        dl_p=dl_p,
        tau_p=tau_p,
        iota_coef=iota_coef
        )
    X_coef_cp = X_coef_cp.append(Xn.filter(traced_max_freq[1]))

    Yn = iterate_Yn_cp_magnetic(n_unknown=2,
        X_coef_cp=X_coef_cp,
        Y_coef_cp=Y_coef_cp,
        Z_coef_cp=Z_coef_cp,
        B_psi_coef_cp=B_psi_coef_cp,
        B_theta_coef_cp=B_theta_coef_cp,
        B_alpha_coef=B_alpha_coef,
        B_denom_coef_c=B_denom_coef_c,
        kap_p=kap_p,
        dl_p=dl_p,
        tau_p=tau_p,
        iota_coef=iota_coef,
        static_max_freq=static_max_freq[1],
        Yn0=Y20
    )
    Y_coef_cp = Y_coef_cp.append(Yn.filter(traced_max_freq[1]))


    ''' Constructing equilibrium '''
    equilibrium_out = Equilibrium.from_known(
        X_coef_cp=X_coef_cp.mask(2),
        Y_coef_cp=Y_coef_cp.mask(2),
        Z_coef_cp=Z_coef_cp.mask(2),
        B_psi_coef_cp=B_psi_coef_cp.mask(0),
        B_theta_coef_cp=B_theta_coef_cp.mask(2),
        B_denom_coef_c=B_denom_coef_c.mask(2),
        B_alpha_coef=B_alpha_coef.mask(1),
        iota_coef=iota_coef.mask(0),
        kap_p=kap_p,
        dl_p=dl_p,
        tau_p=tau_p,
        p_perp_coef_cp=None, # no pressure or delta
        Delta_coef_cp=None,
        axis_info=axis_info,
        magnetic_only=True
    )

    return(equilibrium_out)

def leading_orders(
    nfp, # Field period
    Rc, Rs, Zc, Zs, # Axis shape
    p0, # On-axis pressure
    Delta_0_avg, # Average anisotropy on axis
    B_alpha_1,  # B_alpha
    B0, B11c, 
    B22c, B20, B22s, # Magnetic field strength
    sigma_0=0,
    iota_0=None,
    B_theta_20_avg=None,
    len_phi=1000,
    len_phi_axis=1000,
    static_max_freq=(50, 50),
    traced_max_freq=(50, 50),
    tol_riccati=1e-8,
    max_iter_riccati=50,
    n_shooting_riccati=5000):
    axis_info = get_axis_info(Rc, Rs, Zc, Zs, nfp, len_phi_axis)
    return(
        leading_orders_from_axis(
            nfp=nfp, # Field period
            axis_info=axis_info, # Axis shape
            p0=p0, # On-axis pressure
            Delta_0_avg=Delta_0_avg, # Average anisotropy on axis
            B_alpha_1=B_alpha_1,  # B_alpha
            B0=B0, B11c=B11c, 
            B22c=B22c, B20=B20, B22s=B22s, # Magnetic field strength
            sigma_0=sigma_0,
            iota_0=iota_0,
            B_theta_20_avg=B_theta_20_avg,
            len_phi=len_phi,
            static_max_freq=static_max_freq,
            traced_max_freq=traced_max_freq,
            tol_riccati=tol_riccati,
            max_iter_riccati=max_iter_riccati,
            n_shooting_riccati=n_shooting_riccati,
        )
    )

def leading_orders_from_axis(
    nfp, # Field period
    axis_info, # Axis shape
    p0, # On-axis pressure
    Delta_0_avg, # Average anisotropy on axis
    B_alpha_1,  # B_alpha
    B0, B11c, 
    B22c, B20, B22s, # Magnetic field strength
    sigma_0=0, # initial tilt. Non-zero values break stellarator symmetry
    iota_0=None,
    B_theta_20_avg=None,
    len_phi=100,
    static_max_freq=(50, 50),
    traced_max_freq=(50, 50),
    tol_riccati=1e-8,
    max_iter_riccati=50,
    n_shooting_riccati=5000):


    if static_max_freq[0]<=0:
        static_max_freq = (len_phi//2, static_max_freq[1])
    if static_max_freq[1]<=0:
        static_max_freq = (static_max_freq[0], len_phi//2)
    if len_phi%2!=0:
        raise ValueError('Total grid number must be a even number.')
    if not ((iota_0 is None) ^ (B_theta_20_avg is None)):
        raise ValueError('Please provide one and only one of iota_0 and average B_theta_20.')
    solve_B_theta_20_avg = iota_0 is not None
    dl_p = axis_info['dl_p']
    kap_p = axis_info['kap_p'].filter_reduced_length(len_phi//2)
    tau_p = axis_info['tau_p'].filter_reduced_length(len_phi//2)
    # The following variables will not be included in a pyAQSC equilibrium.
    # self.G0 = G0 # NA. GBC is different from Boozer Coordinate.
    # self.Bbar = self.spsi * self.B0 # NA
    # self.abs_G0_over_B0 = abs_G0_over_B0 # NA
    # self.X11s = np.zeros(nphi) # will be provided in other formats
    # self.X11c = self.etabar / curvature # will be provided in other formats
    # self.min_R0 = fourier_minimum(self.R0)
    ''' 0th order quantities '''
    B_alpha0 = dl_p/jnp.sqrt(B0) # (Rodriguez 2021, J0)
    B1 = ChiPhiFunc(
        jnp.array([
            [0], # Choice of angular coordinate. See eq II.
            [B11c]
        ]), nfp, trig_mode=True
    )
    B2 = ChiPhiFunc(
        jnp.array([
            [B22s],
            [B20],
            [B22c]
        ]), nfp, trig_mode=True
    )
    p0 = p0 * jnp.ones(len_phi)
    p0_chiphifunc = ChiPhiFunc(p0[None, :], nfp=nfp)
    Delta0 = (-B0*p0_chiphifunc) - phi_avg(-B0*p0_chiphifunc) + Delta_0_avg # (Rodriguez 2021, eq. 41)
    eta = -B11c/(2*B0) # Defined for simple notation. (Rodriguez 2021, eq. 14)

    ''' 1st order quantities '''
    # Either iota or B_theta_20_avg can be the free scalar parameter.
    # - iota first appears in Delta1.
    # - B_theta_20_avg first appears in the B_theta ODE.
    # So both Delta1 and the B_theta ODE need to be packaged 
    # for Newton iteration.
    B_denom_coef_c = ChiPhiEpsFunc([B0, B1, B2], nfp, False)
    B_alpha_coef = ChiPhiEpsFunc([B_alpha0, B_alpha_1], nfp, True)
    B_theta_coef_cp = ChiPhiEpsFunc([ChiPhiFuncSpecial(0), ChiPhiFuncSpecial(0)], nfp, False)
    B_psi_coef_cp = ChiPhiEpsFunc([], nfp, False)
    Delta_coef_cp = ChiPhiEpsFunc([Delta0], nfp, False)
    p_perp_coef_cp = ChiPhiEpsFunc([p0_chiphifunc], nfp, False)
    Y_coef_cp = ChiPhiEpsFunc([ChiPhiFuncSpecial(0)], nfp, False)
    Z_coef_cp = ChiPhiEpsFunc([ChiPhiFuncSpecial(0), ChiPhiFuncSpecial(0)], nfp, False)
    # X1 (Rodriguez 2021, eq. 14)
    X11c = eta/kap_p
    X1 = ChiPhiFunc(jnp.array([
        jnp.zeros_like(X11c.content[0]), # sin coeff is zero
        X11c.content[0],
    ]), nfp, trig_mode = True).filter(traced_max_freq[0])
    X_coef_cp = ChiPhiEpsFunc([ChiPhiFuncSpecial(0), X1], nfp, False)
    # p1 and Delta1 has the same formula as higher orders.
    # p1 is not dependent on iota0. (higher order has iota 
    # dependence, though)
    p1 = iterate_p_perp_n(
        1,
        B_theta_coef_cp,
        B_psi_coef_cp,
        B_alpha_coef,
        B_denom_coef_c,
        p_perp_coef_cp,
        Delta_coef_cp,
        ChiPhiEpsFunc([0], nfp, True) # No iota dependence at leading order.
    ).filter(traced_max_freq[0])
    p_perp_coef_cp = p_perp_coef_cp.append(p1)


    '''
    Leading order 'looped' equations.
    The looped equations at higher even orders is composed of
    (Rodriguez 2021, eq. tilde II)'s m!=0 components,
    (Rodriguez 2021, eq. II)'s m=0 component,
    (Rodriguez 2021, eq. D3)'s m=0 component,
    At the leading order, (Rodriguez 2021, eq. tilde II, m!=0)
    vanish, (Rodriguez 2021, eq. II, m=0) is a 1st order, inhomogeneous linear
    ODE containing only B_theta[2,0] with no unique sln.
    (Rodriguez 2021, eq. D3, m=0) is a Riccati equation (Rodriguez 2021, eq. 26)
    of Yc[1,1] and contains B_theta[2,0] in the inhomogeneity. The following
    section solves II, m=0 with spectral method given average B_theta[2,0], and
    then solves the linear 2nd order homogenous form of D3 for Yc[1,1].
    '''
    Y11s = 2*jnp.sqrt(B0)/eta*kap_p
    Y11c0 = jnp.real(Y11s.content[0,0])*sigma_0
    def Delta1_from_iota(iota_temp):
        # First appearance of iota0
        # Delta1 is dependent on iota0.
        # Sinde Delta comes from an ODE solve, it's 
        # linearly dependent on iota but the dependence 
        # doesn't have a easily obtainable symbolic formula.
        iota_coef = ChiPhiEpsFunc([iota_temp], nfp, True)
        Delta1 = iterate_delta_n_0_offset(
            1,
            B_denom_coef_c,
            p_perp_coef_cp,
            Delta_coef_cp,
            iota_coef,
            static_max_freq=None,
            no_iota_masking=True
        ).filter(traced_max_freq[0])
        
        return(Delta1)

    ''' II m = 0 '''
    def calculate_B_theta_20(Delta_coef_cp_temp, B_theta_20_avg):
        # Solving for B_theta 20. This equation will be solved 
        # using integrating factor because a periodic solution 
        # is not guaranteed to exist for incorrect iota.
        # RHS of II[1][0]
        II_2_inhomog = -B_alpha_coef[0]/2*(
            4*B0*B1*p_perp_coef_cp[1].dchi()
            -Delta_coef_cp_temp[1]*B1.dchi()
        )[0]
        # Coefficients of B_theta
        coef_B_theta_20 = -B0**2*diff(p_perp_coef_cp[0],False,1)
        coef_dp_B_theta_20 = B0*(Delta_coef_cp_temp[0]-1)
        # Solving y'+py=f for B_theta[2,0]. 
        p_eff = coef_B_theta_20/coef_dp_B_theta_20
        f_eff = II_2_inhomog/coef_dp_B_theta_20
        # # Not necessarily periodic
        int_p = p_eff.integrate_phi_fft(zero_avg=False)
        e_int_p = int_p.exp()
        e_int_mp = (-int_p).exp()
        # General solution: y_0+C*y_1
        y_0 = e_int_mp*(f_eff*e_int_p).integrate_phi_fft(zero_avg=False)
        y_1 = e_int_mp
        
        C_y_1 = (B_theta_20_avg-jnp.average(y_0.content))/jnp.average(y_1.content)
        B_theta_20 = y_0 + C_y_1*y_1
        return(B_theta_20)
        
    ''' D3 m = 0 '''
    def get_riccati_coeffs(
        iota_0, 
        B_theta_20):
        # Calculate Riccati equation coefficients.
        # These coefficients are dependent on iota_0 and B_theta_20,
        # and B_theta_20 depends on iota_0 or B_theta_20_avg.
        # D3 can be written as y' = q0 + q1y + q2y^2
        q0 = -iota_0*(
            2*jnp.sqrt(B0)/eta*kap_p
            +eta**3/(2*jnp.sqrt(B0)*kap_p**3)
        )+dl_p*(2*tau_p+B_theta_20)*eta/kap_p
        q1 = kap_p.dphi()/kap_p
        q2 = -iota_0*eta/(2*jnp.sqrt(B0)*kap_p)
        return(q0, q1, q2)
    # Attempting to solve the problem with RK4
    # First, y' = f(i, y) = q0 + q1y + q2y^2.
    # For improved speed, we use phi = (0, 2dphi, 4dphi, ...)
    # as RK4 grids, and phi = (dphi, 3dphi, ...) as half-grids.
    # Under this condition, h=2pi/nfp/len_phi*2
    def Y11c_RK4(q0, q1, q2):
        q0_real = jnp.real(q0.content[0])
        q1_real = jnp.real(q1.content[0])
        q2_real = jnp.real(q2.content[0])
        q0_real = (fft_interp1d(q0_real, n_shooting_riccati))
        q1_real = (fft_interp1d(q1_real, n_shooting_riccati))
        q2_real = (fft_interp1d(q2_real, n_shooting_riccati))
        q0_real = jnp.pad(q0_real, [0, 2], mode='wrap')
        q1_real = jnp.pad(q1_real, [0, 2], mode='wrap')
        q2_real = jnp.pad(q2_real, [0, 2], mode='wrap')
        def f_RK4(ind:int, y):
            return(q0_real[ind] + q1_real[ind]*y + q2_real[ind]*y**2)
        # the 
        h = 2 * jnp.pi / nfp / n_shooting_riccati * 2
        def iterate_y(carry, t):
            y_n, ind = carry
            # i+1 correspomds to phi+h/2, the half-grid value
            k1 = f_RK4(ind, y_n)
            k2 = f_RK4(ind+1, y_n+h*k1/2) 
            k3 = f_RK4(ind+1, y_n+h*k2/2)
            k4 = f_RK4(ind+2, y_n+h*k3)
            y_np1 = y_n + h/6*(k1+2*k2+2*k3+k4)
            return((y_np1, ind+2), y_np1)

        ind_list = jnp.arange(0, n_shooting_riccati + 2,2)
        (_, _), y_arr = scan(f=iterate_y, init=(Y11c0, 0), xs=ind_list)
        # y_arr contains 
        # y1, ..., yn, yn+1, yn+2
        # The correct output must contain
        # y0, ..., yn.
        out = jnp.concatenate((jnp.array([Y11c0]), y_arr[:-2]))
        
        # This is an objective function that enforces
        # the boundary condition.
        # For some iota and y(0) the solution is exponentially growing
        # and can grow to infinity. This takes care of it numerically.
        # Returns the array's max abs if array's last item is not
        # a finite number. 
        diff = y_arr[0] - y_arr[-1] + Y11c0 - y_arr[-2]
        # Replaces all inf and nan's with nan
        # y_fin = jnp.where(jnp.isfinite(y_arr), y_arr, jnp.nan)
        # ymax-ymin >= |y(2pi)-y(0)| >= 0
        # y_max = jnp.nanmax(y_fin)
        # y_min = jnp.nanmin(y_fin)
        return(diff, out)
        # return(jnp.nanmin(jnp.array([
        #     # When y_arr contains no nan, 
        #     # gives error or deviation from periodic BC
        #     # Otherwise will be nan
        #     jnp.log10(diff+1),
        #     # diff,
        #     jnp.log10(y_max - y_min + 1)
        # ])), y_arr)
    # Defining the function to be evaluated in the 
    # Newton method loop, based on whether iota or B_theta is unknown.
    # Putting the if statement here saves some JIT time during loop unpacking.
    # Written this way in case other better ways for solving the Riccati equation
    # is found.
    if solve_B_theta_20_avg:
        print('Solving for average B_{\\theta20} self-consistently.')
        # When iota is given, Delta is constant 
        # during the iteration.
        Delta1 = Delta1_from_iota(iota_temp=iota_0)
        # @jit
        def newton_step(x_newton, obj_callable):
            # First calculate B_theta 
            # from yet unknown average
            B_theta_20 = calculate_B_theta_20(
                Delta_coef_cp_temp=Delta_coef_cp.append(Delta1), 
                B_theta_20_avg=x_newton,
            )
            q0, q1, q2 = get_riccati_coeffs(iota_0=iota_0, B_theta_20=B_theta_20)
            f_newton, Y11c_RK4 = obj_callable(q0, q1, q2)
            return(f_newton, Y11c_RK4, Delta1, B_theta_20)
    else:
        print('Solving for \\bar{\\iota}_0 self-consistently.')
        # When iota is unknown, Delta changes during 
        # the iteration.
        # @jit
        def newton_step(x_newton, obj_callable):
            # First calculate delta from yet unknown iota.
            Delta1 = Delta1_from_iota(iota_temp=x_newton)
            # Then calculate B_theta 
            # from a given average
            B_theta_20 = calculate_B_theta_20(
                Delta_coef_cp_temp=Delta_coef_cp.append(Delta1), 
                B_theta_20_avg=B_theta_20_avg,
            )
            # B_theta_20.display_content()
            q0, q1, q2 = get_riccati_coeffs(iota_0=x_newton, B_theta_20=B_theta_20)
            f_newton, Y11c_RK4 = obj_callable(q0, q1, q2)
            return(f_newton, Y11c_RK4, Delta1, B_theta_20)
        
    # Stage-1 iteration: Crude estimate for iota/avg B_theta20 with 
    # RK4. Shooting method. This is treated as a zero-finding problem.
    f1 = lambda x: newton_step(x, Y11c_RK4)[0]
    g1 = grad(f1)
    tol1 = tol_riccati * jnp.max(jnp.abs(Y11s.content))
    # The gradient is nan at exactly x==0 because of case handling in 
    # solve_ODE. This hacky approach works but change solve_ODE to 
    # be always differentiable when you have time.
    x_RK4 = newton_solver_scalar(f1, g1, 1e-15, tol=tol1, max_iter=max_iter_riccati)
    f_newton, Y11c_RK4, Delta1, B_theta_20 = newton_step(x_RK4, Y11c_RK4)
    if len(Y11c_RK4) != len_phi:
        Y11c_RK4 = fft_interp1d(Y11c_RK4, len_phi)
    # So far, the Riccati equation is successfully solved
    Y1 = ChiPhiFunc(jnp.array([
        Y11s.content[0], # sin coeff is zero
        Y11c_RK4,
    ]), nfp, trig_mode = True).filter(traced_max_freq[0])
    Y_coef_cp = Y_coef_cp.append(Y1)
    Delta_coef_cp = Delta_coef_cp.append(Delta1)
    B_theta_coef_cp = B_theta_coef_cp.append(B_theta_20)
    # Giving value to iota
    if solve_B_theta_20_avg:
        iota_coef = ChiPhiEpsFunc([iota_0], nfp, True)
    else:
        iota_coef = ChiPhiEpsFunc([x_RK4], nfp, True)

    ''' 2nd order quantities '''
    # Starting from order 2, the general recursion relations apply.
    solution2 = iterate_looped(
        n_unknown=2, 
        static_max_freq=static_max_freq[1], 
        traced_max_freq=traced_max_freq[1], 
        target_len_phi=len_phi,
        X_coef_cp=X_coef_cp,
        Y_coef_cp=Y_coef_cp,
        Z_coef_cp=Z_coef_cp,
        p_perp_coef_cp=p_perp_coef_cp,
        Delta_coef_cp=Delta_coef_cp,
        B_psi_coef_cp=B_psi_coef_cp,
        B_theta_coef_cp=B_theta_coef_cp,
        B_alpha_coef=B_alpha_coef,
        B_denom_coef_c=B_denom_coef_c,
        kap_p=kap_p,
        tau_p=tau_p,
        dl_p=dl_p,
        iota_coef=iota_coef,
        nfp=nfp,
    )
    B_psi_coef_cp = B_psi_coef_cp.append(solution2['B_psi_nm2'])
    X_coef_cp = X_coef_cp.append(solution2['Xn'])
    Y_coef_cp = Y_coef_cp.append(solution2['Yn'])
    Z_coef_cp = Z_coef_cp.append(solution2['Zn'])
    p_perp_coef_cp = p_perp_coef_cp.append(solution2['pn'])
    Delta_coef_cp = Delta_coef_cp.append(solution2['Deltan'])

    # ''' Constructing equilibrium '''
    equilibrium_out = Equilibrium.from_known(
        X_coef_cp=X_coef_cp.mask(2),
        Y_coef_cp=Y_coef_cp.mask(2),
        Z_coef_cp=Z_coef_cp.mask(2),
        B_psi_coef_cp=B_psi_coef_cp.mask(0),
        B_theta_coef_cp=B_theta_coef_cp.mask(2),
        B_denom_coef_c=B_denom_coef_c.mask(2),
        B_alpha_coef=B_alpha_coef.mask(1),
        iota_coef=iota_coef.mask(0),
        kap_p=kap_p,
        dl_p=dl_p,
        tau_p=tau_p,
        p_perp_coef_cp=p_perp_coef_cp.mask(2), # no pressure or delta
        Delta_coef_cp=Delta_coef_cp.mask(2),
        axis_info=axis_info,
        magnetic_only=False
    )
    return(equilibrium_out)

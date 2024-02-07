import jax.numpy as jnp
from .chiphifunc import *
from .chiphiepsfunc import *
from .math_utilities import diff
from .looped_solver import iterate_looped
from .equilibrium import Equilibrium
from .MHD_parsed import eval_inhomogenous_Delta_n_cp
from .recursion_relations import iterate_p_perp_n, iterate_delta_n_0_offset, \
    iterate_dc_B_psi_nm2, iterate_Zn_cp, iterate_Xn_cp, iterate_Yn_cp_magnetic

# Generate the circular axis case in Rodriguez Bhattacharjee
def circular_axis_legacy():
    Rc, Rs = ([1, 0, 0.0001], [0, 0, 0])
    Zc, Zs = ([0, 0, 0], [0, 0, 0.001])
    phis = jnp.linspace(0,2*jnp.pi*0.999,1000)
    return(leading_orders_legacy(
        nfp=1,
        Rc=Rc,
        Rs=Rs,
        Zc=Zc,
        Zs=Zs,
        p0=1+0.1*jnp.cos(phis)+0.1*jnp.cos(2*phis),
        Delta_0_avg=0,
        iota_0=0.52564852,
        B_theta_20_avg=1.5125089,
        B_alpha_1=0.1,
        B0=1,
        B11c=-1.8,
        B22c=0.01, B20=0.01, B22s=0.01,
        len_phi=1000,
        static_max_freq=(15, 20),
        traced_max_freq=(15, 20)
    ))

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
        len_phi=1000,
        static_max_freq=(15, 20),
        traced_max_freq=(15, 20),
        riccati_secant_n_iter=(15, 20)
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
    # TODO: need to made static.
    mode_num = jnp.arange(RZ_max_len)*nfp
    phi_grids = jnp.linspace(0,2*jnp.pi/nfp*(len_phi-1)/len_phi, len_phi)
    d_phi = phi_grids[1]-phi_grids[0]
    phi_times_mode = mode_num[:, None]*phi_grids[None, :]

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

    # l on cartesian phi grid
    # Setting the first element to 0. Removing the last element.
    l_phi = jnp.cumsum(d_l_d_phi)/len_phi*jnp.pi*2/nfp
    l_phi = jnp.roll(l_phi, 1)
    l_phi = l_phi.at[0].set(0)

    # The Boozer phi on cartesian phi grids.
    varphi = l_phi/dl_p

    # d_l_d_phi_wrapped = np.concatenate([d_l_d_phi, [d_l_d_phi[0]]])
    # d_l_d_phi_spline = scipy.interpolate.CubicSpline(np.linspace(0,2*np.pi/nfp, len_phi+1), d_l_d_phi_wrapped, bc_type = 'periodic')
    # dl_p_spline.integrate(0, 2*np.pi)/jnp.pi/2 # No more accurate than the sum version.

    # dphi/dl
    # dphidl = 1/d_l_d_phi

    # These are cartesian vectors in R, phi, Z frame
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
    # d2r0dphi2 = jnp.array([
    #     R0pp,
    #     jnp.zeros_like(R0pp),
    #     Z0pp
    # ])


    # (db0/dl on cartesian phi grid)
    d_tangent_d_l_cylindrical = (
        -d_r_d_phi_cylindrical * d2_l_d_phi2 / d_l_d_phi \
        + d2_r_d_phi2_cylindrical
    ) / (d_l_d_phi * d_l_d_phi)

    ''' Calculating axis quantities in cartesian coordinate '''
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
    # Although phi_grids will be output as the Cartesian phi,
    # it can be reused as the Boozer phi grid because both 
    # uses the same uniformly spaced endpoint grids.
    kap_p_content = jnp.interp(phi_grids, varphi, curvature, period = 2*jnp.pi/nfp)[None, :]
    kap_p = ChiPhiFunc(kap_p_content, nfp)
    # Note: Rodriguez's paper uses an opposite sign for tau compared to Landreman's.
    tau_p_content = -jnp.interp(phi_grids, varphi, torsion, period = 2*jnp.pi/nfp)[None, :]
    tau_p = ChiPhiFunc(tau_p_content, nfp)

    # Storing axis info. All quantities are identically defined to pyQSC.
    axis_info = {}
    axis_info['dl_p'] = dl_p # Checked
    axis_info['kap_p'] = kap_p # Done
    axis_info['tau_p'] = tau_p # Done
    axis_info['varphi'] = varphi # Checked
    axis_info['phi'] = phi_grids # Checked
    axis_info['d_phi'] = d_phi # Grid spacing. Checked.
    axis_info['R0'] = R0[:, 0] # Checked
    axis_info['Z0'] = Z0[:, 0] # Checked
    axis_info['R0p'] = R0p[:, 0] # Checked
    axis_info['Z0p'] = Z0p[:, 0] # Checked
    axis_info['R0pp'] = R0pp[:, 0] # Checked
    axis_info['Z0pp'] = Z0pp[:, 0] # Checked
    axis_info['R0ppp'] = R0ppp[:, 0] # Checked
    axis_info['Z0ppp'] = Z0ppp[:, 0] # Checked
    # Note to self: cartesian. (dl_p = dl/dphi (Boozer) is important in Eduardo's forumlation.)
    axis_info['d_l_d_phi'] = d_l_d_phi[:, 0] # Checked
    axis_info['axis_length'] = axis_length # Checked
    axis_info['curvature'] = curvature # Checked
    axis_info['torsion'] = torsion # Checked
    axis_info['tangent_cylindrical'] = tangent_cylindrical # axis=1 is R, phi, Z, Checked
    axis_info['normal_cylindrical'] = normal_cylindrical # axis=1 is R, phi, Z, Checked
    axis_info['binormal_cylindrical'] = binormal_cylindrical # axis=1 is R, phi, Z, Checked
    return(axis_info)

def leading_orders_legacy(
    nfp, # Field period
    Rc, Rs, Zc, Zs, # Axis shape
    p0, # On-axis pressure
    Delta_0_avg, # Average anisotropy on axis
    iota_0, # On-axis rotational transform
    B_theta_20_avg, # Average B_theta[2,0]
    B_alpha_1,  # B_alpha
    B0, B11c, 
    B22c, B20, B22s, # Magnetic field strength
    len_phi,
    static_max_freq,
    traced_max_freq):

    axis_info = get_axis_info(Rc, Rs, Zc, Zs, nfp, len_phi)
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
    p0 = p0*np.ones(len_phi)
    p0 = ChiPhiFunc(p0[None, :], nfp=nfp)
    Delta0 = (-B0*p0) - phi_avg(-B0*p0) + Delta_0_avg # (Rodriguez 2021, eq. 41)
    eta = -B11c/(2*B0) # Defined for simple notation. (Rodriguez 2021, eq. 14)

    ''' 1st order quantities '''
    iota_coef = ChiPhiEpsFunc([iota_0], nfp)
    B_denom_coef_c = ChiPhiEpsFunc([B0, B1, B2], nfp)
    B_alpha_coef = ChiPhiEpsFunc([B_alpha0, B_alpha_1], nfp)
    B_theta_coef_cp = ChiPhiEpsFunc([ChiPhiFuncSpecial(0), ChiPhiFuncSpecial(0)], nfp)
    B_psi_coef_cp = ChiPhiEpsFunc([], nfp)
    Delta_coef_cp = ChiPhiEpsFunc([Delta0], nfp)
    p_perp_coef_cp = ChiPhiEpsFunc([p0], nfp)
    Y_coef_cp = ChiPhiEpsFunc([ChiPhiFuncSpecial(0)], nfp)
    Z_coef_cp = ChiPhiEpsFunc([ChiPhiFuncSpecial(0), ChiPhiFuncSpecial(0)], nfp)
    # X1 (Rodriguez 2021, eq. 14)
    X11c = eta/kap_p
    X1 = ChiPhiFunc(jnp.array([
        jnp.zeros_like(X11c.content[0]), # sin coeff is zero
        X11c.content[0],
    ]), nfp, trig_mode = True).filter(traced_max_freq[0])
    X_coef_cp = ChiPhiEpsFunc([ChiPhiFuncSpecial(0), X1], nfp)
    # p_1 and Delta1 has the same formula as higher orders.
    p_1 = iterate_p_perp_n(1,
        B_theta_coef_cp,
        B_psi_coef_cp,
        B_alpha_coef,
        B_denom_coef_c,
        p_perp_coef_cp,
        Delta_coef_cp,
        iota_coef).filter(traced_max_freq[0])
    p_perp_coef_cp = p_perp_coef_cp.append(p_1)
    Delta_1 = iterate_delta_n_0_offset(1,
        B_denom_coef_c,
        p_perp_coef_cp,
        Delta_coef_cp,
        iota_coef,
        static_max_freq=None).filter(traced_max_freq[0])
    Delta_coef_cp = Delta_coef_cp.append(Delta_1)

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
    II_2_inhomog = -B_alpha_coef[0]/2*(
        4*B0*B1*p_perp_coef_cp[1].dchi()
        -Delta_coef_cp[1]*B1.dchi()
    )[0]
    # Coefficients of B_theta
    coef_B_theta_20 = -B0**2*diff(p_perp_coef_cp[0],False,1)
    coef_dp_B_theta_20 = B0*(Delta_coef_cp[0]-1)
    # Solving y'+py=f for B_theta[2,0]. This equation has no unique solution,
    # and an initial condition is provided.
    p_eff = (coef_B_theta_20.content/coef_dp_B_theta_20.content)[0]
    f_eff = (II_2_inhomog.content/coef_dp_B_theta_20.content)[0]
    p_fft = fft_filter(jnp.fft.fft(p_eff), short_length, axis=0)
    f_fft = fft_filter(jnp.fft.fft(f_eff), short_length, axis=0)
    # Creating differential operator and convolution operator
    # as in solve_ODE
    diff_matrix = fft_dphi_op(short_length)*nfp
    conv_matrix = fft_conv_op(p_fft)
    tot_matrix = diff_matrix + conv_matrix

    # # Add a row to the matrix for avg initial condition
    # # and solve as overdetermined system with SVD. Doesn't
    # # work well in practice.
    # svd_norm = jnp.average(jnp.abs(f_fft))
    # tot_matrix_svd = jnp.zeros((short_length+1, short_length))
    # tot_matrix_svd = tot_matrix_svd.at[:-1, :].set(tot_matrix)
    # tot_matrix_svd = tot_matrix_svd.at[-1, 0].set(svd_norm)

    # f_fft_svd = jnp.zeros(short_length+1)
    # f_fft_svd = f_fft_svd.at[:-1].set(f_fft)
    # f_fft_svd = f_fft_svd.at[-1].set(B_theta_20_avg*short_length*svd_norm)
    # sln_svd = linear_least_sq_2d_svd(tot_matrix_svd, f_fft_svd)

    # Original: The average of B_theta[2,0] is its zeroth 
    # element in FFT representation.
    # The zeroth column of B_theta[2,0] acts on this element.
    # By adding 1 to all elements in this column will result in
    # adding B_theta_20_average to all elements in the RHS.
    tot_matrix_normalization = jnp.max(jnp.abs(tot_matrix))
    tot_matrix = tot_matrix.at[:, 0].set(
        tot_matrix[:, 0]+tot_matrix_normalization # was +1
    )
    f_fft = f_fft+B_theta_20_avg*short_length*tot_matrix_normalization
    sln_fft = jnp.linalg.solve(tot_matrix, f_fft)

    B_theta_20 = ChiPhiFunc(jnp.fft.ifft(fft_pad(sln_fft, len_phi, axis=0), axis=0)[None, :], nfp)
    B_theta_coef_cp = B_theta_coef_cp.append(B_theta_20)

    ''' D3 m = 0 '''
    Y11s = 2*jnp.sqrt(B0)/eta*kap_p
    # D3 can be written as y' = q0 + q1y + q2y^2
    q0 = -iota_0*(
        2*jnp.sqrt(B0)/eta*kap_p
        +eta**3/(2*jnp.sqrt(B0)*kap_p**3)
    )+dl_p*(2*tau_p+B_theta_20)*eta/kap_p
    q1 = kap_p.dphi()/kap_p
    q2 = -iota_0*eta/(2*jnp.sqrt(B0)*kap_p)
    # This equation is equivalent to the 2nd order linear ODE:
    # u''-R(x)u'+S(x)u=0, where y =
    S_lin = q0*q2
    R_lin = q1+q2.dphi()/q2
    u_avg = 1 # Doesn't actually impact Y!
    # The differential operator is:
    R_fft = fft_filter(jnp.fft.fft(R_lin.content[0]), short_length, axis=0)
    S_fft = fft_filter(jnp.fft.fft(S_lin.content[0]), short_length, axis=0)
    R_conv_matrix = fft_conv_op(R_fft)
    S_conv_matrix = fft_conv_op(S_fft)
    riccati_matrix = diff_matrix**2 - R_conv_matrix@diff_matrix + S_conv_matrix
    # old BC
    riccati_normalization = jnp.max(jnp.abs(riccati_matrix))
    riccati_matrix = riccati_matrix.at[:, 0].set(
        riccati_matrix[:, 0]+riccati_normalization # was +1
    )
    riccati_RHS = jnp.ones(short_length)*u_avg*short_length*riccati_normalization
    riccati_sln_fft = jnp.linalg.solve(riccati_matrix, riccati_RHS)
    # # Add a row to the matrix for avg initial condition
    # # and solve as overdetermined system with SVD. Doessn't
    # # work well in practice.
    # riccati_matrix_svd = jnp.zeros((riccati_matrix.shape[0]+1, riccati_matrix.shape[1]))
    # riccati_matrix_svd = riccati_matrix_svd.at[:-1, :].set(riccati_matrix)
    # riccati_matrix_svd = riccati_matrix_svd.at[-1, 0].set(1)
    # riccati_RHS_svd = jnp.zeros(short_length+1)
    # riccati_RHS_svd = riccati_RHS_svd.at[-1].set(u_avg*short_length)
    # Solution
    # riccati_sln_svd = linear_least_sq_2d_svd(riccati_matrix_svd, riccati_RHS_svd)
    riccati_u = ChiPhiFunc(jnp.fft.ifft(fft_pad(riccati_sln_fft, len_phi, axis=0), axis=0)[None, :], nfp)
    Y11c = (-riccati_u.dphi()/(q2*riccati_u))
    Y1 = ChiPhiFunc(jnp.array([
        Y11s.content[0], # sin coeff is zero
        Y11c.content[0],
    ]), nfp, trig_mode = True).filter(traced_max_freq[0])
    Y_coef_cp = Y_coef_cp.append(Y1)

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
        p_perp_coef_cp=p_perp_coef_cp.mask(2), # no pressure or delta
        Delta_coef_cp=Delta_coef_cp.mask(2),
        axis_info=axis_info,
        magnetic_only=False
    )

    return(equilibrium_out)

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
    leading_orders_magnetic_from_axis(
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
    )

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
    iota_coef = ChiPhiEpsFunc([iota_0], nfp)
    B_denom_coef_c = ChiPhiEpsFunc([B0, B1, B2], nfp)
    B_alpha_coef = ChiPhiEpsFunc([B_alpha0, B_alpha_1], nfp)
    B_theta_coef_cp = ChiPhiEpsFunc([ChiPhiFuncSpecial(0), ChiPhiFuncSpecial(0)], nfp)
    B_psi_coef_cp = ChiPhiEpsFunc([], nfp)
    Y_coef_cp = ChiPhiEpsFunc([ChiPhiFuncSpecial(0)], nfp)
    Z_coef_cp = ChiPhiEpsFunc([ChiPhiFuncSpecial(0), ChiPhiFuncSpecial(0)], nfp)
    # X1 (Rodriguez 2021, eq. 14)
    X11c = eta/kap_p
    X1 = ChiPhiFunc(jnp.array([
        jnp.zeros_like(X11c.content[0]), # sin coeff is zero
        X11c.content[0],
    ]), nfp, trig_mode = True).filter(traced_max_freq[0])
    X_coef_cp = ChiPhiEpsFunc([ChiPhiFuncSpecial(0), X1], nfp)
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
    iota_0=None,
    B_theta_20_avg=None,
    len_phi=1000,
    static_max_freq=(50, 50),
    traced_max_freq=(50, 50),
    riccati_secant_n_iter=(50, 25)):
    axis_info = get_axis_info(Rc, Rs, Zc, Zs, nfp, len_phi)
    return(
        leading_orders_from_axis(
            nfp=nfp, # Field period
            axis_info=axis_info, # Axis shape
            p0=p0, # On-axis pressure
            Delta_0_avg=Delta_0_avg, # Average anisotropy on axis
            B_alpha_1=B_alpha_1,  # B_alpha
            B0=B0, B11c=B11c, 
            B22c=B22c, B20=B20, B22s=B22s, # Magnetic field strength
            iota_0=iota_0,
            B_theta_20_avg=B_theta_20_avg,
            len_phi=len_phi,
            static_max_freq=static_max_freq,
            traced_max_freq=traced_max_freq,
            riccati_secant_n_iter=riccati_secant_n_iter
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
    iota_0=None,
    B_theta_20_avg=None,
    len_phi=1000,
    static_max_freq=(50, 50),
    traced_max_freq=(50, 50),
    riccati_secant_n_iter=(25, 25)):
    if len_phi%2!=0:
        raise ValueError('Total grid number must be a even number.')
    if not ((iota_0 is None) ^ (B_theta_20_avg is None)):
        raise ValueError('Please provide one and only one of iota_0 and average B_theta_20.')
    solve_B_theta_20_avg = iota_0 is not None
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
    p0 = p0*np.ones(len_phi)
    p0 = ChiPhiFunc(p0[None, :], nfp=nfp)
    Delta0 = (-B0*p0) - phi_avg(-B0*p0) + Delta_0_avg # (Rodriguez 2021, eq. 41)
    eta = -B11c/(2*B0) # Defined for simple notation. (Rodriguez 2021, eq. 14)

    ''' 1st order quantities '''
    # Either iota or B_theta_20_avg can be the free scalar parameter.
    # - iota first appears in Delta1.
    # - B_theta_20_avg first appears in the B_theta ODE.
    # So both Delta1 and the B_theta ODE need to be packaged 
    # for Secant iteration.
    B_denom_coef_c = ChiPhiEpsFunc([B0, B1, B2], nfp)
    B_alpha_coef = ChiPhiEpsFunc([B_alpha0, B_alpha_1], nfp)
    B_theta_coef_cp = ChiPhiEpsFunc([ChiPhiFuncSpecial(0), ChiPhiFuncSpecial(0)], nfp)
    B_psi_coef_cp = ChiPhiEpsFunc([], nfp)
    Delta_coef_cp = ChiPhiEpsFunc([Delta0], nfp)
    p_perp_coef_cp = ChiPhiEpsFunc([p0], nfp)
    Y_coef_cp = ChiPhiEpsFunc([ChiPhiFuncSpecial(0)], nfp)
    Z_coef_cp = ChiPhiEpsFunc([ChiPhiFuncSpecial(0), ChiPhiFuncSpecial(0)], nfp)
    # X1 (Rodriguez 2021, eq. 14)
    X11c = eta/kap_p
    X1 = ChiPhiFunc(jnp.array([
        jnp.zeros_like(X11c.content[0]), # sin coeff is zero
        X11c.content[0],
    ]), nfp, trig_mode = True).filter(traced_max_freq[0])
    X_coef_cp = ChiPhiEpsFunc([ChiPhiFuncSpecial(0), X1], nfp)
    # p1 and Delta1 has the same formula as higher orders.
    # p1 is not dependent on iota0. (higher order has iota 
    # dependence, though)
    print('iterate_Yn_cp_magnetic', iterate_Yn_cp_magnetic)
    print('iterate_p_perp_n', iterate_p_perp_n)
    p1 = iterate_p_perp_n(
        1,
        B_theta_coef_cp,
        B_psi_coef_cp,
        B_alpha_coef,
        B_denom_coef_c,
        p_perp_coef_cp,
        Delta_coef_cp,
        ChiPhiEpsFunc([0], nfp) # No iota dependence at leading order.
    ).filter(traced_max_freq[0])
    p_perp_coef_cp = p_perp_coef_cp.append(p1)

    def iterate_delta_n_0_offset(n_eval,
        B_denom_coef_c,
        p_perp_coef_cp,
        Delta_coef_cp,
        iota_coef,
        static_max_freq=-1,
        no_iota_masking = False): # nfp-dependent!!

        # At even orders, the free parameter is Delta_offset (the average of Delta n0)
        if n_eval%2==0:
            Delta_n_inhomog_component = eval_inhomogenous_Delta_n_cp(n_eval,
            B_denom_coef_c,
            p_perp_coef_cp,
            Delta_coef_cp.mask(n_eval-1).zero_append(),
            iota_coef)
        # At odd orders, the free parameter is iota (n-1)/2. Note the masking on iota_coef.
        # no_iota_masking is for debugging
        else:
            if no_iota_masking:
                Delta_n_inhomog_component = eval_inhomogenous_Delta_n_cp(n_eval,
                B_denom_coef_c,
                p_perp_coef_cp,
                Delta_coef_cp.mask(n_eval-1).zero_append(),
                iota_coef)
            else:
                Delta_n_inhomog_component = eval_inhomogenous_Delta_n_cp(n_eval,
                B_denom_coef_c,
                p_perp_coef_cp,
                Delta_coef_cp.mask(n_eval-1).zero_append(),
                iota_coef.mask((n_eval-3)//2).zero_append())

        # At even orders, setting Delta[even, 0] to have zero average.
        # This average is a free parameter, because the center component of
        # the ODE is dphi x = f.
        # print('Delta inhomog =============')
        # Delta_n_inhomog_component.display_content()
        content = solve_dphi_iota_dchi(
            iota=iota_coef[0]/Delta_n_inhomog_component.nfp,
            f=Delta_n_inhomog_component.content/Delta_n_inhomog_component.nfp,
            static_max_freq=static_max_freq
        )
        Delta_out = ChiPhiFunc(content, Delta_n_inhomog_component.nfp).cap_m(n_eval)
        if n_eval%2==0:
            Delta_out -= jnp.average(Delta_out[0].content)
        return(Delta_out)

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
    short_length = static_max_freq[0]*2
    def Delta1_from_iota(iota_temp):
        # First appearance of iota0
        # Delta1 is dependent on iota0.
        # Sinde Delta comes from an ODE solve, it's 
        # linearly dependent on iota but the dependence 
        # doesn't have a easily obtainable symbolic formula.
        iota_coef = ChiPhiEpsFunc([iota_temp], nfp)
        Delta1 = iterate_delta_n_0_offset(
            1,
            B_denom_coef_c,
            p_perp_coef_cp,
            Delta_coef_cp,
            iota_coef,
            static_max_freq=None,
            no_iota_masking=True
        ).filter(traced_max_freq[0])
        
        # print('Delta1 =============')
        # Delta1.display_content()
        return(Delta1)

    ''' II m = 0 '''
    def calculate_B_theta_20(Delta_coef_cp_temp, B_theta_20_avg):
        short_length = static_max_freq[0]*2
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
    # Sttempting to solve the problem with RK4
    # First, y' = f(i, y) = q0 + q1y + q2y^2.
    # For improved speed, we use phi = (0, 2dphi, 4dphi, ...)
    # as RK4 grids, and phi = (dphi, 3dphi, ...) as half-grids.
    # Under this condition, h=2pi/len_phi/nfp*2
    def Y11c_RK4(q0, q1, q2, y0):
        q0_real = jnp.real(jnp.pad(q0.content[0], [0, 2], mode='wrap'))
        q1_real = jnp.real(jnp.pad(q1.content[0], [0, 2], mode='wrap'))
        q2_real = jnp.real(jnp.pad(q2.content[0], [0, 2], mode='wrap'))
        def f_RK4(ind:int, y):
            return(q0_real[ind] + q1_real[ind]*y + q2_real[ind]*y**2)

        h = 2*jnp.pi/nfp/len_phi*2
        def iterate_y(carry, t):
            y_n, ind = carry
            # i+1 correspomds to phi+h/2, the half-grid value
            k1 = f_RK4(ind, y_n)
            k2 = f_RK4(ind+1, y_n+h*k1/2) 
            k3 = f_RK4(ind+1, y_n+h*k2/2)
            k4 = f_RK4(ind+2, y_n+h*k3)
            y_np1 = y_n + h/6*(k1+2*k2+2*k3+k4)
            return((y_np1, ind+2), y_np1)

        ind_list = jnp.arange(0,len_phi+2,2)
        (_, _), y_arr = jax.lax.scan(f=iterate_y, init=(y0, 0), xs=ind_list)
        # This is an objective function that enforces
        # the boundary condition.
        # For some iota and y(0) the solution is exponentially growing
        # and can grow to infinity. This takes care of it numerically.
        # Returns the array's max abs if array's last item is not
        # a finite number. 
        diff = jnp.abs(y_arr[0]-y_arr[-1])
        # Replaces all inf and nan's with nan
        y_fin = jnp.where(jnp.isfinite(y_arr), y_arr, jnp.nan)
        y_max = jnp.nanmax(y_fin)
        y_min = jnp.nanmin(y_fin)
        # ymax-ymin >= |y(2pi)-y(0)| >= 0
        return(jnp.nanmin(jnp.array([
            # When y_arr contains no nan, 
            # gives error or deviation from periodic BC
            # Otherwise will be nan
            jnp.log10(diff+1),
            # diff,
            jnp.log10(y_max - y_min + 1)
        ])), y_arr)
    def Y11c_spectral(q0, q1, q2, y0):
        # This equation is equivalent to the 2nd order linear ODE:
        # u''-R(x)u'+S(x)u=0.
        S_lin = q0*q2
        R_lin = q1+q2.dphi()/q2
        R_fft = fft_filter(jnp.fft.fft(R_lin.content[0]), static_max_freq[0], axis=0)
        S_fft = fft_filter(jnp.fft.fft(S_lin.content[0]), static_max_freq[0], axis=0)
        R_conv_matrix = fft_conv_op(R_fft)
        S_conv_matrix = fft_conv_op(S_fft)
        diff_matrix = fft_dphi_op(static_max_freq[0])*nfp
        riccati_matrix = diff_matrix**2 - R_conv_matrix@diff_matrix + S_conv_matrix
        # Used to modify the singular matrix
        riccati_normalization = jnp.max(jnp.abs(riccati_matrix))
        # # This is a homogenous ODE, and riccati_matrix is rank short_length-1.
        # # To solve this problem, we prescribe an arbitrary average u. It seems that this yields
        # # a solution with u(0)=u(2pi) but discontinuous derivatives at phi=0.
        u_avg=1
        riccati_normalization = jnp.max(jnp.abs(riccati_matrix))
        riccati_matrix = riccati_matrix.at[:, 0].set(
            riccati_matrix[:, 0]+riccati_normalization # was +1
        )
        riccati_RHS = jnp.ones(static_max_freq[0])*u_avg*static_max_freq[0]*riccati_normalization
        riccati_sln_fft = jnp.linalg.solve(riccati_matrix, riccati_RHS)
        riccati_u = jnp.fft.ifft(fft_pad(riccati_sln_fft, len_phi, axis=0), axis=0)
        riccati_u_chiphifunc = ChiPhiFunc(riccati_u[None, :], nfp)
        Y11c_temp = -(riccati_u_chiphifunc.dphi()/(q2*riccati_u_chiphifunc))
        # 4th order BC matching
        delta_phi = np.pi*2/nfp/len_phi
        return(
            jnp.max(jnp.abs((q0+q1*Y11c_temp+q2*Y11c_temp**2 - Y11c_temp.dphi()).content)), 
            Y11c_temp.content[0]
        )
    # Defining the function to be evaluated in the 
    # Secant method loop, based on whether iota or B_theta is unknown.
    # Putting the if statement here saves some JIT time during loop unpacking.
    if solve_B_theta_20_avg:
        # When iota is given, Delta is constant 
        # during the iteration.
        Delta1 = Delta1_from_iota(iota_temp=iota_0)
        # @jit
        def secant_step(x_secant, y_secant, obj_callable):
            # First calculate B_theta 
            # from yet unknown average
            B_theta_20 = calculate_B_theta_20(
                Delta_coef_cp_temp=Delta_coef_cp.append(Delta1), 
                B_theta_20_avg=x_secant,
            )
            q0, q1, q2 = get_riccati_coeffs(iota_0=iota_0, B_theta_20=B_theta_20)
            f_secant, Y11c_guess = obj_callable(q0, q1, q2, y_secant)
            return(f_secant, Y11c_guess, Delta1, B_theta_20)
    else:
        # When iota is unknown, Delta changes during 
        # the iteration.
        # @jit
        def secant_step(x_secant, y_secant, obj_callable):
            # First calculate delta from yet unknown iota.
            Delta1 = Delta1_from_iota(iota_temp=x_secant)
            # Then calculate B_theta 
            # from a given average
            B_theta_20 = calculate_B_theta_20(
                Delta_coef_cp_temp=Delta_coef_cp.append(Delta1), 
                B_theta_20_avg=B_theta_20_avg,
            )
            # B_theta_20.display_content()
            q0, q1, q2 = get_riccati_coeffs(iota_0=x_secant, B_theta_20=B_theta_20)
            f_secant, Y11c_guess = obj_callable(q0, q1, q2, y_secant)
            return(f_secant, Y11c_guess, Delta1, B_theta_20)
    def secant_scan_callable(carry, t, obj_callable):
        # f_nm1: (Y11c'(0) from BC - Y11c'(0) from ODE)**2
        # Y11c: Y11c
        (
            x_secant_init2, 
            y_secant_init2,
            x_secant_init1, 
            y_secant_init1,
            f_nm2
        ) = carry
        f_nm1, _, _, _ = secant_step(
            x_secant_init1, 
            y_secant_init1,
            obj_callable
        )
        # Iterate x
        x_secant = (
            x_secant_init2*f_nm1
            - x_secant_init1*f_nm2
        )/(f_nm1 - f_nm2)
        # Iterate Y11c(0)
        y_secant = (
            y_secant_init2*f_nm1
            - y_secant_init1*f_nm2
        )/(f_nm1 - f_nm2)
        return(
            (
                x_secant_init1,
                y_secant_init1,
                x_secant,
                y_secant,
                f_nm1,
            ),
            # Tracks iteration progress. Ideally
            # Y11c_RK, Delta1 and B_theta_20 
            # should be in carry, but when the iteration 
            # reaches 0 prematurely f, Y, Delta1 and B_theta_20
            # will all become nan. This allows us to pick the 
            # last group of non-nan results.
            (x_secant, y_secant, f_nm1) 
        )
    secant_scan_callable1 = lambda carry, t: secant_scan_callable(
        carry, t, Y11c_RK4)
    secant_scan_callable2 = lambda carry, t: secant_scan_callable(
        carry, t, Y11c_spectral)

    # Secant iteration.
    if solve_B_theta_20_avg:
        x_secant_init1=1.1*B0
        x_secant_init2=B0
        y_secant_init1=-1.0
        y_secant_init2=1.0
    else:   
        x_secant_init1=0.0
        x_secant_init2=0.17
        y_secant_init1=-1.0
        y_secant_init2=1.0
    f_init, _, _, _ = secant_step(
        x_secant_init2, y_secant_init2, Y11c_RK4)
    # Performing the first iteration
    # to find iota_0/B_theta_20_avg with finite (instead
    # of exponentially growing) and near periodic solutions.
    _, (x_secant_list1, _, f_list1) = jax.lax.scan(
        f=secant_scan_callable1, 
        init=(
            x_secant_init2, 
            y_secant_init2,
            x_secant_init1, 
            y_secant_init1,
            f_init,
        ), 
        xs=jnp.arange(50)# riccati_secant_n_iter[0])
    )
    x_secant_sln1 = x_secant_list1[jnp.nanargmin(f_list1)-1]
    # Performing the second iteration to find the 
    # correct iota_0/B_theta_20_avg
    f2_init, _, _, _ = secant_step(
        x_secant_sln1, 0, Y11c_spectral)
    _, (x_secant_list2, _, f_list2) = jax.lax.scan(
        f=secant_scan_callable2, 
        init=(
            x_secant_sln1, 
            0.0, # Spectral solve does not require Y11c(0)
            x_secant_sln1*1.01, 
            0.0, # Spectral solve does not require Y11c(0)
            f2_init,
        ), 
        xs=jnp.arange(riccati_secant_n_iter[1])
    )
    x_secant_sln2 = x_secant_list2[jnp.nanargmin(f_list2)-1]
    # Calculating the value of Y, Delta and B_theta
    f_fin, Y11c_arr, Delta1, B_theta_20 = secant_step(
        x_secant_sln2, 0, Y11c_spectral)
    # Storing these values
    Y11s = 2*jnp.sqrt(B0)/eta*kap_p
    Y1 = ChiPhiFunc(jnp.array([
        Y11s.content[0], # sin coeff is zero
        Y11c_arr,
    ]), nfp, trig_mode = True).filter(traced_max_freq[0])
    Y_coef_cp = Y_coef_cp.append(Y1)
    Delta_coef_cp = Delta_coef_cp.append(Delta1)
    B_theta_coef_cp = B_theta_coef_cp.append(B_theta_20)


    # Giving value to iota
    if solve_B_theta_20_avg:
        iota_coef = ChiPhiEpsFunc([iota_0], nfp)
    else:
        iota_coef = ChiPhiEpsFunc([x_secant_list2[jnp.argmin(f_list2)-1]], nfp)
    if solve_B_theta_20_avg:
        print('Solving for average B_{\\theta20} self-consistently:')
        print('\\bar{B}_{\\theta20} =', x_secant_list2[jnp.argmin(f_list2)-1])
    else:
        print('Solving for \\bar{\\iota}_0 self-consistently:')
        print('\\bar{\\iota}_0 =', x_secant_list2[jnp.argmin(f_list2)-1])

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
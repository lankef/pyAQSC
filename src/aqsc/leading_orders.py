import jax.numpy as jnp
# from jax import jit

from .looped_solver import *
from .chiphifunc import *
from .chiphiepsfunc import *
from .equilibrium import *

# Generate the circular axis case in Rodriguez Bhattacharjee
def circular_axis():
    Rc, Rs = ([1, 0, 0.0001], [0, 0, 0])
    Zc, Zs = ([0, 0, 0], [0, 0, 0.001])
    phis = jnp.linspace(0,2*jnp.pi*0.999,1000)
    return(leading_orders(
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
    axis_info['varphi'] = varphi # Done
    axis_info['phi'] = phi_grids # Done
    axis_info['d_phi'] = d_phi # Done
    axis_info['R0'] = R0[:, 0] # Done
    axis_info['Z0'] = Z0[:, 0] # Done
    axis_info['R0p'] = R0p[:, 0] # Done
    axis_info['Z0p'] = Z0p[:, 0] # Done
    axis_info['R0pp'] = R0pp[:, 0] # Done
    axis_info['Z0pp'] = Z0pp[:, 0] # Done
    axis_info['R0ppp'] = R0ppp[:, 0] # Done
    axis_info['Z0ppp'] = Z0ppp[:, 0] # Done
    # Note to self: cartesian. (dl_p = dl/dphi (Boozer) is important in Eduardo's forumlation.)
    axis_info['d_l_d_phi'] = d_l_d_phi[:, 0] # Done.
    axis_info['axis_length'] = axis_length # Done
    axis_info['curvature'] = curvature # Done
    axis_info['torsion'] = torsion # Done
    axis_info['tangent_cylindrical'] = tangent_cylindrical # axis=1 is R, phi, Z
    axis_info['normal_cylindrical'] = normal_cylindrical # axis=1 is R, phi, Z
    axis_info['binormal_cylindrical'] = binormal_cylindrical # axis=1 is R, phi, Z
    return(dl_p, kap_p, tau_p, axis_info)

def leading_orders(
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
    traced_max_freq,
):
    dl_p, kap_p, tau_p, axis_info = get_axis_info(Rc, Rs, Zc, Zs, nfp, len_phi)
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
    # p1 and Delta1 has the same formula as higher orders.
    p1 = iterate_p_perp_n(1,
        B_theta_coef_cp,
        B_psi_coef_cp,
        B_alpha_coef,
        B_denom_coef_c,
        p_perp_coef_cp,
        Delta_coef_cp,
        iota_coef).filter(traced_max_freq[0])
    p_perp_coef_cp = p_perp_coef_cp.append(p1)
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
    shortened_length = static_max_freq[0]*2
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
    p_fft = fft_filter(jnp.fft.fft(p_eff), shortened_length, axis=0)
    f_fft = fft_filter(jnp.fft.fft(f_eff), shortened_length, axis=0)
    # Creating differential operator and convolution operator
    # as in solve_ODE
    diff_matrix = fft_dphi_op(shortened_length)
    conv_matrix = fft_conv_op(p_fft)
    tot_matrix = diff_matrix + conv_matrix

    # # Add a row to the matrix for avg initial condition
    # # and solve as overdetermined system with SVD. Doesn't
    # # work well in practice.
    # svd_norm = jnp.average(jnp.abs(f_fft))
    # tot_matrix_svd = jnp.zeros((shortened_length+1, shortened_length))
    # tot_matrix_svd = tot_matrix_svd.at[:-1, :].set(tot_matrix)
    # tot_matrix_svd = tot_matrix_svd.at[-1, 0].set(svd_norm)

    # f_fft_svd = jnp.zeros(shortened_length+1)
    # f_fft_svd = f_fft_svd.at[:-1].set(f_fft)
    # f_fft_svd = f_fft_svd.at[-1].set(B_theta_20_avg*shortened_length*svd_norm)
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
    f_fft = f_fft+B_theta_20_avg*shortened_length*tot_matrix_normalization
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
    R_fft = fft_filter(jnp.fft.fft(R_lin.content[0]), shortened_length, axis=0)
    S_fft = fft_filter(jnp.fft.fft(S_lin.content[0]), shortened_length, axis=0)
    R_conv_matrix = fft_conv_op(R_fft)
    S_conv_matrix = fft_conv_op(S_fft)
    riccati_matrix = diff_matrix**2 - R_conv_matrix@diff_matrix + S_conv_matrix
    # old BC
    riccati_normalization = jnp.max(jnp.abs(riccati_matrix))
    riccati_matrix = riccati_matrix.at[:, 0].set(
        riccati_matrix[:, 0]+riccati_normalization # was +1
    )
    riccati_RHS = jnp.ones(shortened_length)*u_avg*shortened_length*riccati_normalization
    riccati_sln_fft = jnp.linalg.solve(riccati_matrix, riccati_RHS)
    # # Add a row to the matrix for avg initial condition
    # # and solve as overdetermined system with SVD. Doessn't
    # # work well in practice.
    # riccati_matrix_svd = jnp.zeros((riccati_matrix.shape[0]+1, riccati_matrix.shape[1]))
    # riccati_matrix_svd = riccati_matrix_svd.at[:-1, :].set(riccati_matrix)
    # riccati_matrix_svd = riccati_matrix_svd.at[-1, 0].set(1)
    # riccati_RHS_svd = jnp.zeros(shortened_length+1)
    # riccati_RHS_svd = riccati_RHS_svd.at[-1].set(u_avg*shortened_length)
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
        target_len_phi=1000,
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
    traced_max_freq,
):
    dl_p, kap_p, tau_p, axis_info = get_axis_info(Rc, Rs, Zc, Zs, nfp, len_phi)
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
    # p1 and Delta1 has the same formula as higher orders.

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
    shortened_length = static_max_freq[0]*2
    # RHS of II[1][0]
    # Creating differential operator and convolution operator
    # as in solve_ODE
    diff_matrix = fft_dphi_op(shortened_length)
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
    R_fft = fft_filter(jnp.fft.fft(R_lin.content[0]), shortened_length, axis=0)
    S_fft = fft_filter(jnp.fft.fft(S_lin.content[0]), shortened_length, axis=0)
    R_conv_matrix = fft_conv_op(R_fft)
    S_conv_matrix = fft_conv_op(S_fft)
    riccati_matrix = diff_matrix**2 - R_conv_matrix@diff_matrix + S_conv_matrix
    # BC
    riccati_matrix = riccati_matrix.at[:, 0].set(riccati_matrix[:, 0]+1)
    riccati_RHS = jnp.ones(shortened_length)*u_avg*shortened_length
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

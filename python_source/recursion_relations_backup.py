# Wrapped/completed recursion relations based on translated expressions
# in parsed/. Necessary masking and/or n-substitution are included. All iterate_*
# methods returns ChiPhiFuncGrid's.

# ChiPhiFunc and ChiPhiEpsFunc
from chiphifunc import *
from chiphiepsfunc import *
from math_utilities import *
import numpy as np

# parsed relations
import parsed

# Performance
import time

''' I. Magnetic equations '''
# The magnetic equations alone can serve as a set of recursion relations,
# solving for $X, Y, Z, B_{\psi}, \iota$ from $B_{\theta}, B_{alpha}$ and $B$.
''' I.1 Recursion relations for individual variables '''
# Evaluate Xn.
# Requires:
# X_{n-1}, Y_{n-1}, Z_n,
# \iota_{(n-3)/2 or (n-2)/2},
# B_{\alpha (n-1)/2 or n/2}.
def iterate_Xn_cp(n_eval,
    X_coef_cp,
    Y_coef_cp,
    Z_coef_cp,
    B_denom_coef_c,
    B_alpha_coef,
    kap_p, dl_p, tau_p,
    iota_coef):
    return(parsed.eval_xn.eval_Xn_cp(
        n=n_eval,
        X_coef_cp=X_coef_cp.mask(n_eval-1).zero_append(),
        Y_coef_cp=Y_coef_cp.mask(n_eval-1),
        Z_coef_cp=Z_coef_cp.mask(n_eval),
        B_denom_coef_c=B_denom_coef_c,
        B_alpha_coef=B_alpha_coef,
        kap_p=kap_p, dl_p=dl_p, tau_p=tau_p,
        iota_coef=iota_coef))

# Evaluates Yn using Yn+1 general formula. The free component is either Yn0 or
# Yn1c.
# Requires:
# X_{n}, Y_{n-1}, Z_{n-1},
# B_{\theta n-1}, B_{\psi  n-3},
# \iota_{(n-3)/2 or (n-4)/2}, B_{\alpha  (n-1)/2 or (n-2)/2}
def iterate_Yn_cp(n_eval,
    X_coef_cp,
    Y_coef_cp,
    Z_coef_cp,
    B_psi_coef_cp,
    B_theta_coef_cp,
    B_alpha_coef,
    kap_p, dl_p, tau_p,
    iota_coef,
    Yn_free):
    # Getting coeffs
    # Both uses B_alpha0 and X1 only

    coef_a = parsed.eval_ynp1.coef_a(n_eval-1, B_alpha_coef, X_coef_cp)
    coef_b = parsed.eval_ynp1.coef_b(B_alpha_coef, X_coef_cp)

    # Getting rhs-lhs
    # for Yn to work, "n" must be subbed with n-1 here
    ynp1_rhsmlhs = parsed.eval_ynp1.rhs_minus_lhs(n_eval-1,
        X_coef_cp,
        Y_coef_cp.mask(n_eval-1).zero_append(),
        Z_coef_cp,
        B_psi_coef_cp,
        B_theta_coef_cp,
        B_alpha_coef,
        kap_p, dl_p, tau_p,
        iota_coef)

    # Solving (conv(a) + conv(b)@dchi)@Yn+1 = RHS - LHS(Yn+1 = 0)
    # print('v_rhs content and average order of magnitude')
    # ynp1_rhsmlhs.display_content(colormap_mode = True)
    # plt.plot(np.log10(np.mean(np.abs(ynp1_rhsmlhs.content), axis=1)))
    # plt.show()
    # print('rank_rhs', n_eval)
    return(ChiPhiFuncGrid.solve_underdet_degen(Y_mode=True,
                                        v_source_A=coef_a,
                                        v_source_B=coef_b,
                                        v_rhs=ynp1_rhsmlhs,
                                        rank_rhs=n_eval, # Yn has n+1 dof. So, RHS must have n dof.
                                        i_free=0, # (dummy)
                                        vai=Yn_free, ignore_extra=True))

# Evaluates iota_{(n-1)/2}.
# Requires:
# \Lambda_n (X_{n-1}, Y_{n-1}, Z_{n}, \iota_{(n-3)/2}),
# B_{\theta n}, B_0, B_{\alpha 0}$
def iterate_iota_nm1b2(n,
    X_coef_cp, Y_coef_cp, Z_coef_cp,\
               tau_p, dl_p, kap_p,\
               iota_coef, eta,\
               B_denom_coef_c, B_alpha_coef, B_theta_coef_cp):

    if n%2!=1:
        raise ValueError("n must be even to evaluate iota_{(n-1)/2}")

    Yn1s_p, Yn1c_p = Y_coef_cp[n].get_Yn1s_Yn1c()
    Y11s_p, Y11c_p = Y_coef_cp[1].get_Yn1s_Yn1c()
    _, X11c_p = X_coef_cp[1].get_Yn1s_Yn1c()

    sigma_p = Y11c_p/Y11s_p # Definition. On pg 4 below (6)

    # Note: mask n leaves nth-order as the last order in.
    # Xi requires Yn, Zn+1=0. This means mask(n-1) and mask(n)
    Xi_n_p = parsed.eval_full_xi_n.eval_full_Xi_n_p(
        n, X_coef_cp, Y_coef_cp.mask(n-1).zero_append(), Z_coef_cp.mask(n).zero_append(), \
        kap_p, dl_p, tau_p, iota_coef.mask((n-1)//2-1).zero_append()).get_constant()
    iota_0 = iota_coef[0]

    # Evaluates exp(2*iota_bar_0*integral_of_sigma_to_phi').
    exponent = 2*iota_0*sigma_p.integrate_phi(periodic = False)
    exp_factor = exponent.exp()
    exponent_2pi = 2*iota_0*sigma_p.integrate_phi(periodic = True)
    exp_factor_2pi = np.e**exponent_2pi

    # Defnition. On pg 4 above (8)
    sigma_n_tilde   = Yn1c_p/Y11s_p
    sigma_n_tilde_0 = sigma_n_tilde.get_phi_zero()

    # Yn1s_p MUST be evaluated with iota_nm1b2=0!
    Lambda_n_p_tilde = Lambda_n_p(Yn1s_p, Y11s_p, Y11c_p, X11c_p, iota_0, tau_p, dl_p, sigma_p, Xi_n_p)

    # B_0
    B0 = B_denom_coef_c[0]

    # B_{\alpha 0}
    B_alpha0 = B_alpha_coef[0]

    # B_{\theta n 0}. The whole term might be 0 depending on init conds (current free?)
    try:
        B_theta_n0 = B_theta_coef_cp[n].get_constant()
    except AttributeError:
        B_theta_n0 = B_theta_coef_cp[n]

    # The denominator,
    denom = (exp_factor*(1+sigma_p**2+(eta/kap_p)**4/4/B0)).integrate_phi(periodic = True)
    return(
        ((exp_factor*(Lambda_n_p_tilde+2*B_alpha0*B0*B_theta_n0/(Y11s_p**2))).integrate_phi(periodic = True)
        + sigma_n_tilde_0*(1-exp_factor_2pi))/denom
    )

# Evaluates Zn,
# Requires:
# X_{n-1}, Y_{n-1}, Z_{n-1},
# B_{\theta n}, B_{\psi n-1},
# B_{\alpha (n-2)/2 \text{ or } (n-3)/2},
# \iota_{(n-2)/2 \text{ or } (n-3)/2}
# \kappa, \frac{dl}{d\phi}, \tau
def iterate_Zn_cp(
    n_eval,
    X_coef_cp, Y_coef_cp, Z_coef_cp,
    B_theta_coef_cp, B_psi_coef_cp,
    B_alpha_coef,
    kap_p, dl_p, tau_p,
    iota_coef):
    return(parsed.eval_znp1.eval_Zn_cp(
        n=n_eval,
        X_coef_cp=X_coef_cp.mask(n_eval-1),
        Y_coef_cp=Y_coef_cp.mask(n_eval-1),
        Z_coef_cp=Z_coef_cp.mask(n_eval-1).zero_append(),
        B_theta_coef_cp=B_theta_coef_cp,
        B_psi_coef_cp=B_psi_coef_cp.mask(n_eval-1),
        B_alpha_coef=B_alpha_coef,
        kap_p=kap_p, dl_p=dl_p, tau_p=tau_p,
        iota_coef = iota_coef
    ))

# Solving for Yn1c for odd orders
def iterate_Yn1c_p(n, X_coef_cp, Y_coef_cp, Z_coef_cp,\
                  iota_coef,\
                  tau_p, dl_p, kap_p, eta,\
                  B_denom_coef_c, B_alpha_coef,
                  B_psi_coef_cp, B_theta_coef_cp):

    Yn1s_p = eval_ynp1s1.evaluate_ynp1s1_full(n-1,
    X_coef_cp,
    Y_coef_cp.mask(n_eval-1).zero_append(),
    Z_coef_cp,
    B_psi_coef_cp,
    B_theta_coef_cp,
    B_alpha_coef,
    kap_p, dl_p, tau_p, eta,
    iota_coef).get_constant()

    Y11s_p, Y11c_p = Y_coef_cp[1].get_Yn1s_Yn1c()
    _, X11c_p = X_coef_cp[1].get_Yn1s_Yn1c()

    # Note the difference with iota_nm1b2: iota(n-1)//2=0 is no longer applied here.
    Xi_n_p_no_iota_mask = eval_full_xi_n.eval_full_Xi_n_p(
        n, X_coef_cp, Y_coef_cp.mask(n-1).zero_append(), Z_coef_cp.mask(n).zero_append(), \
        kap_p, dl_p, tau_p, iota_coef).get_constant()
    Xi_n_p = eval_full_xi_n.eval_full_Xi_n_p(
        n, X_coef_cp, Y_coef_cp.mask(n-1).zero_append(), Z_coef_cp.mask(n).zero_append(), \
        kap_p, dl_p, tau_p, iota_coef.mask((n-1)//2-1).zero_append()).get_constant()

    iota_0 = iota_coef[0]
    B0 = B_denom_coef_c[0]
    B_alpha0 = B_alpha_coef[0]
    try:
        B_theta_np10 = B_theta_coef_cp[n+1].get_constant()
    except AttributeError:
        B_theta_np10 = B_theta_coef_cp[n+1]

    sigma_p = Y11c_p/Y11s_p # Definition. On pg 4 below (6)
    Lambda_n_p_eval = Lambda_n_p(Yn1s_p, Y11s_p, Y11c_p, X11c_p, iota_0, tau_p, dl_p, sigma_p, Xi_n_p_no_iota_mask)

    # Solving with integration factor
    RHS = Lambda_n_p_eval+2*B_alpha0*B0*B_theta_np10/(Y11s_p**2)
    coeff = 2*iota_0*sigma_p
    sigma_tilde_n, _ = solve_integration_factor(coeff.content, 1, RHS.content)

    return(sigma_tilde_n*Y11s_p)

# Evaluates Yn1s for odd n's.
# X_{n}, Y_{n-1}, Z_{n-1},
# B_{\theta n-1}, B_{\psi  n-3},
# \iota_{(n-3)/2 or (n-4)/2}, B_{\alpha  (n-1)/2 or (n-2)/2}


# Evaluates \Lambda_n. Used in Yn and iota_(n-1)/2
# Must be evaluated with iota_{(n-1)/2} = 0 to get lambda_tilde
def Lambda_n_p(Yn1s_p, Y11s_p, Y11c_p, X11c_p, iota_0, tau_p, dl_p, sigma_p, Xi_n_p):
    return(
        Yn1s_p.dphi()*sigma_p/Y11s_p
        -2*iota_0*Yn1s_p/Y11s_p
        -Yn1s_p*Y11c_p.dphi()/(Y11s_p**2)
        +2*tau_p*dl_p*Yn1s_p*X11c_p/(Y11s_p**2)
        -2*Xi_n_p/(Y11s_p**2)
    )

# Evaluates B_{\psi n-2}
# Requires:
# X_{n-1}, Y_{n-1}, Z_{n-1},
# B_{\theta n-1}, B_0,
# B_{\alpha 0}, \bar{\iota}_{(n-2)/2 or (n-3)/2}$
def iterate_dc_Bpsi_nm2(
    n_eval,
    X_coef_cp, Y_coef_cp, Z_coef_cp,
    B_theta_coef_cp, B_psi_coef_cp,
    B_alpha_coef, B_denom_coef_c,
    kap_p, dl_p, tau_p,
    iota_coef):
    dchi_b_psi_nm2 = parsed.eval_dchi_b_psi_nm2.eval_dchi_B_psi_cp_nm2(n_eval, \
        X_coef_cp, Y_coef_cp, Z_coef_cp.mask(n_eval-1).zero_append(), \
        B_theta_coef_cp, B_psi_coef_cp.mask(n_eval-3).zero_append(), B_alpha_coef, B_denom_coef_c, \
        kap_p, dl_p, tau_p, iota_coef)
    # The evaluation gives an extra component.
    return(dchi_b_psi_nm2.cap_m(n_eval-2))

''' II. Magnetic and MHD equations '''
''' II.1 Recursion relations for individual variables '''

# Uses B_theta [n-2], B_[psi n-2], B_alpha first-order terms, B_denom[n],
# p_[perp n-1], Delta_[n-1],iota_[(n-3)/2 (n-2)/2]
def iterate_p_perp_n(n_eval,
    B_theta_coef_cp,
    B_psi_coef_cp,
    B_alpha_coef,
    B_denom_coef_c,
    p_perp_coef_cp,
    Delta_coef_cp,
    iota_coef):
    return(
        parsed.eval_p_perp_n.eval_p_perp_n_cp(n_eval,
        B_theta_coef_cp.mask(n_eval-2).zero_append().zero_append(), # cancellation for 2 orders
        B_psi_coef_cp,
        B_alpha_coef,
        B_denom_coef_c,
        p_perp_coef_cp.mask(n_eval-1).zero_append(),
        Delta_coef_cp,
        iota_coef)
    )

# Uses B_denom[n], p_[\perp n]$, $\Delta_[n-1]$, $\iota_[(n-1)/2, n/2]$
# TODO Integration constant for zeroth mode. It's just y'=f, and the integration
# constant need to satisfy integral(Bpsi') = 0
def iterate_delta_n(n_eval,
    B_denom_coef_c,
    p_perp_coef_cp,
    Delta_coef_cp,
    iota_coef):
    Delta_n_inhomog_component = parsed.eval_delta_n.eval_inhomogenous_Delta_n_cp(n_eval,
    B_denom_coef_c,
    p_perp_coef_cp,
    Delta_coef_cp.mask(n_eval-1).zero_append(),
    iota_coef)

    content, _ = solve_dphi_iota_dchi(iota_coef[0], Delta_n_inhomog_component.content)
    return(
        ChiPhiFuncGrid(content)
    )

''' II.2 Recursion relations from the looped equation '''
# Iterates Btheta n-1 and loop_suppressing_B_theta_nm1 for even order
# Iterates Btheta n-1 ONLY for odd order.
# Uses Xn-1, Yn-1, Zn-1,  B_theta_n-1, Delta_n-1,
# B_psi_n-3, B_denom n-1,
# iota_coef (n-3)/2 or (n-2)/2, and B_alpha (n-1)/2 or (n-2)/2
# ------------------------------------------------------------------------------
# SPECIAL NOTE: for odd B_psi n-3, the zero components of B_psi and Y are unknown,
# but the non-zero components MUST be provided. It may seem problematic to solve
# for B_theta with incomplete B_psi, but B_psi_n-3,0 Y_n-1,0 only reside in the
# innermost 2 components since their coeffs only have 2 chi components.
# ------------------------------------------------------------------------------
# Must be evaluated with Z_coef_cp[n] = 0, p_perp_coef_cp[n] = 0
# B_psi_coef_cp[n-2] = 0, B_denom_coef_c[n] = 0 and B_theta_coef_cp[n] = 0
# B_theta_n0 must be provided for even n-1.
#
# Input: ChiPhiEpsFunc's, except for the coeff_B_theta's.
# The coeff_B_theta.s are ChiPhiFuncGrid's provided by an equilibrium object.
# Output:
# Even n_eval: B_theta, loop_suppressing_B_theta_nm1
# Even n_eval: B_theta ONLY.
def iterate_B_theta_nm1_cp(n_eval,
    X_coef_cp, Y_coef_cp, Z_coef_cp,
    B_theta_coef_cp, B_psi_coef_cp,
    B_alpha_coef, B_denom_coef_c,
    p_perp_coef_cp,
    Delta_coef_cp,
    kap_p, dl_p, tau_p, iota_coef,
    coeff_B_theta_nm1, # test_equilibrium.coeff_B_theta_nm1
    coeff_dc_B_theta_nm1,
    coeff_dp_B_theta_nm1,
    integral_mode, B_theta_n0=None):


    # Evaluating the RHS
    # Filtering can be introduced
    loop_suppressing_B_theta_nm1 = parsed.eval_loop(n_eval, \
    X_coef_cp, Y_coef_cp, Z_coef_cp.mask(n_eval-1).zero_append(), \
    B_theta_coef_cp.mask(n_eval-2).zero_append().zero_append(), \
    B_psi_coef_cp.mask(n_eval-3).zero_append(),\
    B_alpha_coef, B_denom_coef_c.mask(n_eval-1).zero_append(), \
    p_perp_coef_cp.mask(n_eval-1).zero_append(), \
    Delta_coef_cp.mask(n_eval-1).zero_append(), kap_p, dl_p, tau_p, iota_coef).cap_m(n_eval-2)

    len_phi = loop_suppressing_B_theta_nm1.get_shape()[1]

    # The convolution matrices acting on B_theta and other terms have a simple form:
    # Cre =
    # 1  0  0
    # 1  1  0
    # 0  1  1
    # 0  0  1 ...
    # Cim =
    #-1  0  0
    # 1 -1  0
    # 0  1 -1
    # 0  0  1 ...
    # R and I are defined s.t.
    # R[phi] * Cr = np.real(conv_matrix(coeff[:,phi], n-2))
    # I[phi] * Ci = np.imag(conv_matrix(coeff[:,phi], n-2))
    R = np.real(coeff_B_theta_nm1.content[1])
    R_dc = np.real(coeff_dc_B_theta_nm1.content[1])
    R_dp = np.real(coeff_dp_B_theta_nm1.content[1])
    I = np.imag(coeff_B_theta_nm1.content[1])
    I_dc = np.imag(coeff_dc_B_theta_nm1.content[1])
    I_dp = np.imag(coeff_dp_B_theta_nm1.content[1])


    # Solving the top and bottom elements

    # solves for B_theta from
    # test_equilibrium.coeff_B_theta_nm1*B_theta
    # + test_equilibrium.coeff_dc_B_theta_nm1*B_theta.dchi()
    # + test_equilibrium.coeff_dp_B_theta_nm1*B_theta.dphi()
    # = loop_suppressing_B_theta_nm1
    # using provided loop_suppressing_B_theta_nm1.
    # for testing only.
    # The number of components included here is n-2, rather than n
    # because B_theta's have zero m=n components.
    # Even n-1's center comp not added
    # Blank array for B_theta
    B_theta_out = np.zeros((n_eval-2, len_phi), dtype = np.complex128) # (n-1)+1-2 = n-2.

    # The top and bottom components are not coupled with
    # anything else and are solved first.
    m_prev = n_eval-3 # current mode being evaluated (mode of the outmost comps)
    A_minus_plus = np.array([(R - 1j*I) - 1j*(m_prev)*(R_dc - 1j*I_dc),
                             (R + 1j*I) + 1j*(m_prev)*(R_dc + 1j*I_dc)])
    B_minus_plus = np.array([(R_dp - 1j*I_dp),
                             (R_dp + 1j*I_dp)])
    f_minus_plus = np.array([loop_suppressing_B_theta_nm1.content[0],
                            loop_suppressing_B_theta_nm1.content[-1]])
    B_theta_bottom_top,_ = solve_integration_factor(coeff = A_minus_plus,
                             coeff_dp = B_minus_plus,
                             f = f_minus_plus, integral_mode = integral_mode)

    B_theta_out[0] = B_theta_bottom_top[0]
    B_theta_out[-1] = B_theta_bottom_top[-1]
    # Solve one by one for lower components, if there are lower components.
    # (n_eval > 4 or 5, there're more components to solve
    # for than the top and bottom 2)
    if (n_eval-2)//2>1:
        print('Solving for inner components')
        B_theta_prev = B_theta_bottom_top
        dp_B_theta_prev = ChiPhiFuncGrid(B_theta_prev).dphi().content
        B_curr = B_minus_plus
        # number of component pair besides the outmost pairs
        for i in range((n_eval-2)//2-1):


            # Coeffs of the previous order
            A_prev = np.array([(R + 1j*I) - 1j*(m_prev)*(R_dc + 1j*I_dc),
                               (R - 1j*I) + 1j*(m_prev)*(R_dc - 1j*I_dc)])
            B_prev = np.array([(R_dp + 1j*I_dp),
                               (R_dp - 1j*I_dp)])
            A_curr = np.array([(R - 1j*I) - 1j*(m_prev-2)*(R_dc - 1j*I_dc),
                               (R + 1j*I) + 1j*(m_prev-2)*(R_dc + 1j*I_dc)])
            # B_curr = B_minus_plus doesn't change
            f_curr = np.array([loop_suppressing_B_theta_nm1.content[1+i],
                               loop_suppressing_B_theta_nm1.content[-2-i]])\
                               - A_prev * B_theta_prev - B_prev * dp_B_theta_prev
            B_theta_curr,_ = solve_integration_factor(coeff = A_curr,
                             coeff_dp = B_curr,
                             f = f_curr)

            B_theta_out[1+i] = B_theta_curr[0]
            B_theta_out[-2-i] = B_theta_curr[-1]

            B_theta_prev = B_theta_curr
            dp_B_theta_prev = ChiPhiFuncGrid(B_theta_prev).dphi().content
            m_prev = m_prev-2


    # Check if B_theta_n0 is provided for even n-1.
    if (n_eval-1)%2==0:
        if B_theta_n0 is None:
            raise AttributeError('iterate_B_theta_nm1_cp: B_theta_n-1,0 must '\
                                'be provided for even n-1.')
        B_theta_out[(n_eval-2)//2] = B_theta_n0
        return(ChiPhiFuncGrid(B_theta_out))
        # TODO: Y, Bpsi solver.
    # For odd n-1, B_theta_n0 will be solved.
    else:
        return(ChiPhiFuncGrid(B_theta_out), loop_suppressing_B_theta_nm1)

# Happens to have the same masking pattern with B_theta n-1 (since iterate_B_theta_nm1_cp
# was ran with B_theta_n = 0. As a result, this uses
# loop_suppressing_B_theta_nm1 from iterate_B_theta_nm1_cp.
def iterate_B_theta_n0_cp(loop_suppressing_B_theta_nm1,
    coeff_B_theta_nm1,
    coeff_dc_B_theta_nm1,
    coeff_dp_B_theta_nm1,
    coeff_B_theta_n0,
    coeff_dp_B_theta_n0,
    ):
    constant_component = (loop_suppressing_B_theta_nm1 - \
    - coeff_B_theta_nm1*B_theta_nm1 \
    - coeff_dc_B_theta_nm1*B_theta_nm1.dchi() \
    - coeff_dp_B_theta_nm1*B_theta_nm1.dphi()).get_constant().low_pass()
    B_theta_n0,_ = solve_integration_factor(
        coeff = coeff_B_theta_n0.content,
        coeff_dp = coeff_dp_B_theta_n0.content,
        f = constant_component.content, integral_mode = 'spline')

# Iteration for Y[n-1,0] and B_psi[n-3,0] is inside Equilibrium.
# It seems messy but avoids passing around known constants.

''' III. Equilibrium manager and Iterate '''

# A container for all equilibrium quantities.
# All coef inputs must be ChiPhiEpsFunc's.
class Equilibrium:
    def __init__(self,
        X_coef_cp,
        Y_coef_cp,
        Z_coef_cp,
        B_psi_coef_cp,
        B_theta_coef_cp,
        B_denom_coef_c,
        B_alpha_coef,
        kap_p, dl_p, tau_p,
        iota_coef, eta,
        B_theta_coef_np10,
        p_perp_coef_cp=None,
        Delta_coef_cp=None,
        noise=None
        ):

        # Variables being solved for are stored in dicts for
        # convenience of plotting and saving
        self.noise=noise
        self.unknown = {}
        self.constant = {}

        # The Equilibrium object always stores up to odd X_coef_cp
        # orders, and iterate 2 orders per iteration step.
        # This is due to iota[n-1/2] uses Zn.
        self.B_theta_np10 = B_theta_coef_np10

        self.unknown['X_coef_cp'] = X_coef_cp
        self.unknown['Y_coef_cp'] = Y_coef_cp
        self.unknown['Z_coef_cp'] = Z_coef_cp
        self.unknown['B_psi_coef_cp'] = B_psi_coef_cp
        self.unknown['B_theta_coef_cp'] = B_theta_coef_cp
        self.unknown['iota_coef'] = iota_coef
        self.unknown['p_perp_coef_cp'] = p_perp_coef_cp
        self.unknown['Delta_coef_cp'] = Delta_coef_cp

        self.constant['B_denom_coef_c'] = B_denom_coef_c
        self.constant['B_alpha_coef'] = B_alpha_coef
        self.constant['kap_p'] = kap_p
        self.constant['dl_p'] = dl_p
        self.constant['tau_p'] = tau_p
        self.constant['eta'] = eta

        # Pressure can be trivial
        current_order = X_coef_cp.get_order()
        if not self.unknown['p_perp_coef_cp']:
            self.unknown['p_perp_coef_cp'] = ChiPhiEpsFunc.zeros_to_order(current_order)
        if not self.unknown['Delta_coef_cp']:
            self.unknown['Delta_coef_cp'] = ChiPhiEpsFunc.zeros_to_order(current_order)

        # Manages noises
        if not noise:
            self.noise = {} # dict of dict managing types of noises
            # Tracks different types of noise
            self.noise['filter'] = {}
            for key in self.noise.keys():
                self.noise[key]['X_coef_cp'] = ChiPhiEpsFunc.zeros_like(X_coef_cp)
                self.noise[key]['Y_coef_cp'] = ChiPhiEpsFunc.zeros_like(Y_coef_cp)
                self.noise[key]['Z_coef_cp'] = ChiPhiEpsFunc.zeros_like(Z_coef_cp)
                self.noise[key]['B_psi_coef_cp'] = ChiPhiEpsFunc.zeros_like(B_psi_coef_cp)
                self.noise[key]['B_theta_coef_cp'] = ChiPhiEpsFunc.zeros_like(B_theta_coef_cp)
                self.noise[key]['p_perp_coef_cp'] = ChiPhiEpsFunc.zeros_like(self.unknown['p_perp_coef_cp'])
                self.noise[key]['Delta_coef_cp'] = ChiPhiEpsFunc.zeros_like(self.unknown['Delta_coef_cp'])

        # Check if every term is on the same order
        self.check_order_consistency()

        # Some coeffs are really long. We only calc them once.
        self.prepare_constants()

    # Replaced in function by get_Yn0_ODE_coeffs. Here for debug perpose only.
    # Fetches the coeff of B_psi[n-3,0] and Y[n-1,0] in the looped equation.
    def get_B_psi_nm30_coeffs(self, n):
        coeff = self.coeff_B_psi_nm3_pre/n
        coeff_dp = self.coeff_dp_B_psi_nm3_pre/n
        return(coeff, coeff_dp)

    # Looped equation
    def get_Y_nm1_coeffs(self, n):
        coeff = (n-1)/n*self.coeff_Y_nm1_pre
        coeff_dp = (n-1)/n*self.coeff_dp_Y_nm1_pre
        coeff_dpp = (n-1)/n*self.coeff_dpp_Y_nm1_pre
        return(coeff, coeff_dp, coeff_dpp)

    # Must be run with Yn-1 and Bpsin-3 with zero center components
    # Uses B_theta[n-1]
    # Uses a lot of pre-calculated coefficients.
    # returns B_psi[n-3] and Y[n-1] with center comp replaced.
    def iterate_B_psi_nm30_Y_nm10_p(n_eval, \
        X_coef_cp, Y_coef_cp, Z_coef_cp, \
        B_theta_coef_cp, B_psi_coef_cp, \
        B_alpha_coef, B_denom_coef_c, \
        p_perp_coef_cp, Delta_coef_cp, \
        kap_p, dl_p, tau_p, iota_coef, \
        test_equilibrium):

        if n_eval%2!=1:
            raise AttributeError('iterate_B_psi_nm30_Y_nm10_p: only applicable to even orders.')

        loop_no_mask = parsed.eval_loop(n_eval, \
            X_coef_cp.mask(n_eval-1).zero_append(), \
            Y_coef_cp.mask(n_eval-1).zero_append(), \
            Z_coef_cp.mask(n_eval-1).zero_append(), \
            B_theta_coef_cp.mask(n_eval-1).zero_append(), \
            B_psi_coef_cp.mask(n_eval-3).zero_append(),\
            B_alpha_coef, B_denom_coef_c.mask(n_eval-1).zero_append(), \
            p_perp_coef_cp.mask(n_eval-1).zero_append(), \
            Delta_coef_cp.mask(n_eval-1).zero_append(), \
            kap_p, dl_p, tau_p, iota_coef).cap_m(n_eval-2)

        len_chi = loop_no_mask.get_shape()[0]
        len_phi = loop_no_mask.get_shape()[1]

        # We take the second center component.
        f_raw = loop_no_mask.content[len_chi//2]
        f_eff = test_equilibrium.get_f_eff(f_raw, n_eval)

        Y_ppp_coef_n, Y_pp_coef_n, Y_p_coef_n, Y_coef_n = test_equilibrium.get_Yn0_ODE_coeffs(n_eval)

        # FFT matrices
        Y_ppp_coef_fft = np.fft.fft(Y_ppp_coef_n)
        Y_pp_coef_fft = np.fft.fft(Y_pp_coef_n)
        Y_p_coef_fft = np.fft.fft(Y_p_coef_n)
        Y_coef_fft = np.fft.fft(Y_coef_n)

        # Of shape [n_chi, n_phi_row, n_phi_col]
        Y_ppp_coef_matrix = fft_conv_op_batch(np.array([Y_ppp_coef_fft]))
        Y_pp_coef_matrix = fft_conv_op_batch(np.array([Y_pp_coef_fft]))
        Y_p_coef_matrix = fft_conv_op_batch(np.array([Y_p_coef_fft]))
        Y_coef_matrix = fft_conv_op_batch(np.array([Y_coef_fft]))

        diff_matrix = fft_dphi_op(len_phi)

        linear_diff_op = Y_ppp_coef_matrix@diff_matrix@diff_matrix@diff_matrix\
            +Y_pp_coef_matrix@diff_matrix@diff_matrix\
            +Y_p_coef_matrix@diff_matrix\
            +Y_coef_matrix
        linear_diff_op_inv = np.linalg.inv(linear_diff_op)

        Ynm10 = ChiPhiFuncGrid(
            np.fft.ifft(
                (linear_diff_op_inv@np.fft.fft(RHS.content)[:,:,None])[:,:,0]
            )
        )

        Bnm30 = test_equilibrium.get_B_psi_nm30(
            y_pp=Ynm10.dphi().dphi().content[0],
            y_p=Ynm10.dphi().content[0],
            y=Ynm10.content[0], f_raw=f_raw)

        new_Y = Y_coef_cp[n_eval-1].replace_constant(Ynm10)
        new_Bpsi = B_psi_coef_cp[n_eval-3].replace_constant(Bnm30)

        return(new_Bpsi, new_Y)

    # The coeffs in Yn-1 ODE's are complex factors multiplied with powers of
    # n and n+-1. The Equilibrium object pre-calculates these quantities
    # once (see prepare_constants) and applies the powers of n when they
    # are needed. These are functions that apply the factors of n.
    # SPECIAL NOTE: coeffs returned by this method are 1d arrays, rather
    # than contents or ChiPhiFuncGrids.
    def get_Yn0_ODE_coeffs(self, n):
        return(
            self.y_ppp_coef_pre*(n-1)/(n**3),
            self.y_pp_coef_pre*(n-1)/(n**3),
            self.y_p_coef_pre*(n-1)/(n**3),
            self.y_coef_pre*(n-1)/(n**3)
        )

    # f_eff from f and stored constants.
    def get_f_eff(self, f_raw, n):
        return(
            dphi_direct(np.array([np.imag(f*np.conj(self.ODE3_d))]))*self.ODE3_d/(n**2)
            + np.imag(f*np.conj(self.ODE3_d))*self.ODE3_e/(n**2)
            + f*self.ODE3_e_b/(n**2)
        )

    def get_B_psi_nm30(self, Y_pp, Y_p, Y, f_raw):
        return( \
            ( \
                self.ODE3_a_b*Y_pp*(n-1)*n \
                +self.ODE3_b_b*Y_p*(n-1)*n \
                +self.ODE3_c_b*Y*(n-1)*n \
                -f_raw*(n**2) \
            )/self.ODE3_e_b \
        )

    # Order consistency check --------------------------------------------------
    # Get the current order of an equilibrium
    def get_order(self):
        self.check_order_consistency()
        return(self.unknown['X_coef_cp'].get_order())

    # Check of all terms are on consistent orders
    def check_order_consistency(self):

        # An internal method for error throwing. Checks if all variables are
        # iterated to the correct order.
        def check_order_individial(ChiPhiEpsFunc, name_in_error_msg, n_req):
            n_current = ChiPhiEpsFunc.get_order()
            if n_current!=n_req:
                raise AttributeError(name_in_error_msg
                    + ' has order: '
                    + str(n_current)
                    + ', while required order is: '
                    + str(n_req))

        n = self.unknown['X_coef_cp'].get_order()
        if n%2!=1:
            raise AttributeError('X_coef_cp has even order. Equilibrium is managed'\
            ' and updated every 2 orders.')

        check_order_individial(self.unknown['Y_coef_cp'], 'Y_coef_cp', n)
        check_order_individial(self.unknown['Z_coef_cp'], 'Z_coef_cp', n)
        check_order_individial(self.unknown['B_psi_coef_cp'], 'B_psi_coef_cp', n-2)
        check_order_individial(self.unknown['B_theta_coef_cp'], 'B_theta_coef_cp', n-1)
        check_order_individial(self.constant['B_denom_coef_c'], 'B_denom_coef_c', n)
        check_order_individial(self.constant['B_alpha_coef'], 'B_alpha_coef', (n-1)//2)
        check_order_individial(self.unknown['iota_coef'], 'iota_coef', (n-1)//2)
        check_order_individial(self.unknown['p_perp_coef_cp'], 'p_perp_coef_cp', n)
        check_order_individial(self.unknown['Delta_coef_cp'], 'Delta_coef_cp', n)

        for key in self.noise.keys():
            check_order_individial(self.noise[key]['Y_coef_cp'], 'Y_coef_cp', n)
            check_order_individial(self.noise[key]['Z_coef_cp'], 'Z_coef_cp', n)
            check_order_individial(self.noise[key]['B_psi_coef_cp'], 'B_psi_coef_cp', n-2)
            check_order_individial(self.noise[key]['B_theta_coef_cp'], 'B_theta_coef_cp', n-1)
            check_order_individial(self.noise[key]['p_perp_coef_cp'], 'p_perp_coef_cp', n)
            check_order_individial(self.noise[key]['Delta_coef_cp'], 'Delta_coef_cp', n)

    # The looped equation contains some complicated constants.
    # They are only evaluated once during initialization.
    # Refer to the looped equation Maxima notebook for details.
    def prepare_constants(self):
        # Constants: only calculated once:
        # Loop equation coeffs
        X_coef_cp = self.unknown['X_coef_cp']
        Y_coef_cp = self.unknown['Y_coef_cp']
        Z_coef_cp = self.unknown['Z_coef_cp']
        iota_coef = self.unknown['iota_coef']
        p_perp_coef_cp = self.unknown['p_perp_coef_cp']
        Delta_coef_cp = self.unknown['Delta_coef_cp']

        # Masking all init conds.
        B_denom_coef_c = self.constant['B_denom_coef_c']
        B_alpha_coef = self.constant['B_alpha_coef']
        dl_p = self.constant['dl_p']
        tau_p = self.constant['tau_p']
        eta = self.constant['eta']

        self.coeff_B_theta_nm1 = -(\
            B_denom_coef_c[0]**2*(diff(p_perp_coef_cp[1],'phi',1))\
            +B_denom_coef_c[0]**2*iota_coef[0]*diff(p_perp_coef_cp[1],'chi',1)\
            +0.5*(Delta_coef_cp[0]-2)*iota_coef[0]*diff(B_denom_coef_c[1],'chi',1)\
            +2*B_denom_coef_c[0]*B_denom_coef_c[1]*diff(p_perp_coef_cp[0],'phi',1)\
            +B_denom_coef_c[1]*(diff(Delta_coef_cp[0],'phi',1)))
        self.coeff_dc_B_theta_nm1 = (\
            B_denom_coef_c[0]*iota_coef[0]\
            *Delta_coef_cp[1])
        self.coeff_dp_B_theta_nm1 = B_denom_coef_c[0]*Delta_coef_cp[1]

        self.coeff_B_theta_n0 = -B_denom_coef_c[0]**2*(diff(p_perp_coef_cp[0],'phi',1))
        self.coeff_dp_B_theta_n0 = B_denom_coef_c[0]*Delta_coef_cp[0]-B_denom_coef_c[0]

        #----------------------------------------------------------------------------
        # Debug only. Functionally replaced by the ODE3_#'s and y_coef_pre's.
        self.coeff_B_psi_nm3_pre = -(\
            2*B_denom_coef_c[0]*iota_coef[0]*(diff(Delta_coef_cp[1],'chi',2))\
            +2*B_denom_coef_c[0]*(diff(Delta_coef_cp[1],'chi',1,'phi',1))\
            +(2-2*Delta_coef_cp[0])*iota_coef[0]*(diff(B_denom_coef_c[1],'chi',2)))
        self.coeff_dp_B_psi_nm3_pre = -(2*B_denom_coef_c[0]*\
            diff(Delta_coef_cp[1],'chi',1))
        self.coeff_Y_nm1_pre = ((2*Delta_coef_cp[0]-2)*(diff(X_coef_cp[1],'chi',1))*dl_p*(diff(tau_p,'phi',1))\
            +((2*Delta_coef_cp[0]-2)*iota_coef[0]*(diff(X_coef_cp[1],'chi',2))\
            +(2*Delta_coef_cp[0]-2)*(diff(X_coef_cp[1],'chi',1,'phi',1))\
            +2*(diff(Delta_coef_cp[0],'phi',1))*(diff(X_coef_cp[1],'chi',1)))*dl_p*tau_p\
            +(1-1*Delta_coef_cp[0])*iota_coef[0]**2*(diff(Y_coef_cp[1],'chi',3))\
            +(2-2*Delta_coef_cp[0])*iota_coef[0]*(diff(Y_coef_cp[1],'chi',2,'phi',1))\
            -1*iota_coef[0]*(diff(Delta_coef_cp[0],'phi',1))*(diff(Y_coef_cp[1],'chi',2))\
            +(1-1*Delta_coef_cp[0])*(diff(Y_coef_cp[1],'chi',1,'phi',2))\
            -1*(diff(Delta_coef_cp[0],'phi',1))*(diff(Y_coef_cp[1],'chi',1,'phi',1))\
            )/B_alpha_coef[0]
        self.coeff_dp_Y_nm1_pre = (\
            2*(Delta_coef_cp[0]-1)*diff(X_coef_cp[1],'chi',1)*dl_p*tau_p\
            +diff(Delta_coef_cp[0],'phi',1)*diff(Y_coef_cp[1],'chi',1)\
            )/B_alpha_coef[0]
        self.coeff_dpp_Y_nm1_pre = (Delta_coef_cp[0]-1)*diff(Y_coef_cp[1],'chi',1)/B_alpha_coef[0]
        #----------------------------------------------------------------------------




        # Coeff for B_psi[n-3,0] and Y[n-1,0]. Will not be used elsewhere.
        coeff_B_psi_nm3_pre = -(\
            2*B_denom_coef_c[0]*iota_coef[0]*(diff(Delta_coef_cp[1],'chi',2))\
            +2*B_denom_coef_c[0]*(diff(Delta_coef_cp[1],'chi',1,'phi',1))\
            +(2-2*Delta_coef_cp[0])*iota_coef[0]*(diff(B_denom_coef_c[1],'chi',2)))
        coeff_dp_B_psi_nm3_pre = -(2*B_denom_coef_c[0]*\
            diff(Delta_coef_cp[1],'chi',1))
        coeff_Y_nm1_pre = (
            (2*Delta_coef_cp[0]-2)*(diff(X_coef_cp[1],'chi',1))*dl_p*(diff(tau_p,'phi',1))\
            +((2*Delta_coef_cp[0]-2)*iota_coef[0]*(diff(X_coef_cp[1],'chi',2))\
            +(2*Delta_coef_cp[0]-2)*(diff(X_coef_cp[1],'chi',1,'phi',1))\
            +2*(diff(Delta_coef_cp[0],'phi',1))*(diff(X_coef_cp[1],'chi',1)))*dl_p*tau_p\
            +(1-1*Delta_coef_cp[0])*iota_coef[0]**2*(diff(Y_coef_cp[1],'chi',3))\
            +(2-2*Delta_coef_cp[0])*iota_coef[0]*(diff(Y_coef_cp[1],'chi',2,'phi',1))\
            -1*iota_coef[0]*(diff(Delta_coef_cp[0],'phi',1))*(diff(Y_coef_cp[1],'chi',2))\
            +(1-1*Delta_coef_cp[0])*(diff(Y_coef_cp[1],'chi',1,'phi',2))\
            -1*(diff(Delta_coef_cp[0],'phi',1))*(diff(Y_coef_cp[1],'chi',1,'phi',1))\
            )/B_alpha_coef[0]
        coeff_dp_Y_nm1_pre = (\
            2*(Delta_coef_cp[0]-1)*diff(X_coef_cp[1],'chi',1)*dl_p*tau_p\
            +diff(Delta_coef_cp[0],'phi',1)*diff(Y_coef_cp[1],'chi',1)\
            )/B_alpha_coef[0]
        coeff_dpp_Y_nm1_pre = (Delta_coef_cp[0]-1)*diff(Y_coef_cp[1],'chi',1)/B_alpha_coef[0]

        # Evaluating the coeffs for the 3rd order linear ODE on Yn0
        # These are the coefficient for Y[n-1,0] and B_psi[n-3,0]
        # when their ODE is written as:
        # aY[n-1,0]             + bY[n-1,0]           + cY[n-1,0]
        # + dB_psi[n-3,0]       + eB_psi[n-3,0]       = f
        # conv(a)Y[n-1,0]       + conv(b)Y[n-1,0]     + conv(c)Y[n-1,0]
        # + conv(d)B_psi[n-3,0] + conv(e)B_psi[n-3,0] = conv(f)
        # These quantities are poorly named but they will be used
        # again for f.
        # However, note that these are a different version that does not include
        # the n powers. (see get_Yn0_ODE_coeffs)
        # compared to a - e in the notebook,
        # a, b, c = ODE3_a, ODE3_b, ODE3_c*(n-1)
        # d, e = ODE3_d, ODE3_e/n
        # a_b, b_b, c_b = ODE3_a, ODE3_b, ODE3_c*(n-1)/n
        # e_b = ODE3_d, ODE3_e/(n**2)
        self.ODE3_c, self.ODE3_b, self.ODE3_a = \
            coeff_Y_nm1_pre.content[1], \
            coeff_dp_Y_nm1_pre.content[1], \
            coeff_dpp_Y_nm1_pre.content[1]
        self.ODE3_e, self.ODE3_d = \
            coeff_B_psi_nm3_pre.content[1], \
            coeff_dp_B_psi_nm3_pre.content[1]
        self.ODE3_a_b = np.imag(self.ODE3_a*np.conj(self.ODE3_d))
        self.ODE3_b_b = np.imag(self.ODE3_b*np.conj(self.ODE3_d))
        self.ODE3_c_b = np.imag(self.ODE3_c*np.conj(self.ODE3_d))
        self.ODE3_e_b = np.imag(self.ODE3_e*np.conj(self.ODE3_d))

        # The differential equation is solved by elimating B_psi[n-3,0] to
        # produce
        # y_ppp_coef Y[n-1,0]''' + y_pp_coef Y[n-1,0]'' +
        # y_p_coef Y[n-1,0]'     + y_coef Y[n-1,0]
        # = f_eff. (Please refer to 3rd order lin ODE.ipynb for the derivation.)
        # Most of the calculations involved in the elimination is done here
        # only once, and the coeffs and f_eff can be generated with
        # self.get_Yn0_ODE_coeffs(n) and self.get_f_eff(f, n)
        self.y_ppp_coef_pre = self.ODE3_a_b*self.ODE3_d
        self.y_pp_coef_pre = (\
            dphi_direct(\
                np.array([self.ODE3_a_b])\
            )[0]*self.ODE3_d\
            + self.ODE3_b_b*self.ODE3_d\
            + self.ODE3_a_b*self.ODE3_e\
            + self.ODE3_a*self.ODE3_e_b)
        self.y_p_coef_pre = (\
            dphi_direct(\
                np.array([self.ODE3_b_b])\
            )[0]*self.ODE3_d\
            + self.ODE3_c_b*self.ODE3_d\
            + self.ODE3_b_b*self.ODE3_e\
            + self.ODE3_b*self.ODE3_e_b)
        self.y_coef_pre = (\
            dphi_direct(\
                np.array([self.ODE3_c_b])\
            )[0]*self.ODE3_d\
            + self.ODE3_c_b*self.ODE3_e\
            + self.ODE3_c*self.ODE3_e_b)

        self.y_ppp_coef_pre = low_pass_direct(np.array([self.y_ppp_coef_pre]))[0]
        self.y_pp_coef_pre = low_pass_direct(np.array([self.y_pp_coef_pre]))[0]
        self.y_p_coef_pre = low_pass_direct(np.array([self.y_p_coef_pre]))[0]
        self.y_coef_pre = low_pass_direct(np.array([self.y_coef_pre]))[0]

    # Iteration ----------------------------------------------------------------
    # Evaluates 2 entire orders. Note that no masking is needed for any of the methods
    # defined in this file. Copies the equilibrium.
    def iterate_2_full(self,
        B_alpha_nm1d2,
        B_denom_nm1, B_denom_n,
        n_eval=None,
        integral_mode = 'double_spline',
        filter=True, filter_mode='low_pass', filter_arg=100):

        if n_eval > self.get_order() + 2:
            raise AttributeError('iterate_2_full: can only iterate'\
            ' to the next two order, or repeat previous orders.')

        # If no order is supplied, then iterate to the next order. the Equilibrium
        # will be edited directly.
        if n_eval == None:
            n_eval = self.get_order() + 2 # getting order and checking consistency
            B_theta_nm10 = self.B_theta_np10
        elif n_eval%2 != 1:
            raise ValueError("n_eval must be even.")
        else:
            B_theta_nm10 = self.unknown['B_theta_coef_cp'][n_eval-1].get_constant()


        start_time = time.time()
        print("Evaluating order",n_eval-1, n_eval)

        # Creating new ChiPhiEpsFunc's for the resulting Equilibrium
        X_coef_cp = self.unknown['X_coef_cp'].mask(n_eval-2)
        Y_coef_cp = self.unknown['Y_coef_cp'].mask(n_eval-2)
        Z_coef_cp = self.unknown['Z_coef_cp'].mask(n_eval-2)
        B_theta_coef_cp = self.unknown['B_theta_coef_cp'].mask(n_eval-2)
        B_psi_coef_cp = self.unknown['B_psi_coef_cp'].mask(n_eval-4)
        iota_coef = self.unknown['iota_coef'].mask((n_eval-3)//2)
        p_perp_coef_cp = self.unknown['p_perp_coef_cp'].mask(n_eval-2)
        Delta_coef_cp = self.unknown['Delta_coef_cp'].mask(n_eval-2)
        # Masking all init conds.
        B_denom_coef_c = self.constant['B_denom_coef_c'].mask(n_eval-2)
        B_alpha_coef = self.constant['B_alpha_coef'].mask((n_eval-3)//2)
        kap_p = self.constant['kap_p']
        dl_p = self.constant['dl_p']
        tau_p = self.constant['tau_p']
        eta = self.constant['eta']
        # Appending free functions and initial conditions

        B_alpha_coef.append(B_alpha_nm1d2)
        B_denom_coef_c.append(B_denom_nm1)
        B_denom_coef_c.append(B_denom_n)

        # Evaluating order n_eval-1. Even order,
        # Requires:
        # X_{n-1}, Y_{n-1}, Z_{n-1},
        # B_{\theta n-1}, B_0,
        # B_{\alpha 0}, \bar{\iota}_{(n-2)/2 or (n-3)/2}$
        # The next order uses B_psi with ZERO constant comp.
        B_psi_nm3 = iterate_dc_Bpsi_nm2(n_eval-1,
            X_coef_cp, Y_coef_cp, Z_coef_cp,
            B_theta_coef_cp, B_psi_coef_cp,
            B_alpha_coef, B_denom_coef_c,
            kap_p, dl_p, tau_p,
            iota_coef).integrate_chi(ignore_mode_0=True)
        B_psi_nm3.content[B_psi_nm3.get_shape()[0]//2] = B_psi_nm30

        self.noise['filter']['B_psi_coef_cp'].append(
            B_psi_nm3.noise_filter(filter_mode, filter_arg)
        )
        if filter:
            B_psi_nm3 = B_psi_nm3.filter(filter_mode, filter_arg)
        B_psi_coef_cp.append(B_psi_nm3)

        # B_theta_n-1 requires:
        # Xn-1, Yn-1, Zn-1, B_theta_n-2, Delta_n-1,
        # B_psi_n-3, B_denom n-1,
        # iota_coef (n-3)/2 or (n-2)/2, and B_alpha (n-1)/2 or (n-2)/2
        # Here, "n-1" = n-1
        coeff_B_theta_nm1 = self.coeff_B_theta_nm1
        coeff_dc_B_theta_nm1 = self.coeff_dc_B_theta_nm1
        coeff_dp_B_theta_nm1 = self.coeff_dp_B_theta_nm1
        coeff_B_theta_n0 = self.coeff_B_theta_n0
        coeff_dp_B_theta_n0 = self.coeff_dp_B_theta_n0

        # We want B_theta[n_eval-1]. This means the input order is n_eval,
        # and at such orders we solve for B_psi and Y's zero component.
        B_theta_nm1 = iterate_B_theta_nm1_cp(n_eval,
            X_coef_cp, Y_coef_cp, Z_coef_cp,
            B_theta_coef_cp, B_psi_coef_cp,
            B_alpha_coef, B_denom_coef_c,
            p_perp_coef_cp,
            Delta_coef_cp,
            kap_p, dl_p, tau_p, iota_coef,
            coeff_B_theta_nm1, # test_equilibrium.coeff_B_theta_nm1
            coeff_dc_B_theta_nm1,
            coeff_dp_B_theta_nm1,
            integral_mode, B_theta_n0=B_theta_nm10)
        B_psi_nm3_new, Ynm1_new = iterate_B_psi_nm30_Y_nm10_p(n_eval, \
            X_coef_cp, Y_coef_cp, Z_coef_cp, \
            B_theta_coef_cp, B_psi_coef_cp, \
            B_alpha_coef, B_denom_coef_c, \
            p_perp_coef_cp, Delta_coef_cp, \
            kap_p, dl_p, tau_p, iota_coef, \
            test_equilibrium)
        B_psi_coef_cp = B_psi_coef_cp.mask(n_eval-4).append(B_psi_nm3)
        Y_coef_cp = Y_coef_cp.mask(n_eval-2).append(Ynm1)


        # Zn requires:
        # Here, "n" = n-1
        # X_{n-1}, Y_{n-1}, Z_{n-1},
        # B_{\theta n}, B_{\psi n-2},
        # B_{\alpha (n-2)/2 or (n-3)/2},
        # \iota_{(n-2)/2 or (n-3)/2}
        # \kappa, \frac{dl}{d\phi}, \tau
        Znm1 = iterate_Zn_cp(n_eval-1,
            X_coef_cp, Y_coef_cp, Z_coef_cp,
            B_theta_coef_cp, B_psi_coef_cp,
            B_alpha_coef,
            kap_p, dl_p, tau_p,
            iota_coef)
        self.noise['filter']['Z_coef_cp'].append(
            Znm1.noise_filter(filter_mode, filter_arg)
        )
        if filter:
            Znm1 = Znm1.filter(filter_mode, filter_arg)
        Z_coef_cp.append(Znm1)

        # Xn requires:
        # X_{n-1}, Y_{n-1}, Z_n,
        # \iota_{(n-3)/2 or (n-2)/2},
        # B_{\alpha (n-1)/2 or n/2}.
        # Here, "n" = n-1
        Xnm1 = iterate_Xn_cp(n_eval-1,
            X_coef_cp,
            Y_coef_cp,
            Z_coef_cp,
            B_denom_coef_c,
            B_alpha_coef,
            kap_p, dl_p, tau_p,
            iota_coef)
        self.noise['filter']['X_coef_cp'].append(Xnm1.noise_filter(filter_mode, filter_arg))
        if filter:
            Xnm1 = Xnm1.filter(filter_mode, filter_arg)
        X_coef_cp.append(Xnm1)

        # Yn requires:
        # X_{n}, Y_{n-1}, Z_{n-1},
        # B_{\theta n-1}, B_{\psi n-3},
        # \iota_{(n-3)/2 or (n-4)/2}, B_{\alpha (n-1)/2 or (n-2)/2}
        # Here, "n" = n-1
        # The zeroth component will be left zero
        # until evaluated in the next order.
        Ynm1 = iterate_Yn_cp(n_eval-1,
            X_coef_cp,
            Y_coef_cp,
            Z_coef_cp,
            B_psi_coef_cp,
            B_theta_coef_cp,
            B_alpha_coef,
            kap_p, dl_p, tau_p,
            iota_coef,
            Yn_free = 0)
        self.noise['filter']['Y_coef_cp'].append(Ynm1.noise_filter(filter_mode, filter_arg))
        if filter:
            Ynm1 = Ynm1.filter(filter_mode, filter_arg)
        Y_coef_cp.append(Ynm1)

        # Uses B_denom[n], p_[\perp n]$, $\Delta_[n-1]$, $\iota_[(n-1)/2, n/2]$
        # TODO Integration constant for zeroth mode. It's just y'=f, and the integration
        # constant need to satisfy integral(Bpsi') = 0
        Delta_nm1 = iterate_delta_n(n_eval-1,
            B_denom_coef_c,
            p_perp_coef_cp,
            Delta_coef_cp,
            iota_coef)
        self.noise['filter']['Delta_coef_cp'].append(Delta_nm1.noise_filter(filter_mode, filter_arg))
        if filter:
            Delta_nm1 = Delta_nm1.filter(filter_mode, filter_arg)
        Delta_coef_cp.append(Delta_nm1)

        # Uses B_theta [n-2], B_[psi n-2], B_alpha first-order terms, B_denom[n],
        # p_[perp n-1], Delta_[n-1],iota_[(n-3)/2 (n-2)/2]
        p_perp_nm1 = iterate_p_perp_n(n_eval-1,
            B_theta_coef_cp,
            B_psi_coef_cp,
            B_alpha_coef,
            B_denom_coef_c,
            p_perp_coef_cp,
            Delta_coef_cp,
            iota_coef)
        self.noise['filter']['p_perp_coef_cp'].append(p_perp_nm1.noise_filter(filter_mode, filter_arg))
        if filter:
            p_perp_nm1 = p_perp_nm1.filter(filter_mode, filter_arg)
        p_perp_coef_cp.append(p_perp_nm1)

        # no need to ignore_mode_0 for chi integral. This is an odd order.
        B_psi_nm2 = iterate_dc_Bpsi_nm2(n_eval,
            X_coef_cp, Y_coef_cp, Z_coef_cp,
            B_theta_coef_cp, B_psi_coef_cp,
            B_alpha_coef, B_denom_coef_c,
            kap_p, dl_p, tau_p,
            iota_coef).integrate_chi()
        self.noise['filter']['B_psi_coef_cp'].append(B_psi_nm2.noise_filter(filter_mode, filter_arg))
        if filter:
            B_psi_nm2 = B_psi_nm2.filter(filter_mode, filter_arg)
        B_psi_coef_cp.append(B_psi_nm2)

        Zn = iterate_Zn_cp(n_eval,
            X_coef_cp, Y_coef_cp, Z_coef_cp,
            B_theta_coef_cp, B_psi_coef_cp,
            B_alpha_coef,
            kap_p, dl_p, tau_p,
            iota_coef)
        self.noise['filter']['Z_coef_cp'].append(Zn.noise_filter(filter_mode, filter_arg))
        if filter:
            Zn = Zn.filter(filter_mode, filter_arg)
        Z_coef_cp.append(Zn)

        Xn = iterate_Xn_cp(n_eval,
            X_coef_cp,
            Y_coef_cp,
            Z_coef_cp,
            B_denom_coef_c,
            B_alpha_coef,
            kap_p, dl_p, tau_p,
            iota_coef)
        self.noise['filter']['X_coef_cp'].append(Xn.noise_filter(filter_mode, filter_arg))
        if filter:
            Xn = Xn.filter(filter_mode, filter_arg)
        X_coef_cp.append(Xn)

        Yn1c = iterate_Yn1c_p(n, X_coef_cp, Y_coef_cp, Z_coef_cp,\
                          iota_coef,\
                          tau_p, dl_p, kap_p, eta,\
                          B_denom_coef_c, B_alpha_coef,
                          B_psi_coef_cp, B_theta_coef_cp)
        Yn = iterate_Yn_cp(n_eval,
            X_coef_cp,
            Y_coef_cp,
            Z_coef_cp,
            B_psi_coef_cp,
            B_theta_coef_cp,
            B_alpha_coef,
            kap_p, dl_p, tau_p,
            iota_coef,
            Y_free_n = Yn1c)
        self.noise['filter']['Y_coef_cp'].append(
            Yn.noise_filter(filter_mode, filter_arg)
        )
        if filter:
            Yn = Yn.filter(filter_mode, filter_arg)
        Y_coef_cp.append(Yn)

        Delta_n = iterate_delta_n(n_eval,
            B_denom_coef_c,
            p_perp_coef_cp,
            Delta_coef_cp,
            iota_coef)
        self.noise['filter']['Delta_coef_cp'].append(
            Delta_n.noise_filter(filter_mode, filter_arg)
        )
        if filter:
            Delta_n = Delta_n.filter(filter_mode, filter_arg)
        Delta_coef_cp.append(Delta_n)

        p_perp_n = iterate_p_perp_n(n_eval,
            B_theta_coef_cp,
            B_psi_coef_cp,
            B_alpha_coef,
            B_denom_coef_c,
            p_perp_coef_cp,
            Delta_coef_cp,
            iota_coef)
        self.noise['filter']['p_perp_coef_cp'].append(
            p_perp_n.noise_filter(filter_mode, filter_arg)
        )
        if filter:
            p_perp_n = p_perp_n.filter(filter_mode, filter_arg)
        p_perp_coef_cp.append(p_perp_n)

        # n_eval is odd. This means that to solve for B_theta_n, the
        # input order is n_eval+1. In such order we solve for
        # B_theta[n_eval+1,0]
        B_theta_n,loop_suppressing_B_theta_nm1 = iterate_B_theta_nm1_cp(n_eval+1,
            X_coef_cp, Y_coef_cp, Z_coef_cp,
            B_theta_coef_cp, B_psi_coef_cp,
            B_alpha_coef, B_denom_coef_c,
            p_perp_coef_cp,
            Delta_coef_cp,
            kap_p, dl_p, tau_p, iota_coef,
            coeff_B_theta_nm1, # test_equilibrium.coeff_B_theta_nm1
            coeff_dc_B_theta_nm1,
            coeff_dp_B_theta_nm1,
            integral_mode, B_theta_n0)
        B_theta_np10 = iterate_B_theta_n0_cp(loop_suppressing_B_theta_nm1,
                    coeff_B_theta_nm1,
                    coeff_dc_B_theta_nm1,
                    coeff_dp_B_theta_nm1,
                    coeff_B_theta_n0,
                    coeff_dp_B_theta_n0,
                    )

        # Requires:
        # \Lambda_n (X_{n-1}, Y_{n-1}, Z_{n}, \iota_{(n-3)/2}),
        # B_{\theta n}, B_0, B_{\alpha 0}$
        iota_nm1b2 = iterate_iota_nm1b2(n_eval,
            X_coef_cp, Y_coef_cp, Z_coef_cp,\
            tau_p, dl_p, kap_p,\
            iota_coef, eta,\
            B_denom_coef_c, B_alpha_coef, B_theta_coef_cp)
        iota_coef.append(iota_nm1b2)

        print("Time elapsed(s):",(time.time() - start_time))
        print('X_coef_cp', X_coef_cp.get_order())
        print('B_psi_coef_cp', B_psi_coef_cp.get_order())
        print('B_theta_coef_cp', B_theta_coef_cp.get_order())


        return(Equilibrium(X_coef_cp,
        Y_coef_cp,
        Z_coef_cp,
        B_psi_coef_cp,
        B_theta_coef_cp,
        B_denom_coef_c,
        B_alpha_coef,
        kap_p, dl_p, tau_p,
        iota_coef, eta,
        B_theta_nm10,
        p_perp_coef_cp,
        Delta_coef_cp, self.noise))


    # Iterates the magnetic equations only.
    # Calculates Xn, Yn, Zn, B_psin-2 for 2 orders from lower order values.
    # B_theta, B_psi_nm30, Y_free_nm1 are all free.
    # n_eval must be odd.
    def iterate_2_magnetic_only(self,
        B_theta_nm1, B_theta_n,
        B_psi_nm30,
        B_alpha_nm1d2,
        B_denom_nm1, B_denom_n,
        Y_free_nm1, Y_free_n,
        p_perp_nm1=0, p_perp_n=0,
        Delta_nm1=0, Delta_n=0,
        n_eval=None, filter=True, filter_mode='low_pass', filter_arg=100):


        # If no order is supplied, then iterate to the next order. the Equilibrium
        # will be edited directly.
        if n_eval == None:
            n_eval = self.get_order() + 2 # getting order and checking consistency
        elif n_eval%2 != 1:
            raise ValueError("n must be even to evaluate iota_{(n-1)/2}")

        start_time = time.time()
        print("Evaluating order",n_eval-1, n_eval)

        # Creating new ChiPhiEpsFunc's for the resulting Equilibrium
        X_coef_cp = self.unknown['X_coef_cp'].mask(n_eval-2)
        Y_coef_cp = self.unknown['Y_coef_cp'].mask(n_eval-2)
        Z_coef_cp = self.unknown['Z_coef_cp'].mask(n_eval-2)
        B_theta_coef_cp = self.unknown['B_theta_coef_cp'].mask(n_eval-2)
        B_psi_coef_cp = self.unknown['B_psi_coef_cp'].mask(n_eval-4)
        iota_coef = self.unknown['iota_coef'].mask((n_eval-3)//2)
        p_perp_coef_cp = self.unknown['p_perp_coef_cp'].mask(n_eval-2)
        Delta_coef_cp = self.unknown['Delta_coef_cp'].mask(n_eval-2)
        # Masking all init conds.
        B_denom_coef_c = self.constant['B_denom_coef_c'].mask(n_eval-2)
        B_alpha_coef = self.constant['B_alpha_coef'].mask((n_eval-3)//2)
        kap_p = self.constant['kap_p']
        dl_p = self.constant['dl_p']
        tau_p = self.constant['tau_p']
        eta = self.constant['eta']
        # Appending free functions and initial conditions
        B_theta_coef_cp.append(B_theta_nm1)
        B_theta_coef_cp.append(B_theta_n)
        B_alpha_coef.append(B_alpha_nm1d2)
        B_denom_coef_c.append(B_denom_nm1)
        B_denom_coef_c.append(B_denom_n)
        p_perp_coef_cp.append(p_perp_nm1)
        p_perp_coef_cp.append(p_perp_n)
        Delta_coef_cp.append(Delta_nm1)
        Delta_coef_cp.append(Delta_n)
        # Managing noise
        self.noise['filter']['B_theta_coef_cp'].append(0)
        self.noise['filter']['p_perp_coef_cp'].append(0)
        self.noise['filter']['Delta_coef_cp'].append(0)
        self.noise['filter']['B_theta_coef_cp'].append(0)
        self.noise['filter']['p_perp_coef_cp'].append(0)
        self.noise['filter']['Delta_coef_cp'].append(0)

        # Evaluating order n_eval-1
        # Requires:
        # X_{n-1}, Y_{n-1}, Z_{n-1},
        # B_{\theta n-1}, B_0,
        # B_{\alpha 0}, \bar{\iota}_{(n-2)/2 or (n-3)/2}$
        B_psi_nm3 = iterate_dc_Bpsi_nm2(n_eval-1,
            X_coef_cp, Y_coef_cp, Z_coef_cp,
            B_theta_coef_cp, B_psi_coef_cp,
            B_alpha_coef, B_denom_coef_c,
            kap_p, dl_p, tau_p,
            iota_coef).integrate_chi(ignore_mode_0=True)
        B_psi_nm3.content[B_psi_nm3.get_shape()[0]//2] = B_psi_nm30

        self.noise['filter']['B_psi_coef_cp'].append(
            B_psi_nm3.noise_filter(filter_mode, filter_arg)
        )
        if filter:
            B_psi_nm3 = B_psi_nm3.filter(filter_mode, filter_arg)
        B_psi_coef_cp.append(B_psi_nm3)

        # Requires:
        # X_{n-1}, Y_{n-1}, Z_{n-1},
        # B_{\theta n}, B_{\psi n-2},
        # B_{\alpha (n-2)/2 or (n-3)/2},
        # \iota_{(n-2)/2 or (n-3)/2}
        # \kappa, \frac{dl}{d\phi}, \tau
        Znm1 = iterate_Zn_cp(n_eval-1,
            X_coef_cp, Y_coef_cp, Z_coef_cp,
            B_theta_coef_cp, B_psi_coef_cp,
            B_alpha_coef,
            kap_p, dl_p, tau_p,
            iota_coef)
        self.noise['filter']['Z_coef_cp'].append(
            Znm1.noise_filter(filter_mode, filter_arg)
        )
        if filter:
            Znm1 = Znm1.filter(filter_mode, filter_arg)
        Z_coef_cp.append(Znm1)

        # Requires:
        # X_{n-1}, Y_{n-1}, Z_n,
        # \iota_{(n-3)/2 or (n-2)/2},
        # B_{\alpha (n-1)/2 or n/2}.
        Xnm1 = iterate_Xn_cp(n_eval-1,
            X_coef_cp,
            Y_coef_cp,
            Z_coef_cp,
            B_denom_coef_c,
            B_alpha_coef,
            kap_p, dl_p, tau_p,
            iota_coef)
        self.noise['filter']['X_coef_cp'].append(Xnm1.noise_filter(filter_mode, filter_arg))
        if filter:
            Xnm1 = Xnm1.filter(filter_mode, filter_arg)
        X_coef_cp.append(Xnm1)

        # Requires:
        # X_{n}, Y_{n-1}, Z_{n-1},
        # B_{\theta n-1}, B_{\psi n-3},
        # \iota_{(n-3)/2 or (n-4)/2}, B_{\alpha (n-1)/2 or (n-2)/2}
        Ynm1 = iterate_Yn_cp(n_eval-1,
            X_coef_cp,
            Y_coef_cp,
            Z_coef_cp,
            B_psi_coef_cp,
            B_theta_coef_cp,
            B_alpha_coef,
            kap_p, dl_p, tau_p,
            iota_coef,
            Y_free_nm1)
        self.noise['filter']['Y_coef_cp'].append(Ynm1.noise_filter(filter_mode, filter_arg))
        if filter:
            Ynm1 = Ynm1.filter(filter_mode, filter_arg)
        Y_coef_cp.append(Ynm1)

        # no need to ignore_mode_0 for chi integral. This is an odd order.
        B_psi_nm2 = iterate_dc_Bpsi_nm2(n_eval,
            X_coef_cp, Y_coef_cp, Z_coef_cp,
            B_theta_coef_cp, B_psi_coef_cp,
            B_alpha_coef, B_denom_coef_c,
            kap_p, dl_p, tau_p,
            iota_coef).integrate_chi()
        self.noise['filter']['B_psi_coef_cp'].append(B_psi_nm2.noise_filter(filter_mode, filter_arg))
        if filter:
            B_psi_nm2 = B_psi_nm2.filter(filter_mode, filter_arg)
        B_psi_coef_cp.append(B_psi_nm2)

        Zn = iterate_Zn_cp(n_eval,
            X_coef_cp, Y_coef_cp, Z_coef_cp,
            B_theta_coef_cp, B_psi_coef_cp,
            B_alpha_coef,
            kap_p, dl_p, tau_p,
            iota_coef)
        self.noise['filter']['Z_coef_cp'].append(Zn.noise_filter(filter_mode, filter_arg))
        if filter:
            Zn = Zn.filter(filter_mode, filter_arg)
        Z_coef_cp.append(Zn)

        Xn = iterate_Xn_cp(n_eval,
            X_coef_cp,
            Y_coef_cp,
            Z_coef_cp,
            B_denom_coef_c,
            B_alpha_coef,
            kap_p, dl_p, tau_p,
            iota_coef)
        self.noise['filter']['X_coef_cp'].append(Xn.noise_filter(filter_mode, filter_arg))
        if filter:
            Xn = Xn.filter(filter_mode, filter_arg)
        X_coef_cp.append(Xn)

        Yn = iterate_Yn_cp(n_eval,
            X_coef_cp,
            Y_coef_cp,
            Z_coef_cp,
            B_psi_coef_cp,
            B_theta_coef_cp,
            B_alpha_coef,
            kap_p, dl_p, tau_p,
            iota_coef,
            Y_free_n)
        self.noise['filter']['Y_coef_cp'].append(Yn.noise_filter(filter_mode, filter_arg))
        if filter:
            Yn = Yn.filter(filter_mode, filter_arg)
        Y_coef_cp.append(Yn)

        iota_nm1b2 = iterate_iota_nm1b2(n_eval,
            X_coef_cp, Y_coef_cp, Z_coef_cp,\
            tau_p, dl_p, kap_p,\
            iota_coef, eta,\
            B_denom_coef_c, B_alpha_coef, B_theta_coef_cp)
        iota_coef.append(iota_nm1b2)

        print("Time elapsed(s):",(time.time() - start_time))
        print('X_coef_cp', X_coef_cp.get_order())
        print('B_psi_coef_cp', B_psi_coef_cp.get_order())
        print('B_theta_coef_cp', B_theta_coef_cp.get_order())


        return(Equilibrium(X_coef_cp,
        Y_coef_cp,
        Z_coef_cp,
        B_psi_coef_cp,
        B_theta_coef_cp,
        B_denom_coef_c,
        B_alpha_coef,
        kap_p, dl_p, tau_p,
        iota_coef, eta,
        0,
        p_perp_coef_cp,
        Delta_coef_cp, self.noise))

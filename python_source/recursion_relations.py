# Wrapped/completed recursion relations based on translated expressions 
# in parsed/. Necessary masking and/or n-substitution are included. All iterate_*
# methods returns ChiPhiFuncGrid's.

# ChiPhiFunc and ChiPhiEpsFunc
from chiphifunc import *
from chiphiepsfunc import *

# parsed relations
import parsed

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
# Yn1c ()
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
    return(ChiPhiFuncGrid.solve_underdet_degen(Y_mode=True,
                                        v_source_A=coef_a,
                                        v_source_B=coef_b,
                                        v_rhs=ynp1_rhsmlhs,
                                        rank_rhs=n_eval, # Yn has n+1 dof. So, RHS must have n dof.
                                        i_free=0, # (dummy)
                                        vai=Yn_free))

# Evaluates iota_{(n-1)/2}.
# Requires:
# \Lambda_n (X_{n-1}, Y_{n-1}, Z_{n}, \iota_{(n-3)/2}),
# B_{\theta n}, B_0, B_{\alpha 0}$
def iterate_iota_nm1b2(n, X_coef_cp, Y_coef_cp, Z_coef_cp,\
               iota_coef,\
               tau_p, dl_p, kap_p,\
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

    # Yn1s_p MUST be evaluated with iota_nb2m1=0!
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
    X_coef_cp,
    Y_coef_cp,
    Z_coef_cp,
    B_theta_coef_cp,
    B_psi_coef_cp,
    B_alpha_coe,
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

''' I.2 Iterate '''
# Evaluates an entire order.

''' II. Magnetic and MHD equations '''

''' II.1 Recursion relations for individual variables '''

''' II.2 Iterate '''
# Evaluates an entire order.

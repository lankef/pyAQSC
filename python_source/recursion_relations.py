# Wrapped/completed recursion relations based on translated expressions
# in parsed/. Necessary masking and/or n-substitution are included. All iterate_*
# methods returns ChiPhiFunc's.

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
        iota_coef=iota_coef).cap_m(n_eval))

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
    return(ChiPhiFunc.solve_underdet_degen(Y_mode=True,
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
    ).cap_m(n_eval))

# Solving for Yn1c for odd orders
def iterate_Yn1c_p(n, X_coef_cp, Y_coef_cp, Z_coef_cp,\
                  iota_coef,\
                  tau_p, dl_p, kap_p, eta,\
                  B_denom_coef_c, B_alpha_coef,
                  B_psi_coef_cp, B_theta_coef_cp, B_theta_np10):

    Yn1s_p = parsed.evaluate_ynp1s1_full(n-1,
    X_coef_cp,
    Y_coef_cp.mask(n-1).zero_append(),
    Z_coef_cp,
    B_psi_coef_cp,
    B_theta_coef_cp,
    B_alpha_coef,
    kap_p, dl_p, tau_p, eta,
        iota_coef).get_constant()
    #
    # print('iterate_Yn1c_p: Yn1s_p')
    # Yn1s_p.display_content()

    Y11s_p, Y11c_p = Y_coef_cp[1].get_Yn1s_Yn1c()
    _, X11c_p = X_coef_cp[1].get_Yn1s_Yn1c()

    # Note the difference with iota_nm1b2: iota(n-1)//2=0 is no longer applied here.
    Xi_n_p_no_iota_mask = parsed.eval_full_Xi_n_p(
        n, X_coef_cp, Y_coef_cp.mask(n-1).zero_append(), Z_coef_cp.mask(n).zero_append(), \
        kap_p, dl_p, tau_p, iota_coef).get_constant()
    Xi_n_p = parsed.eval_full_Xi_n_p(
        n, X_coef_cp, Y_coef_cp.mask(n-1).zero_append(), Z_coef_cp.mask(n).zero_append(), \
        kap_p, dl_p, tau_p, iota_coef.mask((n-1)//2-1).zero_append()).get_constant()

    iota_0 = iota_coef[0]
    B0 = B_denom_coef_c[0]
    B_alpha0 = B_alpha_coef[0]

    sigma_p = Y11c_p/Y11s_p # Definition. On pg 4 below (6)
    Lambda_n_p_eval = Lambda_n_p(Yn1s_p, Y11s_p, Y11c_p, X11c_p, iota_0, tau_p, dl_p, sigma_p, Xi_n_p_no_iota_mask)


    exponent = 2*iota_0*sigma_p.integrate_phi(periodic = False)
    exp_factor = exponent.exp()
    exp_factor_neg = (-exponent).exp()

    # B_0
    B0 = B_denom_coef_c[0]

    # B_{\alpha 0}
    B_alpha0 = B_alpha_coef[0]

    print('avg RHS',np.average((Lambda_n_p_eval+2*B_alpha0*B0*B_theta_np10/(Y11s_p**2)).content))
    print('exp_factor_neg')
    # exp_factor_neg.display_content()
    sigma_tilde_n = exp_factor_neg*((\
            exp_factor\
            *(Lambda_n_p_eval+2*B_alpha0*B0*B_theta_np10/(Y11s_p**2))\
        ).integrate_phi(periodic=False)\
    )

    return(sigma_tilde_n, Lambda_n_p_eval)

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
def iterate_dc_B_psi_nm2(
    n_eval,
    X_coef_cp, Y_coef_cp, Z_coef_cp,
    B_theta_coef_cp, B_psi_coef_cp,
    B_alpha_coef, B_denom_coef_c,
    kap_p, dl_p, tau_p,
    iota_coef):
    dchi_b_psi_nm2 = parsed.eval_dchi_b_psi_nm2.eval_dchi_B_psi_cp_nm2(n_eval, \
        X_coef_cp.mask(n_eval-1).zero_append(), \
        Y_coef_cp.mask(n_eval-1).zero_append(), \
        Z_coef_cp.mask(n_eval-1).zero_append(), \
        B_theta_coef_cp, B_psi_coef_cp.mask(n_eval-3).zero_append(), B_alpha_coef, B_denom_coef_c, \
        kap_p, dl_p, tau_p, iota_coef)
    # The evaluation gives an extra component.
    try:
        return(dchi_b_psi_nm2.cap_m(n_eval-2))
    except TypeError:
        return(dchi_b_psi_nm2)

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
        iota_coef).cap_m(n_eval)
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

    content = solve_dphi_iota_dchi(iota_coef[0], Delta_n_inhomog_component.content)
    return(
        ChiPhiFunc(content).cap_m(n_eval)
    )

''' III. Equilibrium manager and Iterate '''

# A container for all equilibrium quantities.
# All coef inputs must be ChiPhiEpsFunc's.
class Equilibrium:
    def __init__(self, unknown, constant, noise):
        self.noise=noise
        self.unknown=unknown
        self.constant=constant

        # Check if every term is on the same order
        self.check_order_consistency()

        # Some coeffs are really long. We only calc them once.
        self.prepare_constants()

    def from_known(
        X_coef_cp,
        Y_coef_cp,
        Z_coef_cp,
        B_psi_coef_cp,
        B_theta_coef_cp,
        B_denom_coef_c,
        B_alpha_coef,
        kap_p, dl_p, tau_p,
        iota_coef, eta,
        p_perp_coef_cp,
        Delta_coef_cp,
        noise=None
        ):

        # Variables being solved for are stored in dicts for
        # convenience of plotting and saving
        unknown = {}
        constant = {}

        unknown['X_coef_cp'] = X_coef_cp
        unknown['Y_coef_cp'] = Y_coef_cp
        unknown['Z_coef_cp'] = Z_coef_cp
        unknown['B_psi_coef_cp'] = B_psi_coef_cp
        unknown['B_theta_coef_cp'] = B_theta_coef_cp
        unknown['iota_coef'] = iota_coef
        unknown['p_perp_coef_cp'] = p_perp_coef_cp
        unknown['Delta_coef_cp'] = Delta_coef_cp

        constant['B_denom_coef_c'] = B_denom_coef_c
        constant['B_alpha_coef'] = B_alpha_coef
        constant['kap_p'] = kap_p
        constant['dl_p'] = dl_p
        constant['tau_p'] = tau_p
        constant['eta'] = eta

        # Pressure can be trivial
        current_order = X_coef_cp.get_order()
        if not unknown['p_perp_coef_cp']:
            unknown['p_perp_coef_cp'] = ChiPhiEpsFunc.zeros_to_order(current_order)
        if not unknown['Delta_coef_cp']:
            unknown['Delta_coef_cp'] = ChiPhiEpsFunc.zeros_to_order(current_order)

        # Manages noises
        if not noise:
            noise = {} # dict of dict managing types of noises
            # Tracks different types of noise
            noise['filter'] = {}
            for key in noise.keys():
                noise[key]['X_coef_cp'] = ChiPhiEpsFunc.zeros_like(X_coef_cp)
                noise[key]['Y_coef_cp'] = ChiPhiEpsFunc.zeros_like(Y_coef_cp)
                noise[key]['Z_coef_cp'] = ChiPhiEpsFunc.zeros_like(Z_coef_cp)
                noise[key]['B_psi_coef_cp'] = ChiPhiEpsFunc.zeros_like(B_psi_coef_cp)
                noise[key]['B_theta_coef_cp'] = ChiPhiEpsFunc.zeros_like(B_theta_coef_cp)
                noise[key]['p_perp_coef_cp'] = ChiPhiEpsFunc.zeros_like(unknown['p_perp_coef_cp'])
                noise[key]['Delta_coef_cp'] = ChiPhiEpsFunc.zeros_like(unknown['Delta_coef_cp'])

        return(Equilibrium(unknown, constant, noise))

    def save_self(self, file_name):

        # Both unknown and noise are entirely made of ChiPhiFuncs.
        unknown_to_content_list = {}
        for key in self.unknown.keys():
            unknown_to_content_list[key] = self.unknown[key].to_content_list()

        noise_to_content_list = {}
        for keya in self.noise.keys():
            noise_to_content_list[keya]={}
            for keyb in self.noise[keya].keys():
                noise_to_content_list[keya][keyb] = self.noise[keya][keyb].to_content_list()

        const_to_content_list={}
        const_to_content_list['B_denom_coef_c']\
            = self.constant['B_denom_coef_c'].to_content_list()
        const_to_content_list['B_alpha_coef']\
            = self.constant['B_alpha_coef'].to_content_list()
        const_to_content_list['kap_p']\
            = self.constant['kap_p'].content
        const_to_content_list['dl_p']\
            = self.constant['dl_p']
        const_to_content_list['tau_p']\
            = self.constant['tau_p'].content
        const_to_content_list['eta']\
            = self.constant['eta']

        big_dict = {\
            'unknown':unknown_to_content_list,\
            'noise':noise_to_content_list,\
            'constant':const_to_content_list,\
        }
        np.savez(file_name, big_dict)

    def load(filename):
        npzfile = np.load(filename, allow_pickle=True)
        big_dict = npzfile['arr_0'].item()
        raw_unknown = big_dict['unknown']
        raw_constant = big_dict['constant']
        raw_noise = big_dict['noise']

        unknown = {}
        for key in raw_unknown.keys():
            unknown[key] = ChiPhiEpsFunc.from_content_list(raw_unknown[key])

        noise = {}
        for keya in raw_noise.keys():
            noise[keya]={}
            for keyb in raw_noise[keya].keys():
                noise[keya][keyb] \
                    = ChiPhiEpsFunc.from_content_list(raw_noise[keya][keyb])

        constant={}
        constant['B_denom_coef_c']\
            = ChiPhiEpsFunc.from_content_list(raw_constant['B_denom_coef_c'])
        constant['B_alpha_coef']\
            = ChiPhiEpsFunc.from_content_list(raw_constant['B_alpha_coef'])
        constant['kap_p']\
            = ChiPhiFunc(raw_constant['kap_p'])
        constant['dl_p']\
            = raw_constant['dl_p']
        constant['tau_p']\
            = ChiPhiFunc(raw_constant['tau_p'])
        constant['eta']\
            = raw_constant['eta']

        return(Equilibrium(unknown, constant, noise))

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
        if n%2==1:
            raise AttributeError('X_coef_cp has odd order. Equilibrium is managed'\
            ' and updated every 2 orders and stops at even orders')

        check_order_individial(self.unknown['Y_coef_cp'], 'Y_coef_cp', n)
        check_order_individial(self.unknown['Z_coef_cp'], 'Z_coef_cp', n)
        check_order_individial(self.unknown['B_psi_coef_cp'], 'B_psi_coef_cp', n-2)
        check_order_individial(self.unknown['B_theta_coef_cp'], 'B_theta_coef_cp', n)
        check_order_individial(self.constant['B_denom_coef_c'], 'B_denom_coef_c', n)
        check_order_individial(self.constant['B_alpha_coef'], 'B_alpha_coef', n//2)
        check_order_individial(self.unknown['iota_coef'], 'iota_coef', (n-2)//2)
        check_order_individial(self.unknown['p_perp_coef_cp'], 'p_perp_coef_cp', n)
        check_order_individial(self.unknown['Delta_coef_cp'], 'Delta_coef_cp', n)

        for key in self.noise.keys():
            check_order_individial(self.noise[key]['Y_coef_cp'], 'Y_coef_cp', n)
            check_order_individial(self.noise[key]['Z_coef_cp'], 'Z_coef_cp', n)
            check_order_individial(self.noise[key]['B_psi_coef_cp'], 'B_psi_coef_cp', n-2)
            check_order_individial(self.noise[key]['B_theta_coef_cp'], 'B_theta_coef_cp', n)
            check_order_individial(self.noise[key]['p_perp_coef_cp'], 'p_perp_coef_cp', n)
            check_order_individial(self.noise[key]['Delta_coef_cp'], 'Delta_coef_cp', n)

    # The looped equation contains some complicated constants.
    # They are only evaluated once during initialization.
    # Refer to the looped equation Maxima notebook for details.
    def prepare_constants(self):
        return()

    # # Iteration ----------------------------------------------------------------
    # # Evaluates 2 entire orders. Note that no masking is needed for any of the methods
    # # defined in this file. Copies the equilibrium.

    # Iterates the magnetic equations only.
    # Calculates Xn, Yn, Zn, B_psin-2 for 2 orders from lower order values.
    # B_theta, B_psi_nm30, Y_free_nm1 are all free.
    # n_eval must be odd.
    def iterate_2_magnetic_only(self,
        B_theta_nm1, B_theta_n,
        B_psi_nm20,
        B_alpha_nb2,
        B_denom_nm1, B_denom_n,
        Y_free_nm1, Y_free_n,
        p_perp_nm1=0, p_perp_n=0,
        Delta_nm1=0, Delta_n=0,
        n_eval=None, filter=False, filter_mode='low_pass', filter_arg=100):


        # If no order is supplied, then iterate to the next order. the Equilibrium
        # will be edited directly.
        if n_eval == None:
            n_eval = self.get_order() + 2 # getting order and checking consistency
        elif n_eval%2 != 0:
            raise ValueError("n must be even to evaluate iota_{(n-1)/2}")

        start_time = time.time()
        print("Evaluating order",n_eval-1, n_eval)

        # Creating new ChiPhiEpsFunc's for the resulting Equilibrium
        X_coef_cp = self.unknown['X_coef_cp'].mask(n_eval-2)
        Y_coef_cp = self.unknown['Y_coef_cp'].mask(n_eval-2)
        Z_coef_cp = self.unknown['Z_coef_cp'].mask(n_eval-2)
        B_theta_coef_cp = self.unknown['B_theta_coef_cp'].mask(n_eval-2)
        B_psi_coef_cp = self.unknown['B_psi_coef_cp'].mask(n_eval-4)
        iota_coef = self.unknown['iota_coef'].mask((n_eval-4)//2)
        p_perp_coef_cp = self.unknown['p_perp_coef_cp'].mask(n_eval-2)
        Delta_coef_cp = self.unknown['Delta_coef_cp'].mask(n_eval-2)

        self.noise['filter']['X_coef_cp']\
            = self.noise['filter']['X_coef_cp'].mask(n_eval-2)
        self.noise['filter']['Y_coef_cp']\
            = self.noise['filter']['Y_coef_cp'].mask(n_eval-2)
        self.noise['filter']['Z_coef_cp']\
            = self.noise['filter']['Z_coef_cp'].mask(n_eval-2)
        self.noise['filter']['B_theta_coef_cp']\
            = self.noise['filter']['B_theta_coef_cp'].mask(n_eval-2)
        self.noise['filter']['B_psi_coef_cp']\
            = self.noise['filter']['B_psi_coef_cp'].mask(n_eval-4)
        self.noise['filter']['p_perp_coef_cp']\
            = self.noise['filter']['p_perp_coef_cp'].mask(n_eval-2)
        self.noise['filter']['Delta_coef_cp']\
            = self.noise['filter']['Delta_coef_cp'].mask(n_eval-2)
        # Masking all init conds.
        B_denom_coef_c = self.constant['B_denom_coef_c'].mask(n_eval-2)
        B_alpha_coef = self.constant['B_alpha_coef'].mask((n_eval)//2-1)
        kap_p = self.constant['kap_p']
        dl_p = self.constant['dl_p']
        tau_p = self.constant['tau_p']
        eta = self.constant['eta']
        # Appending free functions and initial conditions
        B_theta_coef_cp.append(B_theta_nm1)
        B_theta_coef_cp.append(B_theta_n)
        B_alpha_coef.append(B_alpha_nb2)
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
        B_psi_nm3 = iterate_dc_B_psi_nm2(n_eval-1,
            X_coef_cp, Y_coef_cp, Z_coef_cp,
            B_theta_coef_cp, B_psi_coef_cp,
            B_alpha_coef, B_denom_coef_c,
            kap_p, dl_p, tau_p,
            iota_coef).integrate_chi(ignore_mode_0=True)

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

        iota_nm2b2 = iterate_iota_nm1b2(n_eval-1,
            X_coef_cp, Y_coef_cp, Z_coef_cp,\
            tau_p, dl_p, kap_p,\
            iota_coef, eta,\
            B_denom_coef_c, B_alpha_coef, B_theta_coef_cp)
        iota_coef.append(iota_nm2b2)

        # Order n_eval ---------------------------------------------------
        # no need to ignore_mode_0 for chi integral. This is an odd order.
        B_psi_nm2 = iterate_dc_B_psi_nm2(n_eval,
            X_coef_cp, Y_coef_cp, Z_coef_cp,
            B_theta_coef_cp, B_psi_coef_cp,
            B_alpha_coef, B_denom_coef_c,
            kap_p, dl_p, tau_p,
            iota_coef).integrate_chi(ignore_mode_0 = True)
        B_psi_nm2.content[B_psi_nm2.get_shape()[0]//2] = B_psi_nm20
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

        print("Time elapsed(s):",(time.time() - start_time))


        return(Equilibrium.from_known(X_coef_cp,
        Y_coef_cp,
        Z_coef_cp,
        B_psi_coef_cp,
        B_theta_coef_cp,
        B_denom_coef_c,
        B_alpha_coef,
        kap_p, dl_p, tau_p,
        iota_coef, eta,
        p_perp_coef_cp,
        Delta_coef_cp, self.noise))

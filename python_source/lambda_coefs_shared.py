import numpy as np
from math import floor, ceil
from math_utilities import *
from equilibrium import *
from chiphifunc import *


def eval_lambda_coefs_shared(equilibrium_in):
    # Creating new ChiPhiEpsFunc's for the resulting equilibrium_in
    X_coef_cp = equilibrium_in.unknown['X_coef_cp']
    Y_coef_cp = equilibrium_in.unknown['Y_coef_cp']
    Z_coef_cp = equilibrium_in.unknown['Z_coef_cp']
    B_theta_coef_cp = equilibrium_in.unknown['B_theta_coef_cp']
    B_psi_coef_cp = equilibrium_in.unknown['B_psi_coef_cp']
    iota_coef = equilibrium_in.unknown['iota_coef']
    p_perp_coef_cp = equilibrium_in.unknown['p_perp_coef_cp']
    Delta_coef_cp = equilibrium_in.unknown['Delta_coef_cp']
    B_denom_coef_c = equilibrium_in.constant['B_denom_coef_c']
    B_alpha_coef = equilibrium_in.constant['B_alpha_coef']
    kap_p = equilibrium_in.constant['kap_p']
    dl_p = equilibrium_in.constant['dl_p']
    tau_p = equilibrium_in.constant['tau_p']
    eta = equilibrium_in.constant['eta']

    ''' Coefficient of Y in D3 '''
    coef_Yn1p = (
        -2j*dl_p*tau_p*X_coef_cp[1][-1]
        +1j*Y_coef_cp[1][-1].dphi()
        +2*iota_coef[0]*Y_coef_cp[1][-1]
    )
    coef_dp_Yn1p = (
        -1j*Y_coef_cp[1][-1]
    )
    coef_Yn1n = (
        2j*dl_p*tau_p*X_coef_cp[1][1]
        -1j*Y_coef_cp[1][1].dphi()
        +2*iota_coef[0]*Y_coef_cp[1][1]
    )
    coef_dp_Yn1n = (
        1j*Y_coef_cp[1][1]
    )

    def lambda_coef_Yn1p_in_D3(vector_free_coef, nfp):
        coef_Yn1p_in_Yn1n = ChiPhiFunc(vector_free_coef, nfp)[-1]
        return(coef_Yn1p + coef_Yn1n*coef_Yn1p_in_Yn1n + coef_dp_Yn1n*coef_Yn1p_in_Yn1n.dphi())

    def lambda_coef_dp_Yn1p_in_D3(vector_free_coef, nfp):
        coef_Yn1p_in_Yn1n = ChiPhiFunc(vector_free_coef, nfp)[-1]
        return(coef_dp_Yn1p + coef_dp_Yn1n*coef_Yn1p_in_Yn1n)

    return({
        'lambda_coef_Yn1p_in_D3': lambda_coef_Yn1p_in_D3,
        'lambda_coef_dp_Yn1p_in_D3': lambda_coef_dp_Yn1p_in_D3,
    })

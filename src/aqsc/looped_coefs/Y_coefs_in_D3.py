# from math import floor, ceil
from aqsc.math_utilities import *
from aqsc.chiphifunc import *

def lambda_coef_Yn1p_in_D3(vector_free_coef, X_coef_cp, Y_coef_cp, iota_coef, dl_p, tau_p, nfp:int):
    coef_Yn1p = (
        -2j*dl_p*tau_p*X_coef_cp[1][-1]
        +1j*Y_coef_cp[1][-1].dphi()
        +2*iota_coef[0]*Y_coef_cp[1][-1]
    )
    coef_Yn1p_in_Yn1n = ChiPhiFunc(vector_free_coef, nfp)[-1]
    coef_Yn1n = (
        2j*dl_p*tau_p*X_coef_cp[1][1]
        -1j*Y_coef_cp[1][1].dphi()
        +2*iota_coef[0]*Y_coef_cp[1][1]
    )
    coef_dp_Yn1n = (
        1j*Y_coef_cp[1][1]
    )
    return(coef_Yn1p + coef_Yn1n*coef_Yn1p_in_Yn1n + coef_dp_Yn1n*coef_Yn1p_in_Yn1n.dphi())

def lambda_coef_dp_Yn1p_in_D3(vector_free_coef, X_coef_cp, Y_coef_cp, iota_coef, dl_p, tau_p, nfp:int):
    coef_dp_Yn1p = (
        -1j*Y_coef_cp[1][-1]
    )
    coef_Yn1p_in_Yn1n = ChiPhiFunc(vector_free_coef, nfp)[-1]
    coef_dp_Yn1n = (
        1j*Y_coef_cp[1][1]
    )
    return(coef_dp_Yn1p + coef_dp_Yn1n*coef_Yn1p_in_Yn1n)

def lambda_coef_Yn1c_in_D3(vector_free_coef, X_coef_cp, Y_coef_cp, iota_coef, dl_p, tau_p, nfp:int):
    coef_Yn1c_in_Yn1n = ChiPhiFunc(vector_free_coef, nfp)[-1]
    coef_Yn1c_in_Yn1p = ChiPhiFunc(vector_free_coef, nfp)[1]
    coef_Yn1p = (
        -2j*dl_p*tau_p*X_coef_cp[1][-1]
        +1j*Y_coef_cp[1][-1].dphi()
        +2*iota_coef[0]*Y_coef_cp[1][-1]
    )
    coef_Yn1n = (
        2j*dl_p*tau_p*X_coef_cp[1][1]
        -1j*Y_coef_cp[1][1].dphi()
        +2*iota_coef[0]*Y_coef_cp[1][1]
    )
    coef_dp_Yn1n = (
        1j*Y_coef_cp[1][1]
    )
    coef_dp_Yn1p = (
        -1j*Y_coef_cp[1][-1]
    )
    return( 
        # Coefficient of Yn1c in D3
        coef_Yn1n*coef_Yn1c_in_Yn1n 
        + coef_dp_Yn1n*coef_Yn1c_in_Yn1n.dphi()
        + coef_Yn1p*coef_Yn1c_in_Yn1p
        + coef_dp_Yn1p*coef_Yn1c_in_Yn1p.dphi(),
        # Coefficient of dphi Yn1c in D3
        coef_dp_Yn1n*coef_Yn1c_in_Yn1n
        + coef_dp_Yn1p*coef_Yn1c_in_Yn1p
    )


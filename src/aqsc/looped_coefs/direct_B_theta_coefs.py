from aqsc.math_utilities import *

def eval_B_theta_coefs(p_perp_coef_cp, Delta_coef_cp, B_denom_coef_c, iota_coef):
    coef_B_theta = -(
        2*B_denom_coef_c[0]**2*(diff(p_perp_coef_cp[1],False,1))
        +2*B_denom_coef_c[0]**2*iota_coef[0]*(diff(p_perp_coef_cp[1],True,1))
        +(Delta_coef_cp[0]-2)*iota_coef[0]*(diff(B_denom_coef_c[1],True,1))
        +4*B_denom_coef_c[0]*B_denom_coef_c[1]*(diff(p_perp_coef_cp[0],False,1))
        +2*B_denom_coef_c[1]*(diff(Delta_coef_cp[0],False,1)))/(2)
    coef_dchi_B_theta = B_denom_coef_c[0]*iota_coef[0]*Delta_coef_cp[1]
    coef_dphi_B_theta = B_denom_coef_c[0]*Delta_coef_cp[1]

    return({
        'coef_B_theta':coef_B_theta,
        'coef_dchi_B_theta':coef_dchi_B_theta,
        'coef_dphi_B_theta':coef_dphi_B_theta,
    })

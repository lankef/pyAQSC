# Stores looped equation constants.

# ChiPhiFunc and ChiPhiEpsFunc
from chiphifunc import *
from chiphiepsfunc import *
from math_utilities import *
import numpy as np

# The looped equation coefficients are very long and will be costly to calculate.
# So, we do the bulk of the calculation once when initializing/loading each equilibrium,
# and only caluclate the n dependence during each iteration steps.
def calculate_raw(B_psi_coef_cp, B_theta_coef_cp, \
    Delta_coef_cp, p_perp_coef_cp,\
    X_coef_cp, Y_coef_cp, Z_coef_cp, \
    iota_coef, dl_p,\
    B_denom_coef_c, B_alpha_coef, \
    kap_p, tau_p):

    # Coefficient of Y -------------------------------------------------------------
    # No dchi component should be ignored because ALL Y COMPONENTS contain the constant
    # (chi-indep) component
    Coef_Y_raw = (
        2*(Delta_coef_cp[0]-1)*(diff(X_coef_cp[1],'chi',1))*dl_p*(diff(tau_p,'phi',1))
        +(
            2*(Delta_coef_cp[0]-1)*iota_coef[0]*(diff(X_coef_cp[1],'chi',2))
            +2*(Delta_coef_cp[0]-1)*(diff(X_coef_cp[1],'chi',1,'phi',1))
            +2*(diff(Delta_coef_cp[0],'phi',1))*(diff(X_coef_cp[1],'chi',1))
        )*dl_p*tau_p
        +(1-Delta_coef_cp[0])*iota_coef[0]**2*(diff(Y_coef_cp[1],'chi',3))
        +2*(1-Delta_coef_cp[0])*iota_coef[0]*(diff(Y_coef_cp[1],'chi',2,'phi',1))
        -iota_coef[0]*(diff(Delta_coef_cp[0],'phi',1))*(diff(Y_coef_cp[1],'chi',2))
        +(1-Delta_coef_cp[0])*(diff(Y_coef_cp[1],'chi',1,'phi',2))
        -(diff(Delta_coef_cp[0],'phi',1))*(diff(Y_coef_cp[1],'chi',1,'phi',1))
    )/B_alpha_coef[0]

    Coef_dchi_Y_raw_a = -2*(1-Delta_coef_cp[0])*iota_coef[0]*(diff(X_coef_cp[1],'chi',1))*dl_p*tau_p
    Coef_dchi_Y_raw_b = -(
        (2*Delta_coef_cp[0]-2)*X_coef_cp[1]*dl_p*(diff(tau_p,'phi',1))
        +(
            (2*Delta_coef_cp[0]-2)*(diff(X_coef_cp[1],'phi',1))
            +2*X_coef_cp[1]*(diff(Delta_coef_cp[0],'phi',1))
        )*dl_p*tau_p
        +(1-Delta_coef_cp[0])*(diff(Y_coef_cp[1],'phi',2))
        -(diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'phi',1)
        +(1-Delta_coef_cp[0])*iota_coef[0]**2*(diff(Y_coef_cp[1],'chi',2))
        +(2-2*Delta_coef_cp[0])*iota_coef[0]*(diff(Y_coef_cp[1],'chi',1,'phi',1))
        -9*iota_coef[0]*(diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',1)
    )/B_alpha_coef[0]

    Coef_dphi_Y_raw = (
        2*(Delta_coef_cp[0]-1)*(diff(X_coef_cp[1],'chi',1))*dl_p*tau_p
        +(diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',1)
    )/B_alpha_coef[0]

    Coef_dchi_dphi_Y_raw_a = -(2-2*Delta_coef_cp[0])*iota_coef[0]*diff(Y_coef_cp[1],'chi',1)/B_alpha_coef[0]
    Coef_dchi_dphi_Y_raw_b = -(
        (2*Delta_coef_cp[0]-2)*X_coef_cp[1]*dl_p*tau_p
        +Y_coef_cp[1]*(diff(Delta_coef_cp[0],'phi',1))
    )/B_alpha_coef[0]

    Coef_dphi_dphi_Y_raw = (
        (Delta_coef_cp[0]-1)*diff(Y_coef_cp[1],'chi',1)
    )/B_alpha_coef[0]

    Coef_dchi_dchi_Y_raw_a = -(1-Delta_coef_cp[0])*iota_coef[0]**2*diff(Y_coef_cp[1],'chi',1)/B_alpha_coef[0]
    Coef_dchi_dchi_Y_raw_b = -(
        (2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*dl_p*tau_p
        +iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)
    )/B_alpha_coef[0]

    Coef_dchi_dchi_dphi_Y_raw = -(2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]/B_alpha_coef[0]

    Coef_dphi_dphi_dchi_Y_raw = -(Delta_coef_cp[0]-1)*Y_coef_cp[1]/B_alpha_coef[0]

    Coef_dchi_dchi_dchi_Y_raw = -(Delta_coef_cp[0]-1)*iota_coef[0]**2*Y_coef_cp[1]/B_alpha_coef[0]

    Coef_B_theta = -(
        2*B_denom_coef_c[0]**2*(diff(p_perp_coef_cp[1],'phi',1))
        +2*B_denom_coef_c[0]**2*iota_coef[0]*(diff(p_perp_coef_cp[1],'chi',1))
        +(Delta_coef_cp[0]-2)*iota_coef[0]*(diff(B_denom_coef_c[1],'chi',1))
        +4*B_denom_coef_c[0]*B_denom_coef_c[1]*(diff(p_perp_coef_cp[0],'phi',1))
        +2*B_denom_coef_c[1]*(diff(Delta_coef_cp[0],'phi',1))
    )/2
    Coef_dchi_B_theta = B_denom_coef_c[0]*iota_coef[0]*Delta_coef_cp[1]
    Coef_dphi_B_theta = B_denom_coef_c[0]*Delta_coef_cp[1]

    # Coeff of B_psi in Z and p ----------------------------------------------------
    # Coeff of B_psi in Z
    # (For B_psi 5 the denominator is 7. Here we are looking at B_psi n-3 solved with
    # the order n-1 of other vars at the looped equation's nth order. So, the
    # denominator should be n-1.)
    Coef_B_psi_in_Z_raw = \
    (X_coef_cp[1]*(diff(Y_coef_cp[1],'chi',1))-Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1)))
    # Coeff of B_psi in p
    # (For B_psi 5 the denominator is 7. Here we are looking at B_psi n-3 solved with
    # n-1, so the denominator should be n-1.)
    Coef_B_psi_in_p_raw = \
    -(2*(diff(Delta_coef_cp[0],'phi',1)))/(B_alpha_coef[0]*B_denom_coef_c[0])

    # (ignored)
    # Coef_dchi_B_psi =
    # -((2*Delta_coef_cp[0]-2)*iota_coef[0])/((n-1)*B_alpha_coef[0]*B_denom_coef_c[0])

    Coef_dphi_B_psi_in_p_raw = \
    -(2*Delta_coef_cp[0]-2)/(B_alpha_coef[0]*B_denom_coef_c[0])
    # Note that all coeffs in p are scalar. That means no convolution of B_psi0 has
    # occured, and dchi p can be ignored.

    # Coeff of p -------------------------------------------------------------------
    Coef_p_raw = \
    -(2*B_alpha_coef[0]*B_denom_coef_c[0]*(diff(B_denom_coef_c[1],'chi',1)))

    # Coeff of Delta ---------------------------------------------------------------
    Coef_Delta_raw = -(B_alpha_coef[0]*diff(B_denom_coef_c[1],'chi',1))/2

    # Coeff of B_psi ---------------------------------------------------------------
    Coef_B_psi_raw = \
    -(
        2*B_denom_coef_c[0]*iota_coef[0]*(diff(Delta_coef_cp[1],'chi',2))
        +2*B_denom_coef_c[0]*(diff(Delta_coef_cp[1],'chi',1,'phi',1))
        +(2-2*Delta_coef_cp[0])*iota_coef[0]*(diff(B_denom_coef_c[1],'chi',2))
    )
    Coef_dphi_B_psi_raw = -2*(B_denom_coef_c[0]*(diff(Delta_coef_cp[1],'chi',1)))

    # Coeff of X -------------------------------------------------------------------
    # Again, dchi coeffs are not skipped, because B_psi coeffs in Z has chi
    # dependence.
    Coef_X_raw = \
    -(
        (2*Delta_coef_cp[0]-2)*(diff(Y_coef_cp[1],'chi',1))*dl_p*(diff(tau_p,'phi',1))
        +(
            (2*Delta_coef_cp[0]-2)*iota_coef[0]*(diff(Y_coef_cp[1],'chi',2))
            +(2*Delta_coef_cp[0]-2)*(diff(Y_coef_cp[1],'chi',1,'phi',1))
            +2*(diff(Delta_coef_cp[0],'phi',1))*(diff(Y_coef_cp[1],'chi',1))
        )*dl_p*tau_p
        +(Delta_coef_cp[0]-1)*iota_coef[0]**2*(diff(X_coef_cp[1],'chi',3))
        +(2*Delta_coef_cp[0]-2)*iota_coef[0]*(diff(X_coef_cp[1],'chi',2,'phi',1))
        +iota_coef[0]*(diff(Delta_coef_cp[0],'phi',1))*(diff(X_coef_cp[1],'chi',2))
        +(Delta_coef_cp[0]-1)*(diff(X_coef_cp[1],'chi',1,'phi',2))
        +(diff(Delta_coef_cp[0],'phi',1))*(diff(X_coef_cp[1],'chi',1,'phi',1))
    )/B_alpha_coef[0]

    Coef_dchi_X_raw_a = (2-2*Delta_coef_cp[0])*iota_coef[0]*(diff(Y_coef_cp[1],'chi',1))*dl_p*tau_p/B_alpha_coef[0]
    Coef_dchi_X_raw_b = \
    (
        (2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*dl_p*(diff(tau_p,'phi',1))
        +((2*Delta_coef_cp[0]-2)*(diff(Y_coef_cp[1],'phi',1))
            +2*Y_coef_cp[1]*(diff(Delta_coef_cp[0],'phi',1))
        )*dl_p*tau_p
        +(Delta_coef_cp[0]-1)*(diff(X_coef_cp[1],'phi',2))
        +(diff(Delta_coef_cp[0],'phi',1))*(diff(X_coef_cp[1],'phi',1))
        +(Delta_coef_cp[0]-1)*iota_coef[0]**2*(diff(X_coef_cp[1],'chi',2))
        +(2*Delta_coef_cp[0]-2)*iota_coef[0]*(diff(X_coef_cp[1],'chi',1,'phi',1))
    )/B_alpha_coef[0]
    Coef_dchi_X_raw_c = iota_coef[0]*(diff(Delta_coef_cp[0],'phi',1))*(diff(X_coef_cp[1],'chi',1))/B_alpha_coef[0]

    Coef_dphi_X_raw = \
    -(
        (2*Delta_coef_cp[0]-2)*(diff(Y_coef_cp[1],'chi',1))*dl_p*tau_p
        -(diff(Delta_coef_cp[0],'phi',1))*(diff(X_coef_cp[1],'chi',1))
    )/B_alpha_coef[0]

    Coef_dchi_dphi_X_raw_a = (2*Delta_coef_cp[0]-2)*iota_coef[0]*(diff(X_coef_cp[1],'chi',1))/B_alpha_coef[0]
    Coef_dchi_dphi_X_raw_b = \
    (
        (2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*dl_p*tau_p
        -X_coef_cp[1]*(diff(Delta_coef_cp[0],'phi',1))
    )/B_alpha_coef[0]

    Coef_dphi_dphi_X_raw = \
    (Delta_coef_cp[0]-1)*diff(X_coef_cp[1],'chi',1)/B_alpha_coef[0]


    Coef_dchi_dchi_X_raw_a = (Delta_coef_cp[0]-1)*iota_coef[0]**2*diff(X_coef_cp[1],'chi',1)/B_alpha_coef[0]
    Coef_dchi_dchi_X_raw_b = \
    (
        (2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]*dl_p*tau_p
        -iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)
    )/B_alpha_coef[0]


    Coef_dchi_dchi_dphi_X_raw = \
    -((2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1])/B_alpha_coef[0]

    Coef_dphi_dphi_dchi_X_raw = \
    -((Delta_coef_cp[0]-1)*X_coef_cp[1])/B_alpha_coef[0]

    Coef_dchi_dchi_dchi_X_raw = \
    -((Delta_coef_cp[0]-1)*iota_coef[0]**2*X_coef_cp[1])/B_alpha_coef[0]

    # Coeff of Z -------------------------------------------------------------------
    Coef_Z_raw = \
    (
        (2*Delta_coef_cp[0]-2)*(diff(X_coef_cp[1],'chi',1))*dl_p*(diff(kap_p,'phi',1))
        +(
            (2*Delta_coef_cp[0]-2)*iota_coef[0]*(diff(X_coef_cp[1],'chi',2))
            +(2*Delta_coef_cp[0]-2)*(diff(X_coef_cp[1],'chi',1,'phi',1))
            +2*(diff(Delta_coef_cp[0],'phi',1))*(diff(X_coef_cp[1],'chi',1))
        )*dl_p*kap_p
    )/B_alpha_coef[0]

    Coef_dchi_Z_raw_a = \
    -(2-2*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],'chi',1)*dl_p*kap_p/B_alpha_coef[0]
    Coef_dchi_Z_raw_b = \
    -(
        (2*Delta_coef_cp[0]-2)*X_coef_cp[1]*dl_p*(diff(kap_p,'phi',1))
        +(
            (2*Delta_coef_cp[0]-2)*(diff(X_coef_cp[1],'phi',1))
            +2*X_coef_cp[1]*(diff(Delta_coef_cp[0],'phi',1))
        )*dl_p*kap_p
    )/B_alpha_coef[0]

    Coef_dphi_Z_raw = \
    (2*Delta_coef_cp[0]-2)*diff(X_coef_cp[1],'chi',1)*dl_p*kap_p/B_alpha_coef[0]

    Coef_dchi_dphi_Z_raw = \
    -((2*Delta_coef_cp[0]-2)*X_coef_cp[1]*dl_p*kap_p)/B_alpha_coef[0]

    Coef_dchi_dchi_Z_raw = \
    -((2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*dl_p*kap_p)/B_alpha_coef[0]

    return({
        'Coef_Y_raw':Coef_Y_raw,
        'Coef_dchi_Y_raw_a':Coef_dchi_Y_raw_a,
        'Coef_dchi_Y_raw_b':Coef_dchi_Y_raw_b,
        'Coef_dphi_Y_raw':Coef_dphi_Y_raw,
        'Coef_dchi_dphi_Y_raw_a':Coef_dchi_dphi_Y_raw_a,
        'Coef_dchi_dphi_Y_raw_b':Coef_dchi_dphi_Y_raw_b,
        'Coef_dphi_dphi_Y_raw':Coef_dphi_dphi_Y_raw,
        'Coef_dchi_dchi_Y_raw_a':Coef_dchi_dchi_Y_raw_a,
        'Coef_dchi_dchi_Y_raw_b':Coef_dchi_dchi_Y_raw_b,
        'Coef_dchi_dchi_dphi_Y_raw':Coef_dchi_dchi_dphi_Y_raw,
        'Coef_dphi_dphi_dchi_Y_raw':Coef_dphi_dphi_dchi_Y_raw,
        'Coef_dchi_dchi_dchi_Y_raw':Coef_dchi_dchi_dchi_Y_raw,
        'Coef_B_theta':Coef_B_theta,
        'Coef_dchi_B_theta':Coef_dchi_B_theta,
        'Coef_dphi_B_theta':Coef_dphi_B_theta,
        'Coef_B_psi_in_Z_raw':Coef_B_psi_in_Z_raw,
        'Coef_B_psi_in_p_raw':Coef_B_psi_in_p_raw,
        'Coef_dphi_B_psi_in_p_raw':Coef_dphi_B_psi_in_p_raw,
        'Coef_p_raw':Coef_p_raw,
        'Coef_Delta_raw':Coef_Delta_raw,
        'Coef_B_psi_raw':Coef_B_psi_raw,
        'Coef_dphi_B_psi_raw':Coef_dphi_B_psi_raw,
        'Coef_X_raw':Coef_X_raw,
        'Coef_dchi_X_raw_a':Coef_dchi_X_raw_a,
        'Coef_dchi_X_raw_b':Coef_dchi_X_raw_b,
        'Coef_dchi_X_raw_c':Coef_dchi_X_raw_c,
        'Coef_dphi_X_raw':Coef_dphi_X_raw,
        'Coef_dchi_dphi_X_raw_a':Coef_dchi_dphi_X_raw_a,
        'Coef_dchi_dphi_X_raw_b':Coef_dchi_dphi_X_raw_b,
        'Coef_dphi_dphi_X_raw':Coef_dphi_dphi_X_raw,
        'Coef_dchi_dchi_X_raw_a':Coef_dchi_dchi_X_raw_a,
        'Coef_dchi_dchi_X_raw_b':Coef_dchi_dchi_X_raw_b,
        'Coef_dchi_dchi_dphi_X_raw':Coef_dchi_dchi_dphi_X_raw,
        'Coef_dphi_dphi_dchi_X_raw':Coef_dphi_dphi_dchi_X_raw,
        'Coef_dchi_dchi_dchi_X_raw':Coef_dchi_dchi_dchi_X_raw,
        'Coef_Z_raw':Coef_Z_raw,
        'Coef_dchi_Z_raw_a':Coef_dchi_Z_raw_a,
        'Coef_dchi_Z_raw_b':Coef_dchi_Z_raw_b,
        'Coef_dphi_Z_raw':Coef_dphi_Z_raw,
        'Coef_dchi_dphi_Z_raw':Coef_dchi_dphi_Z_raw,
        'Coef_dchi_dchi_Z_raw':Coef_dchi_dchi_Z_raw
    })







def calculate_loop_coefficients(
    n, B_alpha_coef, B_denom_coef_c, iota_coef, dl_p, kap_p,
    raws
):
    Coef_Y_raw = raws['Coef_Y_raw']
    Coef_dchi_Y_raw_a = raws['Coef_dchi_Y_raw_a']
    Coef_dchi_Y_raw_b = raws['Coef_dchi_Y_raw_b']
    Coef_dphi_Y_raw = raws['Coef_dphi_Y_raw']
    Coef_dchi_dphi_Y_raw_a = raws['Coef_dchi_dphi_Y_raw_a']
    Coef_dchi_dphi_Y_raw_b = raws['Coef_dchi_dphi_Y_raw_b']
    Coef_dphi_dphi_Y_raw = raws['Coef_dphi_dphi_Y_raw']
    Coef_dchi_dchi_Y_raw_a = raws['Coef_dchi_dchi_Y_raw_a']
    Coef_dchi_dchi_Y_raw_b = raws['Coef_dchi_dchi_Y_raw_b']
    Coef_dchi_dchi_dphi_Y_raw = raws['Coef_dchi_dchi_dphi_Y_raw']
    Coef_dphi_dphi_dchi_Y_raw = raws['Coef_dphi_dphi_dchi_Y_raw']
    Coef_dchi_dchi_dchi_Y_raw = raws['Coef_dchi_dchi_dchi_Y_raw']
    Coef_B_theta = raws['Coef_B_theta']
    Coef_dchi_B_theta = raws['Coef_dchi_B_theta']
    Coef_dphi_B_theta = raws['Coef_dphi_B_theta']
    Coef_B_psi_in_Z_raw = raws['Coef_B_psi_in_Z_raw']
    Coef_B_psi_in_p_raw = raws['Coef_B_psi_in_p_raw']
    Coef_dphi_B_psi_in_p_raw = raws['Coef_dphi_B_psi_in_p_raw']
    Coef_p_raw = raws['Coef_p_raw']
    Coef_Delta_raw = raws['Coef_Delta_raw']
    Coef_B_psi_raw = raws['Coef_B_psi_raw']
    Coef_dphi_B_psi_raw = raws['Coef_dphi_B_psi_raw']
    Coef_X_raw = raws['Coef_X_raw']
    Coef_dchi_X_raw_a = raws['Coef_dchi_X_raw_a']
    Coef_dchi_X_raw_b = raws['Coef_dchi_X_raw_b']
    Coef_dchi_X_raw_c = raws['Coef_dchi_X_raw_c']
    Coef_dphi_X_raw = raws['Coef_dphi_X_raw']
    Coef_dchi_dphi_X_raw_a = raws['Coef_dchi_dphi_X_raw_a']
    Coef_dchi_dphi_X_raw_b = raws['Coef_dchi_dphi_X_raw_b']
    Coef_dphi_dphi_X_raw = raws['Coef_dphi_dphi_X_raw']
    Coef_dchi_dchi_X_raw_a = raws['Coef_dchi_dchi_X_raw_a']
    Coef_dchi_dchi_X_raw_b = raws['Coef_dchi_dchi_X_raw_b']
    Coef_dchi_dchi_dphi_X_raw = raws['Coef_dchi_dchi_dphi_X_raw']
    Coef_dphi_dphi_dchi_X_raw = raws['Coef_dphi_dphi_dchi_X_raw']
    Coef_dchi_dchi_dchi_X_raw = raws['Coef_dchi_dchi_dchi_X_raw']
    Coef_Z_raw = raws['Coef_Z_raw']
    Coef_dchi_Z_raw_a = raws['Coef_dchi_Z_raw_a']
    Coef_dchi_Z_raw_b = raws['Coef_dchi_Z_raw_b']
    Coef_dphi_Z_raw = raws['Coef_dphi_Z_raw']
    Coef_dchi_dphi_Z_raw = raws['Coef_dchi_dphi_Z_raw']
    Coef_dchi_dchi_Z_raw = raws['Coef_dchi_dchi_Z_raw']
    # Coeff of Y -------------------------------------------------------------------
    Coef_Y = Coef_Y_raw*(n-1)/n
    Coef_dchi_Y = Coef_dchi_Y_raw_a/B_alpha_coef[0]*(n-2)/n + Coef_dchi_Y_raw_b/n
    Coef_dphi_Y = Coef_dphi_Y_raw*(n-1)/n
    Coef_dchi_dphi_Y = Coef_dchi_dphi_Y_raw_a*(n-1)/n + Coef_dchi_dphi_Y_raw_b/n
    Coef_dphi_dphi_Y = Coef_dphi_dphi_Y_raw*(n-1)/n
    Coef_dchi_dchi_Y = Coef_dchi_dchi_Y_raw_a*(n-1)/n + Coef_dchi_dchi_Y_raw_b/n
    Coef_dchi_dchi_dphi_Y = Coef_dchi_dchi_dphi_Y_raw/n
    Coef_dphi_dphi_dchi_Y = Coef_dphi_dphi_dchi_Y_raw/n
    Coef_dchi_dchi_dchi_Y = Coef_dchi_dchi_dchi_Y_raw/n

    # Coeff of B_psi in Z and p ----------------------------------------------------
    Coef_B_psi_in_Z = Coef_B_psi_in_Z_raw/(n-1)
    Coef_B_psi_in_p = Coef_B_psi_in_p_raw/(n-1)
    Coef_dphi_B_psi_in_p = Coef_dphi_B_psi_in_p_raw/(n-1)

    # Coeff of p in Delta and Z in X -----------------------------------------------
    # Coeff of p in Delta
    # Coef_dchi_p_in_Delta = -B_denom_coef_c[0]*iota_coef[0]
    # (ignored because B_psi's coefficients in p are all constants,
    # meaning that dchi p has no B_psi_n0 dependence)
    Coef_dphi_p_in_Delta = -B_denom_coef_c[0]

    # Coeff of Z in X
    Coef_dchi_Z_in_X = iota_coef[0]/(dl_p*kap_p)
    Coef_dphi_Z_in_X = 1/(dl_p*kap_p)

    # Coeff of p -------------------------------------------------------------------
    Coef_p = Coef_p_raw*(n-1)/n

    # Coeff of Delta ---------------------------------------------------------------
    Coef_Delta = Coef_Delta_raw*(n-1)/n

    # Coeff of B_psi ---------------------------------------------------------------
    Coef_B_psi = Coef_B_psi_raw/n
    Coef_dphi_B_psi = Coef_dphi_B_psi_raw/n

    # Coeff of X -------------------------------------------------------------------
    Coef_X = Coef_X_raw*(n-1)/n
    Coef_dchi_X = Coef_dchi_X_raw_a*(n-2)/n + Coef_dchi_X_raw_b/n + Coef_dchi_X_raw_c
    Coef_dphi_X = Coef_dphi_X_raw*(n-1)/n
    Coef_dchi_dphi_X = Coef_dchi_dphi_X_raw_a*(n-1)/n + Coef_dchi_dphi_X_raw_b/n
    Coef_dphi_dphi_X = Coef_dphi_dphi_X_raw*(n-1)/n
    Coef_dchi_dchi_X = Coef_dchi_dchi_X_raw_a*(n-1)/n + Coef_dchi_dchi_X_raw_b/n
    Coef_dchi_dchi_dphi_X = Coef_dchi_dchi_dphi_X_raw/n
    Coef_dphi_dphi_dchi_X = Coef_dphi_dphi_dchi_X_raw/n
    Coef_dchi_dchi_dchi_X = Coef_dchi_dchi_dchi_X_raw/n

    # Coeff of Z -------------------------------------------------------------------
    Coef_Z = Coef_Z_raw*(n-1)/n
    Coef_dchi_Z = Coef_dchi_Z_raw_a*(n-2)/n + Coef_dchi_Z_raw_b/n
    Coef_dphi_Z = Coef_dphi_Z_raw*(n-1)/n
    Coef_dchi_dphi_Z = Coef_dchi_dphi_Z_raw/n
    Coef_dchi_dchi_Z = Coef_dchi_dchi_Z_raw/n

    return({
        'Coef_Y': Coef_Y,
        'Coef_dchi_Y': Coef_dchi_Y,
        'Coef_dphi_Y': Coef_dphi_Y,
        'Coef_dchi_dphi_Y': Coef_dchi_dphi_Y,
        'Coef_dphi_dphi_Y': Coef_dphi_dphi_Y,
        'Coef_dchi_dchi_Y': Coef_dchi_dchi_Y,
        'Coef_dchi_dchi_dphi_Y': Coef_dchi_dchi_dphi_Y,
        'Coef_dphi_dphi_dchi_Y': Coef_dphi_dphi_dchi_Y,
        'Coef_dchi_dchi_dchi_Y': Coef_dchi_dchi_dchi_Y,
        'Coef_B_theta': Coef_B_theta,
        'Coef_dchi_B_theta': Coef_dchi_B_theta,
        'Coef_dphi_B_theta': Coef_dphi_B_theta,
        'Coef_B_psi_in_Z': Coef_B_psi_in_Z,
        'Coef_B_psi_in_p': Coef_B_psi_in_p,
        'Coef_dphi_B_psi_in_p': Coef_dphi_B_psi_in_p,
        'Coef_dphi_p_in_Delta': Coef_dphi_p_in_Delta,
        'Coef_dchi_Z_in_X': Coef_dchi_Z_in_X,
        'Coef_dphi_Z_in_X': Coef_dphi_Z_in_X,
        'Coef_p': Coef_p,
        'Coef_Delta': Coef_Delta,
        'Coef_B_psi': Coef_B_psi,
        'Coef_dphi_B_psi': Coef_dphi_B_psi,
        'Coef_X': Coef_X,
        'Coef_dchi_X': Coef_dchi_X,
        'Coef_dphi_X': Coef_dphi_X,
        'Coef_dchi_dphi_X': Coef_dchi_dphi_X,
        'Coef_dphi_dphi_X': Coef_dphi_dphi_X,
        'Coef_dchi_dchi_X': Coef_dchi_dchi_X,
        'Coef_dchi_dchi_dphi_X': Coef_dchi_dchi_dphi_X,
        'Coef_dphi_dphi_dchi_X': Coef_dphi_dphi_dchi_X,
        'Coef_dchi_dchi_dchi_X': Coef_dchi_dchi_dchi_X,
        'Coef_Z': Coef_Z,
        'Coef_dchi_Z': Coef_dchi_Z,
        'Coef_dphi_Z': Coef_dphi_Z,
        'Coef_dchi_dphi_Z': Coef_dchi_dphi_Z,
        'Coef_dchi_dchi_Z': Coef_dchi_dchi_Z
    })

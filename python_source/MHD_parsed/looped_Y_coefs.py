from math import floor, ceil
from math_utilities import *
import chiphifunc

def eval_y_coefs(n_eval, X_coef_cp, Y_coef_cp,
    Delta_coef_cp, B_alpha_coef, dl_p, tau_p, iota_coef):
    coef_Y = (
        2*(Delta_coef_cp[0]-1)*(diff(X_coef_cp[1],'chi',1))*dl_p*(diff(tau_p,'phi',1))
        +(
            2*(Delta_coef_cp[0]-1)*iota_coef[0]*(diff(X_coef_cp[1],'chi',2))
            +2*((Delta_coef_cp[0]-1)*diff(X_coef_cp[1],'chi',1)).dphi()
        )*dl_p*tau_p
        +(1-Delta_coef_cp[0])*iota_coef[0]**2*(diff(Y_coef_cp[1],'chi',3))
        +2*(1-Delta_coef_cp[0])*iota_coef[0]*(diff(Y_coef_cp[1],'chi',2,'phi',1))
        -iota_coef[0]*(diff(Delta_coef_cp[0],'phi',1))*(diff(Y_coef_cp[1],'chi',2))
        +(1-Delta_coef_cp[0])*(diff(Y_coef_cp[1],'chi',1,'phi',2))
        -(diff(Delta_coef_cp[0],'phi',1))*(diff(Y_coef_cp[1],'chi',1,'phi',1))
    )/B_alpha_coef[0]*(n_eval-1)/n_eval

    coef_dchi_Y = -(
        (2*Delta_coef_cp[0]-2)*X_coef_cp[1]*dl_p*(diff(tau_p,'phi',1))
        +(
            (2*Delta_coef_cp[0]-2)*(diff(X_coef_cp[1],'phi',1))
            +2*(n_eval-2)*(1-Delta_coef_cp[0])*iota_coef[0]*(diff(X_coef_cp[1],'chi',1))
            +2*X_coef_cp[1]*(diff(Delta_coef_cp[0],'phi',1))
        )*dl_p*tau_p
        +(1-Delta_coef_cp[0])*(diff(Y_coef_cp[1],'phi',2))
        -(diff(Delta_coef_cp[0],'phi',1))*(diff(Y_coef_cp[1],'phi',1))
        +(1-Delta_coef_cp[0])*iota_coef[0]**2*(diff(Y_coef_cp[1],'chi',2))
        +(2-2*Delta_coef_cp[0])*iota_coef[0]*(diff(Y_coef_cp[1],'chi',1,'phi',1))
        -n_eval*iota_coef[0]*(diff(Delta_coef_cp[0],'phi',1))*(diff(Y_coef_cp[1],'chi',1))
    )/B_alpha_coef[0]/n_eval

    coef_dphi_Y = (
        2*(Delta_coef_cp[0]-1)*(diff(X_coef_cp[1],'chi',1))*dl_p*tau_p
        +(diff(Delta_coef_cp[0],'phi',1))*(diff(Y_coef_cp[1],'chi',1))
    )/B_alpha_coef[0]*(n_eval-1)/n_eval

    coef_dchi_dphi_Y = -(
        (2*Delta_coef_cp[0]-2)*X_coef_cp[1]*dl_p*tau_p
        +(n_eval-1)*(2-2*Delta_coef_cp[0])*iota_coef[0]*(diff(Y_coef_cp[1],'chi',1))
        +Y_coef_cp[1]*(diff(Delta_coef_cp[0],'phi',1))
    )/B_alpha_coef[0]/n_eval

    coef_dphi_dphi_Y = (
        (Delta_coef_cp[0]-1)*diff(Y_coef_cp[1],'chi',1)
    )/B_alpha_coef[0]*(n_eval-1)/n_eval

    coef_dchi_dchi_Y = -(
        (2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*dl_p*tau_p
        +(n_eval-1)*(1-Delta_coef_cp[0])*iota_coef[0]**2*(diff(Y_coef_cp[1],'chi',1))
        +iota_coef[0]*Y_coef_cp[1]*(diff(Delta_coef_cp[0],'phi',1))
    )/B_alpha_coef[0]/n_eval

    coef_dchi_dchi_dphi_Y = -(2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]/B_alpha_coef[0]/n_eval

    coef_dphi_dphi_dchi_Y = -(Delta_coef_cp[0]-1)*Y_coef_cp[1]/B_alpha_coef[0]/n_eval

    coef_dchi_dchi_dchi_Y = -(Delta_coef_cp[0]-1)*iota_coef[0]**2*Y_coef_cp[1]/B_alpha_coef[0]/n_eval

    return({
        'coef_Y':coef_Y.cap_m(1),
        'coef_dchi_Y':coef_dchi_Y.cap_m(1),
        'coef_dphi_Y':coef_dphi_Y.cap_m(1),
        'coef_dchi_dphi_Y':coef_dchi_dphi_Y.cap_m(1),
        'coef_dphi_dphi_Y':coef_dphi_dphi_Y.cap_m(1),
        'coef_dchi_dchi_Y':coef_dchi_dchi_Y.cap_m(1),
        'coef_dchi_dchi_dphi_Y':coef_dchi_dchi_dphi_Y.cap_m(1),
        'coef_dphi_dphi_dchi_Y':coef_dphi_dphi_dchi_Y.cap_m(1),
        'coef_dchi_dchi_dchi_Y':coef_dchi_dchi_dchi_Y.cap_m(1),
    })

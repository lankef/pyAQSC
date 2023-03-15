# Evaluates coefficient of diff(B_psi_dummy,phi,j)
from math import floor, ceil
from math_utilities import *
import chiphifunc

def eval_B_psi_lambdas_full(X_coef_cp, Y_coef_cp,
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,
    dl_p, tau_p, kap_p, iota_coef,
    n_jobs = 3,
    backend='threading'):

    ''' Denominators '''
    denom_a = lambda n_eval : B_alpha_coef[0]*n_eval*(n_eval-1)
    denom_b = lambda n_eval : B_alpha_coef[0]*dl_p*kap_p**2*n_eval*(n_eval-1)
    denom_c = lambda n_eval : B_alpha_coef[0]*dl_p*kap_p*n_eval*(n_eval-1)
    denom_d = lambda n_eval : 2*dl_p*kap_p*(n_eval-1)
    denom_e = lambda n_eval : B_alpha_coef[0]*dl_p*kap_p**3*n_eval*(n_eval-1)

    # --------------------------------------------------------------------------
    coef_B_psi_dphi_0_dchi_0_all_but_Y_const_1 = (
        (2*Delta_coef_cp[0]-2)*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'phi',1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',2)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1,'phi',1)+(2-2*Delta_coef_cp[0])*diff(X_coef_cp[1],'phi',1)*(diff(Y_coef_cp[1],'chi',1))**2+((2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1))*diff(Y_coef_cp[1],'chi',1)
    )
    coef_B_psi_dphi_0_dchi_0_all_but_Y_const_2 = (
        ((2-2*Delta_coef_cp[0])*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'phi',1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',2)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1,'phi',1)+(2*Delta_coef_cp[0]-2)*diff(X_coef_cp[1],'phi',1)*(diff(Y_coef_cp[1],'chi',1))**2+((2-2*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1))*diff(Y_coef_cp[1],'chi',1))*dl_p*kap_p*diff(kap_p,'phi',1)+((2*Delta_coef_cp[0]-2)*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'phi',2)+((2*Delta_coef_cp[0]-2)*iota_coef[0]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',2)+(2*Delta_coef_cp[0]-2)*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1,'phi',1)+((4*Delta_coef_cp[0]-4)*iota_coef[0]*diff(X_coef_cp[1],'chi',2)+(4*Delta_coef_cp[0]-4)*diff(X_coef_cp[1],'chi',1,'phi',1)+2*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'phi',1)+(2-2*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',3)+(4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',2,'phi',1)+(2-2*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*(diff(Y_coef_cp[1],'chi',2))**2+((4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],'chi',1,'phi',1)+((6-6*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],'phi',1)+(2-2*Delta_coef_cp[0])*iota_coef[0]**2*diff(X_coef_cp[1],'chi',1)-2*iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1))*diff(Y_coef_cp[1],'chi',2)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1,'phi',2)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*(diff(Y_coef_cp[1],'chi',1,'phi',1))**2+(((6-6*Delta_coef_cp[0])*diff(X_coef_cp[1],'phi',1)-2*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1))*diff(Y_coef_cp[1],'chi',1,'phi',1)+((2-2*Delta_coef_cp[0])*diff(X_coef_cp[1],'phi',2)-2*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'phi',1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]**2*diff(X_coef_cp[1],'chi',2))*(diff(Y_coef_cp[1],'chi',1))**2+((2*Delta_coef_cp[0]-2)*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',3)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2,'phi',1)+2*iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',2)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',2)+2*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1,'phi',1))*diff(Y_coef_cp[1],'chi',1))*dl_p*kap_p**2
    )
    coef_B_psi_dphi_0_dchi_0_all_but_Y_const_3 = (
        ((Delta_coef_cp[0]-1)*(diff(X_coef_cp[1],'chi',1))**2*diff(Y_coef_cp[1],'phi',1)+(1-Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',2)+(1-Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1,'phi',1)+(1-Delta_coef_cp[0])*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)*diff(Y_coef_cp[1],'chi',1)+(Delta_coef_cp[0]-1)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',2)+(Delta_coef_cp[0]-1)*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',1))*kap_p*diff(kap_p,'phi',2)+((2-2*Delta_coef_cp[0])*(diff(X_coef_cp[1],'chi',1))**2*diff(Y_coef_cp[1],'phi',1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',2)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1,'phi',1)+(2*Delta_coef_cp[0]-2)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)*diff(Y_coef_cp[1],'chi',1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',2)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',1))*(diff(kap_p,'phi',1))**2+((2*Delta_coef_cp[0]-2)*(diff(X_coef_cp[1],'chi',1))**2*diff(Y_coef_cp[1],'phi',2)+((4*Delta_coef_cp[0]-4)*iota_coef[0]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',2)+(4*Delta_coef_cp[0]-4)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',1)+diff(Delta_coef_cp[0],'phi',1)*(diff(X_coef_cp[1],'chi',1))**2)*diff(Y_coef_cp[1],'phi',1)+(2-2*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',3)+(4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',2,'phi',1)+((4-4*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)+(2-2*Delta_coef_cp[0])*iota_coef[0]**2*(diff(X_coef_cp[1],'chi',1))**2-iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',2)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1,'phi',2)+((4-4*Delta_coef_cp[0])*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)-X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1,'phi',1)+((2-2*Delta_coef_cp[0])*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',2)-diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]**2*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',2))*diff(Y_coef_cp[1],'chi',1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',3)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',2,'phi',1)+iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',2)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',2)+Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',1))*kap_p*diff(kap_p,'phi',1)+((1-Delta_coef_cp[0])*(diff(X_coef_cp[1],'chi',1))**2*diff(Y_coef_cp[1],'phi',3)+((3-3*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',2)+(3-3*Delta_coef_cp[0])*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',1)-diff(Delta_coef_cp[0],'phi',1)*(diff(X_coef_cp[1],'chi',1))**2)*diff(Y_coef_cp[1],'phi',2)+((2-2*Delta_coef_cp[0])*iota_coef[0]**2*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',3)+(4-4*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',2,'phi',1)-iota_coef[0]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',2)+(2-2*Delta_coef_cp[0])*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',2)-diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',1))*diff(Y_coef_cp[1],'phi',1)+(Delta_coef_cp[0]-1)*iota_coef[0]**3*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',4)+(3*Delta_coef_cp[0]-3)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',3,'phi',1)+((3*Delta_coef_cp[0]-3)*iota_coef[0]**2*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]**3*(diff(X_coef_cp[1],'chi',1))**2+iota_coef[0]**2*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',3)+(3*Delta_coef_cp[0]-3)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',2,'phi',2)+((6*Delta_coef_cp[0]-6)*iota_coef[0]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)+(3*Delta_coef_cp[0]-3)*iota_coef[0]**2*(diff(X_coef_cp[1],'chi',1))**2+2*iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',2,'phi',1)+((3*Delta_coef_cp[0]-3)*iota_coef[0]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',2)+2*iota_coef[0]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)+(1-Delta_coef_cp[0])*iota_coef[0]**3*X_coef_cp[1]*diff(X_coef_cp[1],'chi',3)+(2-2*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2,'phi',1)-iota_coef[0]**2*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',2)+(1-Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',2)+((3*Delta_coef_cp[0]-3)*iota_coef[0]**2*diff(X_coef_cp[1],'chi',1)-iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',1,'phi',1)+iota_coef[0]**2*diff(Delta_coef_cp[0],'phi',1)*(diff(X_coef_cp[1],'chi',1))**2)*diff(Y_coef_cp[1],'chi',2)+(Delta_coef_cp[0]-1)*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1,'phi',3)+((3*Delta_coef_cp[0]-3)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)+X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1,'phi',2)+((3*Delta_coef_cp[0]-3)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',2)+2*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)+(1-Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',3)+(2-2*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2,'phi',1)+((3-3*Delta_coef_cp[0])*iota_coef[0]**2*diff(X_coef_cp[1],'chi',1)-iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',2)+(1-Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',2)-X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1,'phi',1))*diff(Y_coef_cp[1],'chi',1,'phi',1)+((Delta_coef_cp[0]-1)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',3)+diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',2)+((1-Delta_coef_cp[0])*iota_coef[0]**2*diff(X_coef_cp[1],'chi',3)+(2-2*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],'chi',2,'phi',1)-iota_coef[0]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',2)+(1-Delta_coef_cp[0])*diff(X_coef_cp[1],'chi',1,'phi',2)-diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1,'phi',1))*diff(X_coef_cp[1],'phi',1)+(2-2*Delta_coef_cp[0])*iota_coef[0]**3*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',3)+(3-3*Delta_coef_cp[0])*iota_coef[0]**2*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',2,'phi',1)-iota_coef[0]**2*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',2))*diff(Y_coef_cp[1],'chi',1)+(1-Delta_coef_cp[0])*iota_coef[0]**3*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',4)+(3-3*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',3,'phi',1)+((Delta_coef_cp[0]-1)*iota_coef[0]**3*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(Delta_coef_cp[0]-1)*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1)-iota_coef[0]**2*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(X_coef_cp[1],'chi',3)+(3-3*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',2,'phi',2)+((2*Delta_coef_cp[0]-2)*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1)-2*iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(X_coef_cp[1],'chi',2,'phi',1)+iota_coef[0]**2*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*(diff(X_coef_cp[1],'chi',2))**2+((Delta_coef_cp[0]-1)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',2)+2*iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1,'phi',1))*diff(X_coef_cp[1],'chi',2)+(1-Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',3)+((Delta_coef_cp[0]-1)*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1)-Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(X_coef_cp[1],'chi',1,'phi',2)+Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*(diff(X_coef_cp[1],'chi',1,'phi',1))**2)*kap_p**2
    )
    coef_B_psi_dphi_0_dchi_0_all_but_Y_const_4 = (
        ((2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1))**2)*dl_p*diff(kap_p,'phi',1)+((2-2*Delta_coef_cp[0])*(diff(X_coef_cp[1],'chi',1))**2*diff(Y_coef_cp[1],'phi',1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',2)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1,'phi',1)+((2*Delta_coef_cp[0]-2)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1)+2*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1)+(4-4*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',2)+(4-4*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',1)-2*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*(diff(X_coef_cp[1],'chi',1))**2)*dl_p*kap_p
    )*dl_p*kap_p**3
    coef_B_psi_dphi_0_dchi_0_all_but_Y_const_5 = (
        (((2*Delta_coef_cp[0]-2)*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2))*diff(Y_coef_cp[1],'phi',1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',3)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',2,'phi',1)+((2-2*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',2)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1,'phi',1)+(2-2*Delta_coef_cp[0])*diff(X_coef_cp[1],'phi',1)*(diff(Y_coef_cp[1],'chi',1))**2+((4*Delta_coef_cp[0]-4)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1))*diff(Y_coef_cp[1],'chi',1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',3)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',2,'phi',1))*dl_p*kap_p*diff(kap_p,'phi',1)+(((2-2*Delta_coef_cp[0])*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2))*diff(Y_coef_cp[1],'phi',2)+(2-2*Delta_coef_cp[0])*diff(X_coef_cp[1],'chi',2)*(diff(Y_coef_cp[1],'phi',1))**2+((2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],'chi',3)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(Y_coef_cp[1],'chi',2,'phi',1)+(2*Delta_coef_cp[0]-2)*diff(X_coef_cp[1],'phi',1)*diff(Y_coef_cp[1],'chi',2)+(2-2*Delta_coef_cp[0])*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1,'phi',1)+((8-8*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],'chi',2)+(4-4*Delta_coef_cp[0])*diff(X_coef_cp[1],'chi',1,'phi',1)-2*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1)+(6-6*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',3)+(6-6*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2,'phi',1)-2*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',2))*diff(Y_coef_cp[1],'phi',1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',4)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',3,'phi',1)+((4*Delta_coef_cp[0]-4)*iota_coef[0]**2*X_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)+2*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',3)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',2,'phi',2)+((6*Delta_coef_cp[0]-6)*iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)+(4*Delta_coef_cp[0]-4)*Y_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)+2*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',2,'phi',1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]**2*X_coef_cp[1]*(diff(Y_coef_cp[1],'chi',2))**2+((4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],'chi',1,'phi',1)+((8*Delta_coef_cp[0]-8)*iota_coef[0]*diff(X_coef_cp[1],'phi',1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]**2*diff(X_coef_cp[1],'chi',1)+2*iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',1)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],'phi',2)+2*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'phi',1)+(2-2*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1)+2*iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',2)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1,'phi',2)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*(diff(Y_coef_cp[1],'chi',1,'phi',1))**2+(((6*Delta_coef_cp[0]-6)*diff(X_coef_cp[1],'phi',1)+2*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',1)+(6-6*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1))*diff(Y_coef_cp[1],'chi',1,'phi',1)+((2*Delta_coef_cp[0]-2)*diff(X_coef_cp[1],'phi',2)+2*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'phi',1)+(4-4*Delta_coef_cp[0])*iota_coef[0]**2*diff(X_coef_cp[1],'chi',2))*(diff(Y_coef_cp[1],'chi',1))**2+((8-8*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',3)+(10-10*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2,'phi',1)-4*iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',2)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',2)-2*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1,'phi',1))*diff(Y_coef_cp[1],'chi',1)+(2-2*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',4)+(4-4*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',3,'phi',1)-2*iota_coef[0]*Y_coef_cp[1]**2*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',3)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',2,'phi',2)-2*Y_coef_cp[1]**2*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',2,'phi',1))*dl_p*kap_p**2
    )*tau_p
    coef_B_psi_dphi_0_dchi_0_all_but_Y_const_6 = (
        ((2-2*Delta_coef_cp[0])*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',2)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1))**2)*dl_p*diff(kap_p,'phi',1)+(((2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(2*Delta_coef_cp[0]-2)*(diff(X_coef_cp[1],'chi',1))**2)*diff(Y_coef_cp[1],'phi',1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',3)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',2,'phi',1)+((4-4*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(6-6*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)-2*X_coef_cp[1]**2*diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',2)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1,'phi',1)+((2-2*Delta_coef_cp[0])*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1)-2*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)*diff(X_coef_cp[1],'phi',1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',3)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2,'phi',1)+((6*Delta_coef_cp[0]-6)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)+2*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',2)+(4*Delta_coef_cp[0]-4)*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',1)+2*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*(diff(X_coef_cp[1],'chi',1))**2)*dl_p*kap_p
    )*dl_p*kap_p**3
    coef_B_psi_dphi_0_dchi_0_all_but_Y_const_7 = (
        (
            ((1-Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(1-Delta_coef_cp[0])*(diff(X_coef_cp[1],'chi',1))**2)*diff(Y_coef_cp[1],'phi',1)+(Delta_coef_cp[0]-1)*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',3)+(Delta_coef_cp[0]-1)*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',2,'phi',1)+((Delta_coef_cp[0]-1)*X_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',2)+(Delta_coef_cp[0]-1)*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1,'phi',1)+((Delta_coef_cp[0]-1)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)+(1-Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2))*diff(Y_coef_cp[1],'chi',1)+(1-Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',3)+(1-Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2,'phi',1)+(1-Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',2)+(1-Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',1)
        )*kap_p*diff(kap_p,'phi',2)
        +(
            (((2-2*Delta_coef_cp[0])*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2))*diff(Y_coef_cp[1],'phi',1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',3)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',2,'phi',1)+((2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',2)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1,'phi',1)+(2*Delta_coef_cp[0]-2)*diff(X_coef_cp[1],'phi',1)*(diff(Y_coef_cp[1],'chi',1))**2+((4-4*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1))*diff(Y_coef_cp[1],'chi',1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',3)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',2,'phi',1))*dl_p*kap_p**2
        )*diff(tau_p,'phi',1)
        +(((2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(2*Delta_coef_cp[0]-2)*(diff(X_coef_cp[1],'chi',1))**2)*diff(Y_coef_cp[1],'phi',1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',3)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',2,'phi',1)+((2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',2)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1,'phi',1)+((2-2*Delta_coef_cp[0])*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2))*diff(Y_coef_cp[1],'chi',1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',3)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2,'phi',1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',2)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',1))*(diff(kap_p,'phi',1))**2
        +(((2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(2-2*Delta_coef_cp[0])*(diff(X_coef_cp[1],'chi',1))**2)*diff(Y_coef_cp[1],'phi',2)+((4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',3)+(4-4*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2,'phi',1)+((4-4*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],'chi',1)-X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',2)+(4-4*Delta_coef_cp[0])*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',1)-diff(Delta_coef_cp[0],'phi',1)*(diff(X_coef_cp[1],'chi',1))**2)*diff(Y_coef_cp[1],'phi',1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]**2*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',4)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',3,'phi',1)+((4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(6*Delta_coef_cp[0]-6)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)+iota_coef[0]*X_coef_cp[1]**2*diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',3)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',2,'phi',2)+((4*Delta_coef_cp[0]-4)*X_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(8*Delta_coef_cp[0]-8)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)+X_coef_cp[1]**2*diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',2,'phi',1)+((2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(X_coef_cp[1],'phi',2)+((4*Delta_coef_cp[0]-4)*iota_coef[0]*diff(X_coef_cp[1],'chi',1)+X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'phi',1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]**2*(diff(X_coef_cp[1],'chi',1))**2+2*iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',2)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1,'phi',2)+((4*Delta_coef_cp[0]-4)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)+(4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1,'phi',1)+((2*Delta_coef_cp[0]-2)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',2)+diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)+(4-4*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',3)+(4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2,'phi',1)+((2-2*Delta_coef_cp[0])*iota_coef[0]**2*diff(X_coef_cp[1],'chi',1)-iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',2))*diff(Y_coef_cp[1],'chi',1)+(2-2*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',4)+(4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',3,'phi',1)+((2-2*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)-iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',3)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2,'phi',2)+((4-4*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)-X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',2,'phi',1)-iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',2)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',2)-Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',1))*kap_p*diff(kap_p,'phi',1)
        +(((Delta_coef_cp[0]-1)*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(Delta_coef_cp[0]-1)*(diff(X_coef_cp[1],'chi',1))**2)*diff(Y_coef_cp[1],'phi',3)+((3*Delta_coef_cp[0]-3)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',3)+(3*Delta_coef_cp[0]-3)*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2,'phi',1)+((3*Delta_coef_cp[0]-3)*iota_coef[0]*diff(X_coef_cp[1],'chi',1)+X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',2)+(3*Delta_coef_cp[0]-3)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',1)+diff(Delta_coef_cp[0],'phi',1)*(diff(X_coef_cp[1],'chi',1))**2)*diff(Y_coef_cp[1],'phi',2)+((1-Delta_coef_cp[0])*diff(X_coef_cp[1],'chi',2)*diff(X_coef_cp[1],'phi',2)-diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',2)*diff(X_coef_cp[1],'phi',1)+(3*Delta_coef_cp[0]-3)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',4)+(6*Delta_coef_cp[0]-6)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',3,'phi',1)+((2*Delta_coef_cp[0]-2)*iota_coef[0]**2*diff(X_coef_cp[1],'chi',1)+2*iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',3)+(3*Delta_coef_cp[0]-3)*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2,'phi',2)+((4*Delta_coef_cp[0]-4)*iota_coef[0]*diff(X_coef_cp[1],'chi',1)+2*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',2,'phi',1)+(1-Delta_coef_cp[0])*iota_coef[0]**2*(diff(X_coef_cp[1],'chi',2))**2+(2-2*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],'chi',1,'phi',1)*diff(X_coef_cp[1],'chi',2)+(2*Delta_coef_cp[0]-2)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',2)+diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',1))*diff(Y_coef_cp[1],'phi',1)+(1-Delta_coef_cp[0])*iota_coef[0]**3*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',5)+(3-3*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',4,'phi',1)+((3-3*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(4-4*Delta_coef_cp[0])*iota_coef[0]**3*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)-iota_coef[0]**2*X_coef_cp[1]**2*diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',4)+(3-3*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',3,'phi',2)+((6-6*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(9-9*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)-2*iota_coef[0]*X_coef_cp[1]**2*diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',3,'phi',1)+((2-2*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'phi',2)+((3-3*Delta_coef_cp[0])*iota_coef[0]**2*diff(X_coef_cp[1],'chi',1)-iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'phi',1)+(1-Delta_coef_cp[0])*iota_coef[0]**3*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(4-4*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1)+(2-2*Delta_coef_cp[0])*iota_coef[0]**3*(diff(X_coef_cp[1],'chi',1))**2-2*iota_coef[0]**2*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',3)+(1-Delta_coef_cp[0])*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',2,'phi',3)+((3-3*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(6-6*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)-X_coef_cp[1]**2*diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',2,'phi',2)+((2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'phi',2)+((6-6*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],'chi',1)-X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'phi',1)+(Delta_coef_cp[0]-1)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1)+(3-3*Delta_coef_cp[0])*iota_coef[0]**2*(diff(X_coef_cp[1],'chi',1))**2-3*iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',2,'phi',1)+((1-Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'phi',3)+((Delta_coef_cp[0]-1)*diff(X_coef_cp[1],'phi',1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],'chi',1)-X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'phi',2)+diff(Delta_coef_cp[0],'phi',1)*(diff(X_coef_cp[1],'phi',1))**2+((Delta_coef_cp[0]-1)*iota_coef[0]**2*diff(X_coef_cp[1],'chi',2)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*diff(X_coef_cp[1],'chi',1,'phi',1))*diff(X_coef_cp[1],'phi',1)+(3*Delta_coef_cp[0]-3)*iota_coef[0]**3*X_coef_cp[1]*diff(X_coef_cp[1],'chi',3)+(2*Delta_coef_cp[0]-2)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2,'phi',1)+((Delta_coef_cp[0]-1)*iota_coef[0]**3*diff(X_coef_cp[1],'chi',1)+iota_coef[0]**2*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',2)+(2-2*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',2)+((1-Delta_coef_cp[0])*iota_coef[0]**2*diff(X_coef_cp[1],'chi',1)-iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',1,'phi',1))*diff(Y_coef_cp[1],'chi',2)+(1-Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1,'phi',3)+((3-3*Delta_coef_cp[0])*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)+(3*Delta_coef_cp[0]-3)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2)-X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1,'phi',2)+((3-3*Delta_coef_cp[0])*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',2)-2*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)+(7*Delta_coef_cp[0]-7)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',3)+(8*Delta_coef_cp[0]-8)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2,'phi',1)+((3*Delta_coef_cp[0]-3)*iota_coef[0]**2*diff(X_coef_cp[1],'chi',1)+3*iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',2)+(Delta_coef_cp[0]-1)*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',2)+X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1,'phi',1))*diff(Y_coef_cp[1],'chi',1,'phi',1)+((1-Delta_coef_cp[0])*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',3)+((1-Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],'chi',2)-diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(X_coef_cp[1],'phi',2)+((Delta_coef_cp[0]-1)*iota_coef[0]**2*diff(X_coef_cp[1],'chi',3)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*diff(X_coef_cp[1],'chi',2,'phi',1)+(Delta_coef_cp[0]-1)*diff(X_coef_cp[1],'chi',1,'phi',2)+diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1,'phi',1))*diff(X_coef_cp[1],'phi',1)+(3*Delta_coef_cp[0]-3)*iota_coef[0]**3*X_coef_cp[1]*diff(X_coef_cp[1],'chi',4)+(6*Delta_coef_cp[0]-6)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',3,'phi',1)+((2*Delta_coef_cp[0]-2)*iota_coef[0]**3*diff(X_coef_cp[1],'chi',1)+2*iota_coef[0]**2*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',3)+(3*Delta_coef_cp[0]-3)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2,'phi',2)+((3*Delta_coef_cp[0]-3)*iota_coef[0]**2*diff(X_coef_cp[1],'chi',1)+2*iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',2,'phi',1)+(1-Delta_coef_cp[0])*iota_coef[0]**3*(diff(X_coef_cp[1],'chi',2))**2+(2-2*Delta_coef_cp[0])*iota_coef[0]**2*diff(X_coef_cp[1],'chi',1,'phi',1)*diff(X_coef_cp[1],'chi',2))*diff(Y_coef_cp[1],'chi',1)+((1-Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',3)+(1-Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2,'phi',1))*diff(X_coef_cp[1],'phi',2)+((-iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',3))-Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',2,'phi',1))*diff(X_coef_cp[1],'phi',1)+(Delta_coef_cp[0]-1)*iota_coef[0]**3*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',5)+(3*Delta_coef_cp[0]-3)*iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',4,'phi',1)+((Delta_coef_cp[0]-1)*iota_coef[0]**3*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)+iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',4)+(3*Delta_coef_cp[0]-3)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',3,'phi',2)+((3*Delta_coef_cp[0]-3)*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)+2*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',3,'phi',1)+((2-2*Delta_coef_cp[0])*iota_coef[0]**3*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(3-3*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1))*diff(X_coef_cp[1],'chi',3)+(Delta_coef_cp[0]-1)*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2,'phi',3)+((3*Delta_coef_cp[0]-3)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)+X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',2,'phi',2)+((3-3*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(4-4*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1)+iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(X_coef_cp[1],'chi',2,'phi',1)-iota_coef[0]**2*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*(diff(X_coef_cp[1],'chi',2))**2+((1-Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',2)-2*iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1,'phi',1))*diff(X_coef_cp[1],'chi',2)+(Delta_coef_cp[0]-1)*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',3)+((1-Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1)+Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(X_coef_cp[1],'chi',1,'phi',2)-Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*(diff(X_coef_cp[1],'chi',1,'phi',1))**2)*kap_p**2
        +coef_B_psi_dphi_0_dchi_0_all_but_Y_const_5
        +coef_B_psi_dphi_0_dchi_0_all_but_Y_const_6
    )
    coef_B_psi_dphi_0_dchi_0_all_but_Y_const_8 = (
        -(2*B_denom_coef_c[0]*iota_coef[0]*diff(Delta_coef_cp[1],'chi',2)+2*B_denom_coef_c[0]*diff(Delta_coef_cp[1],'chi',1,'phi',1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*diff(B_denom_coef_c[1],'chi',2))
        +(3*diff(Delta_coef_cp[0],'phi',1)*diff(B_denom_coef_c[1],'chi',1))
    )
    coef_B_psi_dphi_0_dchi_0_all_but_Y_const_9 = (
        coef_B_psi_dphi_0_dchi_0_all_but_Y_const_1*dl_p*kap_p**2*diff(tau_p,'phi',1)
        +coef_B_psi_dphi_0_dchi_0_all_but_Y_const_2*tau_p
        +coef_B_psi_dphi_0_dchi_0_all_but_Y_const_3
        +coef_B_psi_dphi_0_dchi_0_all_but_Y_const_4
    )
    coef_B_psi_dphi_0_dchi_0_all_but_Y = lambda n_eval : (
        (
            coef_B_psi_dphi_0_dchi_0_all_but_Y_const_9*n_eval
            +coef_B_psi_dphi_0_dchi_0_all_but_Y_const_7
        )/denom_e(n_eval)
        +coef_B_psi_dphi_0_dchi_0_all_but_Y_const_8/n_eval
    )

    # --------------------------------------------------------------------------
    coef_B_psi_dphi_0_dchi_0_in_Y_RHS_const_1 = (
        B_alpha_coef[0]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'phi',1)-B_alpha_coef[0]*iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',2)-B_alpha_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1,'phi',1)-B_alpha_coef[0]*diff(X_coef_cp[1],'phi',1)*(diff(Y_coef_cp[1],'chi',1))**2+(B_alpha_coef[0]*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+B_alpha_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1))*diff(Y_coef_cp[1],'chi',1)
    )
    coef_B_psi_dphi_0_dchi_0_in_Y_RHS_const_2 = (
        ((-B_alpha_coef[0]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1))-B_alpha_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2))*diff(Y_coef_cp[1],'phi',1)+B_alpha_coef[0]*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',3)+B_alpha_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',2,'phi',1)+(B_alpha_coef[0]*iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)+B_alpha_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+B_alpha_coef[0]*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',2)+B_alpha_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1,'phi',1)+B_alpha_coef[0]*diff(X_coef_cp[1],'phi',1)*(diff(Y_coef_cp[1],'chi',1))**2+((-2*B_alpha_coef[0]*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2))-B_alpha_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1))*diff(Y_coef_cp[1],'chi',1)-B_alpha_coef[0]*iota_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',3)-B_alpha_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',2,'phi',1)
    )
    coef_B_psi_dphi_0_dchi_0_in_Y_RHS = lambda n_eval : (
        coef_B_psi_dphi_0_dchi_0_in_Y_RHS_const_1*n_eval
        + coef_B_psi_dphi_0_dchi_0_in_Y_RHS_const_2
    )/denom_d(n_eval)

    # --------------------------------------------------------------------------
    coef_B_psi_dphi_0_dchi_1_all_but_Y_const_1 = (
        (2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*(diff(Y_coef_cp[1],'chi',1))**2+(2-2*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)
    )*dl_p*kap_p**2
    coef_B_psi_dphi_0_dchi_1_all_but_Y_const_2 = (
        ((2-2*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*(diff(Y_coef_cp[1],'chi',1))**2+(2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1))*dl_p*kap_p*diff(kap_p,'phi',1)+((4-4*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'phi',1)+((6*Delta_coef_cp[0]-6)*iota_coef[0]**2*X_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)+(2-2*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',2)+((6*Delta_coef_cp[0]-6)*iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1,'phi',1)+((4*Delta_coef_cp[0]-4)*iota_coef[0]*diff(X_coef_cp[1],'phi',1)+2*iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*(diff(Y_coef_cp[1],'chi',1))**2+((4-4*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(4-4*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1)-2*iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1))*dl_p*kap_p**2
    )
    coef_B_psi_dphi_0_dchi_1_all_but_Y_const_3 = (
        ((Delta_coef_cp[0]-1)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)+(1-Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1))**2)*kap_p*diff(kap_p,'phi',2)+((2-2*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1))**2)*(diff(kap_p,'phi',1))**2+((4-4*Delta_coef_cp[0])*iota_coef[0]*(diff(X_coef_cp[1],'chi',1))**2*diff(Y_coef_cp[1],'phi',1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',2)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1,'phi',1)+((4*Delta_coef_cp[0]-4)*iota_coef[0]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)+iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1)+(4-4*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',2)+(4-4*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',1)-iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*(diff(X_coef_cp[1],'chi',1))**2)*kap_p*diff(kap_p,'phi',1)+((3*Delta_coef_cp[0]-3)*iota_coef[0]*(diff(X_coef_cp[1],'chi',1))**2*diff(Y_coef_cp[1],'phi',2)+((6*Delta_coef_cp[0]-6)*iota_coef[0]**2*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',2)+(6*Delta_coef_cp[0]-6)*iota_coef[0]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',1)+2*iota_coef[0]*diff(Delta_coef_cp[0],'phi',1)*(diff(X_coef_cp[1],'chi',1))**2)*diff(Y_coef_cp[1],'phi',1)+(3-3*Delta_coef_cp[0])*iota_coef[0]**3*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',3)+(6-6*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',2,'phi',1)+((6-6*Delta_coef_cp[0])*iota_coef[0]**2*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)+(3-3*Delta_coef_cp[0])*iota_coef[0]**3*(diff(X_coef_cp[1],'chi',1))**2-2*iota_coef[0]**2*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',2)+(3-3*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1,'phi',2)+((6-6*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)-2*iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1,'phi',1)+((3-3*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',2)-2*iota_coef[0]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)+(Delta_coef_cp[0]-1)*iota_coef[0]**3*X_coef_cp[1]*diff(X_coef_cp[1],'chi',3)+(2*Delta_coef_cp[0]-2)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2,'phi',1)+((3*Delta_coef_cp[0]-3)*iota_coef[0]**3*diff(X_coef_cp[1],'chi',1)+iota_coef[0]**2*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',2)+(Delta_coef_cp[0]-1)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',2)+iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1,'phi',1))*diff(Y_coef_cp[1],'chi',1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]**3*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',3)+(4*Delta_coef_cp[0]-4)*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',2,'phi',1)+iota_coef[0]**2*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',2)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',2)+iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',1))*kap_p**2
        +coef_B_psi_dphi_0_dchi_1_all_but_Y_const_1*diff(tau_p,'phi',1)
        +coef_B_psi_dphi_0_dchi_1_all_but_Y_const_2*tau_p
    )
    coef_B_psi_dphi_0_dchi_1_all_but_Y_const_4 = (
        (2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1))**2
    )*dl_p*kap_p
    coef_B_psi_dphi_0_dchi_1_all_but_Y_const_5 = (
        ((2-2*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'phi',1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',2)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',1,'phi',1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*(diff(Y_coef_cp[1],'chi',1))**2+((2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1)+(4-4*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',2)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',1,'phi',1))*dl_p*kap_p*diff(kap_p,'phi',1)+((2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'phi',2)+(2*Delta_coef_cp[0]-2)*diff(X_coef_cp[1],'chi',1)*(diff(Y_coef_cp[1],'phi',1))**2+((4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],'chi',2)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(Y_coef_cp[1],'chi',1,'phi',1)+((2-2*Delta_coef_cp[0])*diff(X_coef_cp[1],'phi',1)+(6*Delta_coef_cp[0]-6)*iota_coef[0]*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1)+(12*Delta_coef_cp[0]-12)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(6*Delta_coef_cp[0]-6)*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1)+2*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'phi',1)+(6-6*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',3)+(8-8*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',2,'phi',1)+((10-10*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)+(8-8*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(4-4*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)-4*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',2)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',1,'phi',2)+((8-8*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)+(4-4*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)-2*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',1,'phi',1)+((6-6*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],'phi',1)-2*iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*(diff(Y_coef_cp[1],'chi',1))**2+((2-2*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],'phi',2)-2*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'phi',1)+(14*Delta_coef_cp[0]-14)*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(6*Delta_coef_cp[0]-6)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1)+2*iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1)+(6*Delta_coef_cp[0]-6)*iota_coef[0]**2*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',3)+(8*Delta_coef_cp[0]-8)*iota_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',2,'phi',1)+4*iota_coef[0]*Y_coef_cp[1]**2*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',2)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',1,'phi',2)+2*Y_coef_cp[1]**2*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1,'phi',1))*dl_p*kap_p**2
    )
    coef_B_psi_dphi_0_dchi_1_all_but_Y_const_6 = (
        ((2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'phi',1)+(4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',2)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',1,'phi',1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*(diff(Y_coef_cp[1],'chi',1))**2+((2-2*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',2)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',1,'phi',1))*dl_p*kap_p**2
    )
    coef_B_psi_dphi_0_dchi_1_all_but_Y_const_7 = (
        ((Delta_coef_cp[0]-1)*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'phi',1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',2)+(1-Delta_coef_cp[0])*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',1,'phi',1)+((1-Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(1-Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(Delta_coef_cp[0]-1)*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1)+(Delta_coef_cp[0]-1)*iota_coef[0]*Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1))**2)*kap_p*diff(kap_p,'phi',2)+((2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'phi',1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',2)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',1,'phi',1)+((2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1)+(4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1))**2)*(diff(kap_p,'phi',1))**2+((2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'phi',2)+((8*Delta_coef_cp[0]-8)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(4*Delta_coef_cp[0]-4)*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*(diff(X_coef_cp[1],'chi',1))**2+X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'phi',1)+(6-6*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',3)+(8-8*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',2,'phi',1)+((8-8*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(10-10*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)-2*iota_coef[0]*X_coef_cp[1]**2*diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',2)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',1,'phi',2)+((4-4*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)-X_coef_cp[1]**2*diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',1,'phi',1)+((2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'phi',2)+((4-4*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],'chi',1)-X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'phi',1)+(6*Delta_coef_cp[0]-6)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2)-iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1)+(6*Delta_coef_cp[0]-6)*iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',3)+(8*Delta_coef_cp[0]-8)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2,'phi',1)+((4*Delta_coef_cp[0]-4)*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)+2*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',2)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',2)+((4*Delta_coef_cp[0]-4)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)+X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',1,'phi',1)+iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*(diff(X_coef_cp[1],'chi',1))**2)*kap_p*diff(kap_p,'phi',1)+((1-Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'phi',3)+((6-6*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(3-3*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1)+(3-3*Delta_coef_cp[0])*iota_coef[0]*(diff(X_coef_cp[1],'chi',1))**2-X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'phi',2)+((Delta_coef_cp[0]-1)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',2)+diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)+(9-9*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',3)+(12-12*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2,'phi',1)+((5-5*Delta_coef_cp[0])*iota_coef[0]**2*diff(X_coef_cp[1],'chi',1)-4*iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',2)+(3-3*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',2)+((4-4*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],'chi',1)-2*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',1,'phi',1)-iota_coef[0]*diff(Delta_coef_cp[0],'phi',1)*(diff(X_coef_cp[1],'chi',1))**2)*diff(Y_coef_cp[1],'phi',1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]**3*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',4)+(9*Delta_coef_cp[0]-9)*iota_coef[0]**2*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',3,'phi',1)+((9*Delta_coef_cp[0]-9)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(11*Delta_coef_cp[0]-11)*iota_coef[0]**3*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)+3*iota_coef[0]**2*X_coef_cp[1]**2*diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',3)+(6*Delta_coef_cp[0]-6)*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',2,'phi',2)+((12*Delta_coef_cp[0]-12)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(15*Delta_coef_cp[0]-15)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)+4*iota_coef[0]*X_coef_cp[1]**2*diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',2,'phi',1)+((4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'phi',2)+((6*Delta_coef_cp[0]-6)*iota_coef[0]**2*diff(X_coef_cp[1],'chi',1)+2*iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'phi',1)+(2-2*Delta_coef_cp[0])*iota_coef[0]**3*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(5*Delta_coef_cp[0]-5)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1)+(3*Delta_coef_cp[0]-3)*iota_coef[0]**3*(diff(X_coef_cp[1],'chi',1))**2+3*iota_coef[0]**2*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',2)+(Delta_coef_cp[0]-1)*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',1,'phi',3)+((3*Delta_coef_cp[0]-3)*X_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(3*Delta_coef_cp[0]-3)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)+X_coef_cp[1]**2*diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',1,'phi',2)+((2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(X_coef_cp[1],'phi',2)+((6*Delta_coef_cp[0]-6)*iota_coef[0]*diff(X_coef_cp[1],'chi',1)+X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'phi',1)+(10-10*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(2-2*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1)+iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1,'phi',1)+((Delta_coef_cp[0]-1)*X_coef_cp[1]*diff(X_coef_cp[1],'phi',3)+((1-Delta_coef_cp[0])*diff(X_coef_cp[1],'phi',1)+(3*Delta_coef_cp[0]-3)*iota_coef[0]*diff(X_coef_cp[1],'chi',1)+X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'phi',2)-diff(Delta_coef_cp[0],'phi',1)*(diff(X_coef_cp[1],'phi',1))**2+((1-Delta_coef_cp[0])*iota_coef[0]**2*diff(X_coef_cp[1],'chi',2)+(2-2*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],'chi',1,'phi',1)+iota_coef[0]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(X_coef_cp[1],'phi',1)+(9-9*Delta_coef_cp[0])*iota_coef[0]**3*X_coef_cp[1]*diff(X_coef_cp[1],'chi',3)+(11-11*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2,'phi',1)+((3-3*Delta_coef_cp[0])*iota_coef[0]**3*diff(X_coef_cp[1],'chi',1)-4*iota_coef[0]**2*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',2)+(1-Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',2)-iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1,'phi',1))*diff(Y_coef_cp[1],'chi',1)+((2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(Delta_coef_cp[0]-1)*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1))*diff(X_coef_cp[1],'phi',2)+(2*iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',2)+Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1,'phi',1))*diff(X_coef_cp[1],'phi',1)+(4-4*Delta_coef_cp[0])*iota_coef[0]**3*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',4)+(9-9*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',3,'phi',1)+((2-2*Delta_coef_cp[0])*iota_coef[0]**3*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)-3*iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',3)+(6-6*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2,'phi',2)+((4-4*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)-4*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',2,'phi',1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]**3*Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',2))**2+((5*Delta_coef_cp[0]-5)*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1)+iota_coef[0]**2*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(X_coef_cp[1],'chi',2)+(1-Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',3)+((2-2*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)-X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',1,'phi',2)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1,'phi',1))**2)*kap_p**2
        +coef_B_psi_dphi_0_dchi_1_all_but_Y_const_6*diff(tau_p,'phi',1)
        +coef_B_psi_dphi_0_dchi_1_all_but_Y_const_5*tau_p
    )
    coef_B_psi_dphi_0_dchi_1_all_but_Y_const_8 = (
        ((2-2*Delta_coef_cp[0])*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',1)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1))*dl_p*diff(kap_p,'phi',1)+((2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'phi',1)+(4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',2)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',1,'phi',1)+((4-4*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)-2*X_coef_cp[1]**2*diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',1)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1))**2+2*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*dl_p*kap_p
    )
    coef_B_psi_dphi_0_dchi_1_all_but_Y_const_9 = (
        (4-4*Delta_coef_cp[0])*iota_coef[0]*diff(B_denom_coef_c[1],'chi',1)
        -4*B_denom_coef_c[1]*diff(Delta_coef_cp[0],'phi',1)
        -(1-Delta_coef_cp[0])*iota_coef[0]*diff(B_denom_coef_c[1],'chi',1)
        +B_denom_coef_c[1]*diff(Delta_coef_cp[0],'phi',1)
    )
    coef_B_psi_dphi_0_dchi_1_all_but_Y_const_10 = (
        2*B_denom_coef_c[0]*diff(Delta_coef_cp[1],'phi',1)+4*B_denom_coef_c[0]*iota_coef[0]*diff(Delta_coef_cp[1],'chi',1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*diff(B_denom_coef_c[1],'chi',1)
    )
    coef_B_psi_dphi_0_dchi_1_all_but_Y_const_11 = (
        (4*Delta_coef_cp[0]-4)*iota_coef[0]*diff(B_denom_coef_c[1],'chi',1)
        -(Delta_coef_cp[0]-1)*iota_coef[0]*diff(B_denom_coef_c[1],'chi',1)
    )
    coef_B_psi_dphi_0_dchi_1_all_but_Y = lambda n_eval : (
        -(
            coef_B_psi_dphi_0_dchi_1_all_but_Y_const_3*n_eval
            +coef_B_psi_dphi_0_dchi_1_all_but_Y_const_7
        )/denom_e(n_eval)
        +(
            coef_B_psi_dphi_0_dchi_1_all_but_Y_const_4*n_eval
            +coef_B_psi_dphi_0_dchi_1_all_but_Y_const_8
        )/denom_a(n_eval)
        +coef_B_psi_dphi_0_dchi_1_all_but_Y_const_9/(n_eval**2-n_eval)
        +coef_B_psi_dphi_0_dchi_1_all_but_Y_const_11/(n_eval-1)
        -coef_B_psi_dphi_0_dchi_1_all_but_Y_const_10/n_eval
    )

    # --------------------------------------------------------------------------
    coef_B_psi_dphi_0_dchi_1_in_Y_RHS_const_1 = (
        B_alpha_coef[0]*iota_coef[0]*X_coef_cp[1]*(diff(Y_coef_cp[1],'chi',1))**2-B_alpha_coef[0]*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)
    )
    coef_B_psi_dphi_0_dchi_1_in_Y_RHS_const_2 = (
        B_alpha_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'phi',1)-2*B_alpha_coef[0]*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',2)-B_alpha_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',1,'phi',1)-B_alpha_coef[0]*iota_coef[0]*X_coef_cp[1]*(diff(Y_coef_cp[1],'chi',1))**2+(B_alpha_coef[0]*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)-B_alpha_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'phi',1))*diff(Y_coef_cp[1],'chi',1)+2*B_alpha_coef[0]*iota_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',2)+B_alpha_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',1,'phi',1)
    )
    coef_B_psi_dphi_0_dchi_1_in_Y_RHS = lambda n_eval : -(
        coef_B_psi_dphi_0_dchi_1_in_Y_RHS_const_1*n_eval
        + coef_B_psi_dphi_0_dchi_1_in_Y_RHS_const_2
    )/denom_d(n_eval)

    # --------------------------------------------------------------------------
    coef_B_psi_dphi_0_dchi_2_all_but_Y_const_1 = (
        (2-2*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*(diff(Y_coef_cp[1],'chi',1))**2+(2*Delta_coef_cp[0]-2)*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)
    )*dl_p*kap_p**2
    coef_B_psi_dphi_0_dchi_2_all_but_Y_const_2 = (
        ((2-2*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]**2*Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1))**2)*kap_p*diff(kap_p,'phi',1)+((3-3*Delta_coef_cp[0])*iota_coef[0]**2*(diff(X_coef_cp[1],'chi',1))**2*diff(Y_coef_cp[1],'phi',1)+(3*Delta_coef_cp[0]-3)*iota_coef[0]**3*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',2)+(3*Delta_coef_cp[0]-3)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1,'phi',1)+((3*Delta_coef_cp[0]-3)*iota_coef[0]**2*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)+iota_coef[0]**2*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1)+(3-3*Delta_coef_cp[0])*iota_coef[0]**3*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',2)+(3-3*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',1)-iota_coef[0]**2*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*(diff(X_coef_cp[1],'chi',1))**2)*kap_p**2
    )
    coef_B_psi_dphi_0_dchi_2_all_but_Y_const_3 = (
        ((Delta_coef_cp[0]-1)*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',1)+(1-Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1))*kap_p*diff(kap_p,'phi',2)
        +((2-2*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1))*(diff(kap_p,'phi',1))**2+((4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'phi',1)+(6*Delta_coef_cp[0]-6)*iota_coef[0]**2*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',2)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',1,'phi',1)+((4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)+iota_coef[0]*X_coef_cp[1]**2*diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',1)+(6-6*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1)+(2-2*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1))**2-iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*kap_p*diff(kap_p,'phi',1)+((3*Delta_coef_cp[0]-3)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'phi',2)+((9*Delta_coef_cp[0]-9)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(6*Delta_coef_cp[0]-6)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1)+(3*Delta_coef_cp[0]-3)*iota_coef[0]**2*(diff(X_coef_cp[1],'chi',1))**2+2*iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'phi',1)+(6-6*Delta_coef_cp[0])*iota_coef[0]**3*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',3)+(9-9*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',2,'phi',1)+((9-9*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(9-9*Delta_coef_cp[0])*iota_coef[0]**3*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)-3*iota_coef[0]**2*X_coef_cp[1]**2*diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',2)+(3-3*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',1,'phi',2)+((6-6*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(3-3*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)-2*iota_coef[0]*X_coef_cp[1]**2*diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',1,'phi',1)+((2-2*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'phi',2)+((3-3*Delta_coef_cp[0])*iota_coef[0]**2*diff(X_coef_cp[1],'chi',1)-iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'phi',1)+(7*Delta_coef_cp[0]-7)*iota_coef[0]**3*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(2*Delta_coef_cp[0]-2)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1))*diff(Y_coef_cp[1],'chi',1)+(1-Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',2)-iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)+(6*Delta_coef_cp[0]-6)*iota_coef[0]**3*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',3)+(9*Delta_coef_cp[0]-9)*iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2,'phi',1)+((2*Delta_coef_cp[0]-2)*iota_coef[0]**3*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)+3*iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',2)+(3*Delta_coef_cp[0]-3)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',2)+((Delta_coef_cp[0]-1)*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)+2*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',1,'phi',1))*kap_p**2
        +((2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',1))*dl_p*kap_p**2*diff(tau_p,'phi',1)
    )
    coef_B_psi_dphi_0_dchi_2_all_but_Y_const_4 = (
        ((2-2*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',1))*dl_p*kap_p*diff(kap_p,'phi',1)
        +(((2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)+(6-6*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'phi',1)+(6*Delta_coef_cp[0]-6)*iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',2)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',1,'phi',1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]**2*X_coef_cp[1]*(diff(Y_coef_cp[1],'chi',1))**2+((4*Delta_coef_cp[0]-4)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(4-4*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)+2*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',1)+(6-6*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',2)+(4-4*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',1,'phi',1)-2*iota_coef[0]*Y_coef_cp[1]**2*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*dl_p*kap_p**2
    )
    coef_B_psi_dphi_0_dchi_2_all_but_Y_const_5 = (
        (Delta_coef_cp[0]-1)*iota_coef[0]*B_denom_coef_c[1]-(4*Delta_coef_cp[0]-4)*iota_coef[0]*B_denom_coef_c[1]
    )
    coef_B_psi_dphi_0_dchi_2_all_but_Y_const_6 = (
        2*B_denom_coef_c[0]*iota_coef[0]*Delta_coef_cp[1]
    )
    coef_B_psi_dphi_0_dchi_2_all_but_Y_const_7 = (
        ((2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1))*dl_p*kap_p
    )
    coef_B_psi_dphi_0_dchi_2_all_but_Y = lambda n_eval : (
        (
            (
                coef_B_psi_dphi_0_dchi_2_all_but_Y_const_1*n_eval
                +coef_B_psi_dphi_0_dchi_2_all_but_Y_const_4
            )*tau_p
            +coef_B_psi_dphi_0_dchi_2_all_but_Y_const_2*n_eval
            +coef_B_psi_dphi_0_dchi_2_all_but_Y_const_3
        )/denom_e(n_eval)
        -coef_B_psi_dphi_0_dchi_2_all_but_Y_const_7/denom_a(n_eval)
        +coef_B_psi_dphi_0_dchi_2_all_but_Y_const_5/(n_eval**2-n_eval)
        -coef_B_psi_dphi_0_dchi_2_all_but_Y_const_6/n_eval
    )

    # --------------------------------------------------------------------------
    coef_B_psi_dphi_0_dchi_2_in_Y_RHS_const = (
        B_alpha_coef[0]*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)-B_alpha_coef[0]*iota_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',1)
    )
    coef_B_psi_dphi_0_dchi_2_in_Y_RHS = lambda n_eval : coef_B_psi_dphi_0_dchi_2_in_Y_RHS_const/denom_d(n_eval)

    # --------------------------------------------------------------------------
    coef_B_psi_dphi_0_dchi_3_all_but_Y_const_1 = (
        (Delta_coef_cp[0]-1)*iota_coef[0]**3*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)+(1-Delta_coef_cp[0])*iota_coef[0]**3*Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1))**2
    )*kap_p
    coef_B_psi_dphi_0_dchi_3_all_but_Y_const_2 = (
        ((2*Delta_coef_cp[0]-2)*iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)+(2-2*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',1))*dl_p*kap_p*tau_p+((2*Delta_coef_cp[0]-2)*iota_coef[0]**2*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',1)+(2-2*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1))*diff(kap_p,'phi',1)+((3*Delta_coef_cp[0]-3)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'phi',1)+(4-4*Delta_coef_cp[0])*iota_coef[0]**3*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',2)+(3-3*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',1,'phi',1)+((3-3*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(1-Delta_coef_cp[0])*iota_coef[0]**3*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)-iota_coef[0]**2*X_coef_cp[1]**2*diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]**3*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(3*Delta_coef_cp[0]-3)*iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1)+(Delta_coef_cp[0]-1)*iota_coef[0]**3*Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1))**2+iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*kap_p
    )
    coef_B_psi_dphi_0_dchi_3_all_but_Y = lambda n_eval : (
        coef_B_psi_dphi_0_dchi_3_all_but_Y_const_1*n_eval
        +coef_B_psi_dphi_0_dchi_3_all_but_Y_const_2
    )/denom_b(n_eval)

    # --------------------------------------------------------------------------
    coef_B_psi_dphi_0_dchi_3_in_Y_RHS = lambda n_eval : 0

    # --------------------------------------------------------------------------
    coef_B_psi_dphi_0_dchi_4_all_but_Y_const = (
        (Delta_coef_cp[0]-1)*iota_coef[0]**3*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',1)+(1-Delta_coef_cp[0])*iota_coef[0]**3*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)
    )
    coef_B_psi_dphi_0_dchi_4_all_but_Y = lambda n_eval : -coef_B_psi_dphi_0_dchi_4_all_but_Y_const/denom_c(n_eval)

    # --------------------------------------------------------------------------
    coef_B_psi_dphi_0_dchi_4_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_0_dchi_5_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_0_dchi_5_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_0_dchi_6_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_0_dchi_6_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_0_dchi_7_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_0_dchi_7_in_Y_RHS = lambda n_eval : 0

    # --------------------------------------------------------------------------
    coef_B_psi_dphi_1_dchi_0_all_but_Y_const_1 = (
        (2*Delta_coef_cp[0]-2)*X_coef_cp[1]*(diff(Y_coef_cp[1],'chi',1))**2+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)
    )*dl_p*kap_p**2
    coef_B_psi_dphi_1_dchi_0_all_but_Y_const_2 = (
        ((2-2*Delta_coef_cp[0])*X_coef_cp[1]*(diff(Y_coef_cp[1],'chi',1))**2+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1))*dl_p*kap_p*diff(kap_p,'phi',1)+((4-4*Delta_coef_cp[0])*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'phi',1)+((6*Delta_coef_cp[0]-6)*iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',2)+((6*Delta_coef_cp[0]-6)*X_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1,'phi',1)+((4*Delta_coef_cp[0]-4)*diff(X_coef_cp[1],'phi',1)+2*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*(diff(Y_coef_cp[1],'chi',1))**2+((4-4*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(4-4*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1)-2*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1))*dl_p*kap_p**2
    )
    coef_B_psi_dphi_1_dchi_0_all_but_Y_const_3 = (
        ((Delta_coef_cp[0]-1)*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)+(1-Delta_coef_cp[0])*Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1))**2)*kap_p*diff(kap_p,'phi',2)+((2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1))**2)*(diff(kap_p,'phi',1))**2+((4-4*Delta_coef_cp[0])*(diff(X_coef_cp[1],'chi',1))**2*diff(Y_coef_cp[1],'phi',1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',2)+(4*Delta_coef_cp[0]-4)*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1,'phi',1)+((4*Delta_coef_cp[0]-4)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)+X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1)+(4-4*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',2)+(4-4*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',1)-Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*(diff(X_coef_cp[1],'chi',1))**2)*kap_p*diff(kap_p,'phi',1)+((3*Delta_coef_cp[0]-3)*(diff(X_coef_cp[1],'chi',1))**2*diff(Y_coef_cp[1],'phi',2)+((6*Delta_coef_cp[0]-6)*iota_coef[0]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',2)+(6*Delta_coef_cp[0]-6)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',1)+2*diff(Delta_coef_cp[0],'phi',1)*(diff(X_coef_cp[1],'chi',1))**2)*diff(Y_coef_cp[1],'phi',1)+(3-3*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',3)+(6-6*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',2,'phi',1)+((6-6*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)+(3-3*Delta_coef_cp[0])*iota_coef[0]**2*(diff(X_coef_cp[1],'chi',1))**2-2*iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',2)+(3-3*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1,'phi',2)+((6-6*Delta_coef_cp[0])*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)-2*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1,'phi',1)+((3-3*Delta_coef_cp[0])*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',2)-2*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)+(Delta_coef_cp[0]-1)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',3)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2,'phi',1)+((3*Delta_coef_cp[0]-3)*iota_coef[0]**2*diff(X_coef_cp[1],'chi',1)+iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',2)+(Delta_coef_cp[0]-1)*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',2)+X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1,'phi',1))*diff(Y_coef_cp[1],'chi',1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',3)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',2,'phi',1)+iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',2)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',2)+Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',1))*kap_p**2
        +coef_B_psi_dphi_1_dchi_0_all_but_Y_const_2*tau_p
        +coef_B_psi_dphi_1_dchi_0_all_but_Y_const_1*diff(tau_p,'phi',1)
    )
    coef_B_psi_dphi_1_dchi_0_all_but_Y_const_4 = (
        (
            ((2-2*Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',2)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*(diff(Y_coef_cp[1],'chi',1))**2+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',2))*dl_p*kap_p**2
        )*diff(tau_p,'phi',1)
        +(
            ((2*Delta_coef_cp[0]-2)*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',2)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*(diff(Y_coef_cp[1],'chi',1))**2+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',2))*dl_p*kap_p*diff(kap_p,'phi',1)+(((2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(Y_coef_cp[1],'chi',2)+(4*Delta_coef_cp[0]-4)*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)+(6*Delta_coef_cp[0]-6)*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2))*diff(Y_coef_cp[1],'phi',1)+(4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',3)+(4-4*Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',2,'phi',1)+((8-8*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)+(4-4*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)-2*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',2)+((6-6*Delta_coef_cp[0])*X_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1,'phi',1)+((4-4*Delta_coef_cp[0])*diff(X_coef_cp[1],'phi',1)-2*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*(diff(Y_coef_cp[1],'chi',1))**2+((10*Delta_coef_cp[0]-10)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(4*Delta_coef_cp[0]-4)*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1)+2*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',3)+(4*Delta_coef_cp[0]-4)*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',2,'phi',1)+2*Y_coef_cp[1]**2*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',2))*dl_p*kap_p**2
        )*tau_p
        +((1-Delta_coef_cp[0])*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',2)+(1-Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)+(Delta_coef_cp[0]-1)*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(Delta_coef_cp[0]-1)*Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1))**2)*kap_p*diff(kap_p,'phi',2)+((2*Delta_coef_cp[0]-2)*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',2)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1))**2)*(diff(kap_p,'phi',1))**2+(((4*Delta_coef_cp[0]-4)*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(4*Delta_coef_cp[0]-4)*(diff(X_coef_cp[1],'chi',1))**2)*diff(Y_coef_cp[1],'phi',1)+(4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',3)+(4-4*Delta_coef_cp[0])*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',2,'phi',1)+((4-4*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(8-8*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)-X_coef_cp[1]**2*diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',2)+(4-4*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1,'phi',1)+((4-4*Delta_coef_cp[0])*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2)-X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',3)+(4*Delta_coef_cp[0]-4)*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2,'phi',1)+((4*Delta_coef_cp[0]-4)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)+X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',2)+(4*Delta_coef_cp[0]-4)*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',1)+Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*(diff(X_coef_cp[1],'chi',1))**2)*kap_p*diff(kap_p,'phi',1)+(((3-3*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(3-3*Delta_coef_cp[0])*(diff(X_coef_cp[1],'chi',1))**2)*diff(Y_coef_cp[1],'phi',2)+((6-6*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',3)+(6-6*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2,'phi',1)+((6-6*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],'chi',1)-2*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',2)+(6-6*Delta_coef_cp[0])*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',1)-2*diff(Delta_coef_cp[0],'phi',1)*(diff(X_coef_cp[1],'chi',1))**2)*diff(Y_coef_cp[1],'phi',1)+(3*Delta_coef_cp[0]-3)*iota_coef[0]**2*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',4)+(6*Delta_coef_cp[0]-6)*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',3,'phi',1)+((6*Delta_coef_cp[0]-6)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(9*Delta_coef_cp[0]-9)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)+2*iota_coef[0]*X_coef_cp[1]**2*diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',3)+(3*Delta_coef_cp[0]-3)*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',2,'phi',2)+((6*Delta_coef_cp[0]-6)*X_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(12*Delta_coef_cp[0]-12)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)+2*X_coef_cp[1]**2*diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',2,'phi',1)+((2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(X_coef_cp[1],'phi',2)+((6*Delta_coef_cp[0]-6)*iota_coef[0]*diff(X_coef_cp[1],'chi',1)+X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'phi',1)+(1-Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1)+(3*Delta_coef_cp[0]-3)*iota_coef[0]**2*(diff(X_coef_cp[1],'chi',1))**2+3*iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',2)+(3*Delta_coef_cp[0]-3)*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1,'phi',2)+((6*Delta_coef_cp[0]-6)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)+(6-6*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+2*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1,'phi',1)+((3*Delta_coef_cp[0]-3)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',2)+2*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)+(7-7*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',3)+(8-8*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2,'phi',1)+((3-3*Delta_coef_cp[0])*iota_coef[0]**2*diff(X_coef_cp[1],'chi',1)-3*iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',2)+(1-Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',2)-X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1,'phi',1))*diff(Y_coef_cp[1],'chi',1)+(Delta_coef_cp[0]-1)*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)*diff(X_coef_cp[1],'phi',2)+Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',2)*diff(X_coef_cp[1],'phi',1)+(3-3*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',4)+(6-6*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',3,'phi',1)+((2-2*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)-2*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',3)+(3-3*Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2,'phi',2)+((4-4*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)-2*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',2,'phi',1)+(Delta_coef_cp[0]-1)*iota_coef[0]**2*Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',2))**2+(2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1)*diff(X_coef_cp[1],'chi',2)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',2)-Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',1))*kap_p**2
    )
    coef_B_psi_dphi_1_dchi_0_all_but_Y_const_5 = ((2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1))**2)*dl_p*kap_p
    coef_B_psi_dphi_1_dchi_0_all_but_Y_const_6 = (
        ((2-2*Delta_coef_cp[0])*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',2)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1))**2)*dl_p*kap_p
    )
    coef_B_psi_dphi_1_dchi_0_all_but_Y_const_7 = (
        (4*Delta_coef_cp[0]-4)*diff(B_denom_coef_c[1],'chi',1)
        -2*B_denom_coef_c[0]*diff(Delta_coef_cp[1],'chi',1)
        -(Delta_coef_cp[0]-1)*diff(B_denom_coef_c[1],'chi',1)
    )
    coef_B_psi_dphi_1_dchi_0_all_but_Y = lambda n_eval : (
        (
            -(
                coef_B_psi_dphi_1_dchi_0_all_but_Y_const_3*n_eval
                +coef_B_psi_dphi_1_dchi_0_all_but_Y_const_4
            )/denom_e(n_eval)
        )
        +(
            coef_B_psi_dphi_1_dchi_0_all_but_Y_const_5*n_eval
            +coef_B_psi_dphi_1_dchi_0_all_but_Y_const_6
        )/denom_a(n_eval)
        +coef_B_psi_dphi_1_dchi_0_all_but_Y_const_7/n_eval
    )

    # --------------------------------------------------------------------------
    coef_B_psi_dphi_1_dchi_0_in_Y_RHS_const_1 = (
        B_alpha_coef[0]*X_coef_cp[1]*(diff(Y_coef_cp[1],'chi',1))**2-B_alpha_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)
    )
    coef_B_psi_dphi_1_dchi_0_in_Y_RHS_const_2 = (
        -B_alpha_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',2)-B_alpha_coef[0]*X_coef_cp[1]*(diff(Y_coef_cp[1],'chi',1))**2+B_alpha_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)+B_alpha_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',2)
    )
    coef_B_psi_dphi_1_dchi_0_in_Y_RHS = lambda n_eval : -(
        coef_B_psi_dphi_1_dchi_0_in_Y_RHS_const_1*n_eval
        +coef_B_psi_dphi_1_dchi_0_in_Y_RHS_const_2
    )/denom_d(n_eval)

    # --------------------------------------------------------------------------
    coef_B_psi_dphi_1_dchi_1_all_but_Y_const_1 = (
        (4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*(diff(Y_coef_cp[1],'chi',1))**2+(4*Delta_coef_cp[0]-4)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)
    )*dl_p*kap_p**2
    coef_B_psi_dphi_1_dchi_1_all_but_Y_const_2 = (
        ((4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1))**2)*kap_p*diff(kap_p,'phi',1)+((6-6*Delta_coef_cp[0])*iota_coef[0]*(diff(X_coef_cp[1],'chi',1))**2*diff(Y_coef_cp[1],'phi',1)+(6*Delta_coef_cp[0]-6)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',2)+(6*Delta_coef_cp[0]-6)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1,'phi',1)+((6*Delta_coef_cp[0]-6)*iota_coef[0]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)+2*iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1)+(6-6*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',2)+(6-6*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',1)-2*iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*(diff(X_coef_cp[1],'chi',1))**2)*kap_p**2
        +coef_B_psi_dphi_1_dchi_1_all_but_Y_const_1*tau_p
    )
    coef_B_psi_dphi_1_dchi_1_all_but_Y_const_3 = (
        ((2*Delta_coef_cp[0]-2)*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',1))*dl_p*kap_p**2*diff(tau_p,'phi',1)
        +(
            ((2-2*Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',1))*dl_p*kap_p*diff(kap_p,'phi',1)
            +(((2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)+(6-6*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'phi',1)+(8*Delta_coef_cp[0]-8)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',2)+(4*Delta_coef_cp[0]-4)*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',1,'phi',1)+(6*Delta_coef_cp[0]-6)*iota_coef[0]*X_coef_cp[1]*(diff(Y_coef_cp[1],'chi',1))**2+((4*Delta_coef_cp[0]-4)*Y_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(6-6*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)+2*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',1)+(8-8*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',2)+(4-4*Delta_coef_cp[0])*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',1,'phi',1)-2*Y_coef_cp[1]**2*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*dl_p*kap_p**2
        )*tau_p
        +((Delta_coef_cp[0]-1)*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',1)+(1-Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1))*kap_p*diff(kap_p,'phi',2)+((2-2*Delta_coef_cp[0])*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',1)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1))*(diff(kap_p,'phi',1))**2+((4-4*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'phi',1)+(8*Delta_coef_cp[0]-8)*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',2)+(4*Delta_coef_cp[0]-4)*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',1,'phi',1)+((4*Delta_coef_cp[0]-4)*X_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)+X_coef_cp[1]**2*diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',1)+(8-8*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(4-4*Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1)+(4-4*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1))**2-X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*kap_p*diff(kap_p,'phi',1)+((3*Delta_coef_cp[0]-3)*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'phi',2)+((12*Delta_coef_cp[0]-12)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(6*Delta_coef_cp[0]-6)*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1)+(6*Delta_coef_cp[0]-6)*iota_coef[0]*(diff(X_coef_cp[1],'chi',1))**2+2*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'phi',1)+(9-9*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',3)+(12-12*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',2,'phi',1)+((12-12*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(15-15*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)-4*iota_coef[0]*X_coef_cp[1]**2*diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',2)+(3-3*Delta_coef_cp[0])*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',1,'phi',2)+((6-6*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(6-6*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)-2*X_coef_cp[1]**2*diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',1,'phi',1)+((2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'phi',2)+((6-6*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],'chi',1)-X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'phi',1)+(10*Delta_coef_cp[0]-10)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1)-iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1)+(1-Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',2)-Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)+(9*Delta_coef_cp[0]-9)*iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',3)+(12*Delta_coef_cp[0]-12)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2,'phi',1)+((5*Delta_coef_cp[0]-5)*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)+4*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',2)+(3*Delta_coef_cp[0]-3)*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',2)+((4*Delta_coef_cp[0]-4)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)+2*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',1,'phi',1)+iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*(diff(X_coef_cp[1],'chi',1))**2)*kap_p**2
    )
    coef_B_psi_dphi_1_dchi_1_all_but_Y_const_4 = (
        ((2*Delta_coef_cp[0]-2)*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',1)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1))*dl_p*kap_p
    )
    coef_B_psi_dphi_1_dchi_1_all_but_Y_const_5 = (
        (Delta_coef_cp[0]-1)*B_denom_coef_c[1]-(4*Delta_coef_cp[0]-4)*B_denom_coef_c[1]
    )
    coef_B_psi_dphi_1_dchi_1_all_but_Y = lambda n_eval : (
        (
            coef_B_psi_dphi_1_dchi_1_all_but_Y_const_2*n_eval
            + coef_B_psi_dphi_1_dchi_1_all_but_Y_const_3
        )/denom_e(n_eval)
        -coef_B_psi_dphi_1_dchi_1_all_but_Y_const_4/denom_a(n_eval)
        +coef_B_psi_dphi_1_dchi_1_all_but_Y_const_5/(n_eval**2-n_eval)
        -(2*B_denom_coef_c[0]*Delta_coef_cp[1])/n_eval
    )

    # --------------------------------------------------------------------------
    coef_B_psi_dphi_1_dchi_1_in_Y_RHS_const = (
        B_alpha_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)-B_alpha_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',1)
    )
    coef_B_psi_dphi_1_dchi_1_in_Y_RHS = lambda n_eval : coef_B_psi_dphi_1_dchi_1_in_Y_RHS_const/denom_d(n_eval)

    # --------------------------------------------------------------------------
    coef_B_psi_dphi_1_dchi_2_all_but_Y_const_1 = (
        (3*Delta_coef_cp[0]-3)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)+(3-3*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1))**2
    )*kap_p
    coef_B_psi_dphi_1_dchi_2_all_but_Y_const_2 = (
        (
            (4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)+(4-4*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',1)
        )*dl_p*kap_p*tau_p
        +((4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',1)+(4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1))*diff(kap_p,'phi',1)+((6*Delta_coef_cp[0]-6)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'phi',1)+(9-9*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',2)+(6-6*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',1,'phi',1)+((6-6*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(3-3*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)-2*iota_coef[0]*X_coef_cp[1]**2*diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',1)+(9*Delta_coef_cp[0]-9)*iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(6*Delta_coef_cp[0]-6)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1)+(3*Delta_coef_cp[0]-3)*iota_coef[0]**2*Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1))**2+2*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*kap_p
    )
    coef_B_psi_dphi_1_dchi_2_all_but_Y = lambda n_eval : (
        coef_B_psi_dphi_1_dchi_2_all_but_Y_const_1*n_eval
        +coef_B_psi_dphi_1_dchi_2_all_but_Y_const_2
    )/denom_b(n_eval)

    # --------------------------------------------------------------------------
    coef_B_psi_dphi_1_dchi_2_in_Y_RHS = lambda n_eval : 0

    # --------------------------------------------------------------------------
    coef_B_psi_dphi_1_dchi_3_all_but_Y_const = (
        (3*Delta_coef_cp[0]-3)*iota_coef[0]**2*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',1)+(3-3*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)
    )
    coef_B_psi_dphi_1_dchi_3_all_but_Y = lambda n_eval : -coef_B_psi_dphi_1_dchi_3_all_but_Y_const/denom_c(n_eval)

    # --------------------------------------------------------------------------
    coef_B_psi_dphi_1_dchi_3_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_1_dchi_4_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_1_dchi_4_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_1_dchi_5_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_1_dchi_5_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_1_dchi_6_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_1_dchi_6_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_1_dchi_7_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_1_dchi_7_in_Y_RHS = lambda n_eval : 0

    # --------------------------------------------------------------------------
    coef_B_psi_dphi_2_dchi_0_all_but_Y_const_1 = (
        ((2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1))**2)*diff(kap_p,'phi',1)+((3*Delta_coef_cp[0]-3)*(diff(X_coef_cp[1],'chi',1))**2*diff(Y_coef_cp[1],'phi',1)+(3-3*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',2)+(3-3*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1,'phi',1)+((3-3*Delta_coef_cp[0])*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)-X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1)+(3*Delta_coef_cp[0]-3)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',2)+(3*Delta_coef_cp[0]-3)*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',1)+Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*(diff(X_coef_cp[1],'chi',1))**2)*kap_p
    )
    coef_B_psi_dphi_2_dchi_0_all_but_Y_const_2 = (
        (2*Delta_coef_cp[0]-2)*X_coef_cp[1]*(diff(Y_coef_cp[1],'chi',1))**2+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)
    )*dl_p*kap_p*tau_p + coef_B_psi_dphi_2_dchi_0_all_but_Y_const_1
    coef_B_psi_dphi_2_dchi_0_all_but_Y_const_3 = (
        (
            ((2-2*Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',2)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*(diff(Y_coef_cp[1],'chi',1))**2+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',2))*dl_p*kap_p
        )*tau_p
        +((2-2*Delta_coef_cp[0])*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',2)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1))**2)*diff(kap_p,'phi',1)+(((3-3*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(3-3*Delta_coef_cp[0])*(diff(X_coef_cp[1],'chi',1))**2)*diff(Y_coef_cp[1],'phi',1)+(3*Delta_coef_cp[0]-3)*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',3)+(3*Delta_coef_cp[0]-3)*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',2,'phi',1)+((3*Delta_coef_cp[0]-3)*X_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(6*Delta_coef_cp[0]-6)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)+X_coef_cp[1]**2*diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',2)+(3*Delta_coef_cp[0]-3)*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1,'phi',1)+((3*Delta_coef_cp[0]-3)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)+(3-3*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1)+(3-3*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',3)+(3-3*Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2,'phi',1)+((3-3*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)-X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1))*diff(X_coef_cp[1],'chi',2)+(3-3*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',1)-Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*(diff(X_coef_cp[1],'chi',1))**2)*kap_p
    )
    coef_B_psi_dphi_2_dchi_0_all_but_Y = lambda n_eval : -(
        coef_B_psi_dphi_2_dchi_0_all_but_Y_const_2*n_eval
        + coef_B_psi_dphi_2_dchi_0_all_but_Y_const_3
    )/denom_b(n_eval)

    # --------------------------------------------------------------------------
    coef_B_psi_dphi_2_dchi_0_in_Y_RHS = lambda n_eval : 0

    # --------------------------------------------------------------------------
    coef_B_psi_dphi_2_dchi_1_all_but_Y_const_1 = (
        ((3*Delta_coef_cp[0]-3)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)+(3-3*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1))**2)*kap_p
    )
    coef_B_psi_dphi_2_dchi_1_all_but_Y_const_2 = (
        ((2*Delta_coef_cp[0]-2)*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',1))*dl_p*kap_p*tau_p
        +((2*Delta_coef_cp[0]-2)*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',1)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1))*diff(kap_p,'phi',1)+((3*Delta_coef_cp[0]-3)*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'phi',1)+(6-6*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',2)+(3-3*Delta_coef_cp[0])*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',1,'phi',1)+((3-3*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+(3-3*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)-X_coef_cp[1]**2*diff(Delta_coef_cp[0],'phi',1))*diff(Y_coef_cp[1],'chi',1)+(6*Delta_coef_cp[0]-6)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(3*Delta_coef_cp[0]-3)*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1)+(3*Delta_coef_cp[0]-3)*iota_coef[0]*Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1))**2+X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',1))*kap_p
    )
    coef_B_psi_dphi_2_dchi_1_all_but_Y = lambda n_eval : (
        coef_B_psi_dphi_2_dchi_1_all_but_Y_const_1*n_eval
        + coef_B_psi_dphi_2_dchi_1_all_but_Y_const_2
    )/denom_b(n_eval)

    # --------------------------------------------------------------------------
    coef_B_psi_dphi_2_dchi_1_in_Y_RHS = lambda n_eval : 0

    # --------------------------------------------------------------------------
    coef_B_psi_dphi_2_dchi_2_all_but_Y_const = (
        (3*Delta_coef_cp[0]-3)*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',1)+(3-3*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)
    )
    coef_B_psi_dphi_2_dchi_2_all_but_Y = lambda n_eval : -coef_B_psi_dphi_2_dchi_2_all_but_Y_const/denom_c(n_eval)
    coef_B_psi_dphi_2_dchi_2_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_2_dchi_3_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_2_dchi_3_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_2_dchi_4_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_2_dchi_4_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_2_dchi_5_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_2_dchi_5_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_2_dchi_6_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_2_dchi_6_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_2_dchi_7_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_2_dchi_7_in_Y_RHS = lambda n_eval : 0

    # --------------------------------------------------------------------------
    coef_B_psi_dphi_3_dchi_0_all_but_Y_const_1 = (
        (Delta_coef_cp[0]-1)*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)+(1-Delta_coef_cp[0])*Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1))**2
    )
    coef_B_psi_dphi_3_dchi_0_all_but_Y_const_2 = (
        (1-Delta_coef_cp[0])*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',2)+(1-Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)+(Delta_coef_cp[0]-1)*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(Delta_coef_cp[0]-1)*Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1))**2
    )
    coef_B_psi_dphi_3_dchi_0_all_but_Y = lambda n_eval : (
        coef_B_psi_dphi_3_dchi_0_all_but_Y_const_1*n_eval
        + coef_B_psi_dphi_3_dchi_0_all_but_Y_const_2
    )/denom_c(n_eval)

    # --------------------------------------------------------------------------
    coef_B_psi_dphi_3_dchi_0_in_Y_RHS = lambda n_eval : 0

    # --------------------------------------------------------------------------
    coef_B_psi_dphi_3_dchi_1_all_but_Y_const = (
        (Delta_coef_cp[0]-1)*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',1)+(1-Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)
    )
    coef_B_psi_dphi_3_dchi_1_all_but_Y = lambda n_eval : -coef_B_psi_dphi_3_dchi_1_all_but_Y_const/denom_c(n_eval)
    coef_B_psi_dphi_3_dchi_1_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_3_dchi_2_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_3_dchi_2_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_3_dchi_3_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_3_dchi_3_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_3_dchi_4_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_3_dchi_4_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_3_dchi_5_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_3_dchi_5_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_3_dchi_6_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_3_dchi_6_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_3_dchi_7_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_3_dchi_7_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_4_dchi_0_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_4_dchi_0_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_4_dchi_1_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_4_dchi_1_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_4_dchi_2_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_4_dchi_2_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_4_dchi_3_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_4_dchi_3_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_4_dchi_4_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_4_dchi_4_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_4_dchi_5_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_4_dchi_5_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_4_dchi_6_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_4_dchi_6_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_4_dchi_7_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_4_dchi_7_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_5_dchi_0_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_5_dchi_0_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_5_dchi_1_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_5_dchi_1_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_5_dchi_2_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_5_dchi_2_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_5_dchi_3_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_5_dchi_3_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_5_dchi_4_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_5_dchi_4_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_5_dchi_5_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_5_dchi_5_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_5_dchi_6_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_5_dchi_6_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_5_dchi_7_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_5_dchi_7_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_6_dchi_0_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_6_dchi_0_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_6_dchi_1_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_6_dchi_1_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_6_dchi_2_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_6_dchi_2_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_6_dchi_3_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_6_dchi_3_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_6_dchi_4_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_6_dchi_4_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_6_dchi_5_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_6_dchi_5_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_6_dchi_6_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_6_dchi_6_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_6_dchi_7_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_6_dchi_7_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_7_dchi_0_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_7_dchi_0_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_7_dchi_1_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_7_dchi_1_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_7_dchi_2_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_7_dchi_2_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_7_dchi_3_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_7_dchi_3_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_7_dchi_4_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_7_dchi_4_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_7_dchi_5_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_7_dchi_5_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_7_dchi_6_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_7_dchi_6_in_Y_RHS = lambda n_eval : 0

    coef_B_psi_dphi_7_dchi_7_all_but_Y = lambda n_eval : 0
    coef_B_psi_dphi_7_dchi_7_in_Y_RHS = lambda n_eval : 0

    out_dict = {
        "coef_B_psi_dphi_0_dchi_0_all_but_Y": coef_B_psi_dphi_0_dchi_0_all_but_Y,
        "coef_B_psi_dphi_0_dchi_0_in_Y_RHS": coef_B_psi_dphi_0_dchi_0_in_Y_RHS,

        "coef_B_psi_dphi_0_dchi_1_all_but_Y": coef_B_psi_dphi_0_dchi_1_all_but_Y,
        "coef_B_psi_dphi_0_dchi_1_in_Y_RHS": coef_B_psi_dphi_0_dchi_1_in_Y_RHS,

        "coef_B_psi_dphi_0_dchi_2_all_but_Y": coef_B_psi_dphi_0_dchi_2_all_but_Y,
        "coef_B_psi_dphi_0_dchi_2_in_Y_RHS": coef_B_psi_dphi_0_dchi_2_in_Y_RHS,

        "coef_B_psi_dphi_0_dchi_3_all_but_Y": coef_B_psi_dphi_0_dchi_3_all_but_Y,
        "coef_B_psi_dphi_0_dchi_3_in_Y_RHS": coef_B_psi_dphi_0_dchi_3_in_Y_RHS,

        "coef_B_psi_dphi_0_dchi_4_all_but_Y": coef_B_psi_dphi_0_dchi_4_all_but_Y,
        "coef_B_psi_dphi_0_dchi_4_in_Y_RHS": coef_B_psi_dphi_0_dchi_4_in_Y_RHS,

        "coef_B_psi_dphi_0_dchi_5_all_but_Y": coef_B_psi_dphi_0_dchi_5_all_but_Y,
        "coef_B_psi_dphi_0_dchi_5_in_Y_RHS": coef_B_psi_dphi_0_dchi_5_in_Y_RHS,

        "coef_B_psi_dphi_0_dchi_6_all_but_Y": coef_B_psi_dphi_0_dchi_6_all_but_Y,
        "coef_B_psi_dphi_0_dchi_6_in_Y_RHS": coef_B_psi_dphi_0_dchi_6_in_Y_RHS,

        "coef_B_psi_dphi_0_dchi_7_all_but_Y": coef_B_psi_dphi_0_dchi_7_all_but_Y,
        "coef_B_psi_dphi_0_dchi_7_in_Y_RHS": coef_B_psi_dphi_0_dchi_7_in_Y_RHS,

        "coef_B_psi_dphi_1_dchi_0_all_but_Y": coef_B_psi_dphi_1_dchi_0_all_but_Y,
        "coef_B_psi_dphi_1_dchi_0_in_Y_RHS": coef_B_psi_dphi_1_dchi_0_in_Y_RHS,

        "coef_B_psi_dphi_1_dchi_1_all_but_Y": coef_B_psi_dphi_1_dchi_1_all_but_Y,
        "coef_B_psi_dphi_1_dchi_1_in_Y_RHS": coef_B_psi_dphi_1_dchi_1_in_Y_RHS,

        "coef_B_psi_dphi_1_dchi_2_all_but_Y": coef_B_psi_dphi_1_dchi_2_all_but_Y,
        "coef_B_psi_dphi_1_dchi_2_in_Y_RHS": coef_B_psi_dphi_1_dchi_2_in_Y_RHS,

        "coef_B_psi_dphi_1_dchi_3_all_but_Y": coef_B_psi_dphi_1_dchi_3_all_but_Y,
        "coef_B_psi_dphi_1_dchi_3_in_Y_RHS": coef_B_psi_dphi_1_dchi_3_in_Y_RHS,

        "coef_B_psi_dphi_1_dchi_4_all_but_Y": coef_B_psi_dphi_1_dchi_4_all_but_Y,
        "coef_B_psi_dphi_1_dchi_4_in_Y_RHS": coef_B_psi_dphi_1_dchi_4_in_Y_RHS,

        "coef_B_psi_dphi_1_dchi_5_all_but_Y": coef_B_psi_dphi_1_dchi_5_all_but_Y,
        "coef_B_psi_dphi_1_dchi_5_in_Y_RHS": coef_B_psi_dphi_1_dchi_5_in_Y_RHS,

        "coef_B_psi_dphi_1_dchi_6_all_but_Y": coef_B_psi_dphi_1_dchi_6_all_but_Y,
        "coef_B_psi_dphi_1_dchi_6_in_Y_RHS": coef_B_psi_dphi_1_dchi_6_in_Y_RHS,

        "coef_B_psi_dphi_1_dchi_7_all_but_Y": coef_B_psi_dphi_1_dchi_7_all_but_Y,
        "coef_B_psi_dphi_1_dchi_7_in_Y_RHS": coef_B_psi_dphi_1_dchi_7_in_Y_RHS,

        "coef_B_psi_dphi_2_dchi_0_all_but_Y": coef_B_psi_dphi_2_dchi_0_all_but_Y,
        "coef_B_psi_dphi_2_dchi_0_in_Y_RHS": coef_B_psi_dphi_2_dchi_0_in_Y_RHS,

        "coef_B_psi_dphi_2_dchi_1_all_but_Y": coef_B_psi_dphi_2_dchi_1_all_but_Y,
        "coef_B_psi_dphi_2_dchi_1_in_Y_RHS": coef_B_psi_dphi_2_dchi_1_in_Y_RHS,

        "coef_B_psi_dphi_2_dchi_2_all_but_Y": coef_B_psi_dphi_2_dchi_2_all_but_Y,
        "coef_B_psi_dphi_2_dchi_2_in_Y_RHS": coef_B_psi_dphi_2_dchi_2_in_Y_RHS,

        "coef_B_psi_dphi_2_dchi_3_all_but_Y": coef_B_psi_dphi_2_dchi_3_all_but_Y,
        "coef_B_psi_dphi_2_dchi_3_in_Y_RHS": coef_B_psi_dphi_2_dchi_3_in_Y_RHS,

        "coef_B_psi_dphi_2_dchi_4_all_but_Y": coef_B_psi_dphi_2_dchi_4_all_but_Y,
        "coef_B_psi_dphi_2_dchi_4_in_Y_RHS": coef_B_psi_dphi_2_dchi_4_in_Y_RHS,

        "coef_B_psi_dphi_2_dchi_5_all_but_Y": coef_B_psi_dphi_2_dchi_5_all_but_Y,
        "coef_B_psi_dphi_2_dchi_5_in_Y_RHS": coef_B_psi_dphi_2_dchi_5_in_Y_RHS,

        "coef_B_psi_dphi_2_dchi_6_all_but_Y": coef_B_psi_dphi_2_dchi_6_all_but_Y,
        "coef_B_psi_dphi_2_dchi_6_in_Y_RHS": coef_B_psi_dphi_2_dchi_6_in_Y_RHS,

        "coef_B_psi_dphi_2_dchi_7_all_but_Y": coef_B_psi_dphi_2_dchi_7_all_but_Y,
        "coef_B_psi_dphi_2_dchi_7_in_Y_RHS": coef_B_psi_dphi_2_dchi_7_in_Y_RHS,

        "coef_B_psi_dphi_3_dchi_0_all_but_Y": coef_B_psi_dphi_3_dchi_0_all_but_Y,
        "coef_B_psi_dphi_3_dchi_0_in_Y_RHS": coef_B_psi_dphi_3_dchi_0_in_Y_RHS,

        "coef_B_psi_dphi_3_dchi_1_all_but_Y": coef_B_psi_dphi_3_dchi_1_all_but_Y,
        "coef_B_psi_dphi_3_dchi_1_in_Y_RHS": coef_B_psi_dphi_3_dchi_1_in_Y_RHS,

        "coef_B_psi_dphi_3_dchi_2_all_but_Y": coef_B_psi_dphi_3_dchi_2_all_but_Y,
        "coef_B_psi_dphi_3_dchi_2_in_Y_RHS": coef_B_psi_dphi_3_dchi_2_in_Y_RHS,

        "coef_B_psi_dphi_3_dchi_3_all_but_Y": coef_B_psi_dphi_3_dchi_3_all_but_Y,
        "coef_B_psi_dphi_3_dchi_3_in_Y_RHS": coef_B_psi_dphi_3_dchi_3_in_Y_RHS,

        "coef_B_psi_dphi_3_dchi_4_all_but_Y": coef_B_psi_dphi_3_dchi_4_all_but_Y,
        "coef_B_psi_dphi_3_dchi_4_in_Y_RHS": coef_B_psi_dphi_3_dchi_4_in_Y_RHS,

        "coef_B_psi_dphi_3_dchi_5_all_but_Y": coef_B_psi_dphi_3_dchi_5_all_but_Y,
        "coef_B_psi_dphi_3_dchi_5_in_Y_RHS": coef_B_psi_dphi_3_dchi_5_in_Y_RHS,

        "coef_B_psi_dphi_3_dchi_6_all_but_Y": coef_B_psi_dphi_3_dchi_6_all_but_Y,
        "coef_B_psi_dphi_3_dchi_6_in_Y_RHS": coef_B_psi_dphi_3_dchi_6_in_Y_RHS,

        "coef_B_psi_dphi_3_dchi_7_all_but_Y": coef_B_psi_dphi_3_dchi_7_all_but_Y,
        "coef_B_psi_dphi_3_dchi_7_in_Y_RHS": coef_B_psi_dphi_3_dchi_7_in_Y_RHS,

        "coef_B_psi_dphi_4_dchi_0_all_but_Y": coef_B_psi_dphi_4_dchi_0_all_but_Y,
        "coef_B_psi_dphi_4_dchi_0_in_Y_RHS": coef_B_psi_dphi_4_dchi_0_in_Y_RHS,

        "coef_B_psi_dphi_4_dchi_1_all_but_Y": coef_B_psi_dphi_4_dchi_1_all_but_Y,
        "coef_B_psi_dphi_4_dchi_1_in_Y_RHS": coef_B_psi_dphi_4_dchi_1_in_Y_RHS,

        "coef_B_psi_dphi_4_dchi_2_all_but_Y": coef_B_psi_dphi_4_dchi_2_all_but_Y,
        "coef_B_psi_dphi_4_dchi_2_in_Y_RHS": coef_B_psi_dphi_4_dchi_2_in_Y_RHS,

        "coef_B_psi_dphi_4_dchi_3_all_but_Y": coef_B_psi_dphi_4_dchi_3_all_but_Y,
        "coef_B_psi_dphi_4_dchi_3_in_Y_RHS": coef_B_psi_dphi_4_dchi_3_in_Y_RHS,

        "coef_B_psi_dphi_4_dchi_4_all_but_Y": coef_B_psi_dphi_4_dchi_4_all_but_Y,
        "coef_B_psi_dphi_4_dchi_4_in_Y_RHS": coef_B_psi_dphi_4_dchi_4_in_Y_RHS,

        "coef_B_psi_dphi_4_dchi_5_all_but_Y": coef_B_psi_dphi_4_dchi_5_all_but_Y,
        "coef_B_psi_dphi_4_dchi_5_in_Y_RHS": coef_B_psi_dphi_4_dchi_5_in_Y_RHS,

        "coef_B_psi_dphi_4_dchi_6_all_but_Y": coef_B_psi_dphi_4_dchi_6_all_but_Y,
        "coef_B_psi_dphi_4_dchi_6_in_Y_RHS": coef_B_psi_dphi_4_dchi_6_in_Y_RHS,

        "coef_B_psi_dphi_4_dchi_7_all_but_Y": coef_B_psi_dphi_4_dchi_7_all_but_Y,
        "coef_B_psi_dphi_4_dchi_7_in_Y_RHS": coef_B_psi_dphi_4_dchi_7_in_Y_RHS,

        "coef_B_psi_dphi_5_dchi_0_all_but_Y": coef_B_psi_dphi_5_dchi_0_all_but_Y,
        "coef_B_psi_dphi_5_dchi_0_in_Y_RHS": coef_B_psi_dphi_5_dchi_0_in_Y_RHS,

        "coef_B_psi_dphi_5_dchi_1_all_but_Y": coef_B_psi_dphi_5_dchi_1_all_but_Y,
        "coef_B_psi_dphi_5_dchi_1_in_Y_RHS": coef_B_psi_dphi_5_dchi_1_in_Y_RHS,

        "coef_B_psi_dphi_5_dchi_2_all_but_Y": coef_B_psi_dphi_5_dchi_2_all_but_Y,
        "coef_B_psi_dphi_5_dchi_2_in_Y_RHS": coef_B_psi_dphi_5_dchi_2_in_Y_RHS,

        "coef_B_psi_dphi_5_dchi_3_all_but_Y": coef_B_psi_dphi_5_dchi_3_all_but_Y,
        "coef_B_psi_dphi_5_dchi_3_in_Y_RHS": coef_B_psi_dphi_5_dchi_3_in_Y_RHS,

        "coef_B_psi_dphi_5_dchi_4_all_but_Y": coef_B_psi_dphi_5_dchi_4_all_but_Y,
        "coef_B_psi_dphi_5_dchi_4_in_Y_RHS": coef_B_psi_dphi_5_dchi_4_in_Y_RHS,

        "coef_B_psi_dphi_5_dchi_5_all_but_Y": coef_B_psi_dphi_5_dchi_5_all_but_Y,
        "coef_B_psi_dphi_5_dchi_5_in_Y_RHS": coef_B_psi_dphi_5_dchi_5_in_Y_RHS,

        "coef_B_psi_dphi_5_dchi_6_all_but_Y": coef_B_psi_dphi_5_dchi_6_all_but_Y,
        "coef_B_psi_dphi_5_dchi_6_in_Y_RHS": coef_B_psi_dphi_5_dchi_6_in_Y_RHS,

        "coef_B_psi_dphi_5_dchi_7_all_but_Y": coef_B_psi_dphi_5_dchi_7_all_but_Y,
        "coef_B_psi_dphi_5_dchi_7_in_Y_RHS": coef_B_psi_dphi_5_dchi_7_in_Y_RHS,

        "coef_B_psi_dphi_6_dchi_0_all_but_Y": coef_B_psi_dphi_6_dchi_0_all_but_Y,
        "coef_B_psi_dphi_6_dchi_0_in_Y_RHS": coef_B_psi_dphi_6_dchi_0_in_Y_RHS,

        "coef_B_psi_dphi_6_dchi_1_all_but_Y": coef_B_psi_dphi_6_dchi_1_all_but_Y,
        "coef_B_psi_dphi_6_dchi_1_in_Y_RHS": coef_B_psi_dphi_6_dchi_1_in_Y_RHS,

        "coef_B_psi_dphi_6_dchi_2_all_but_Y": coef_B_psi_dphi_6_dchi_2_all_but_Y,
        "coef_B_psi_dphi_6_dchi_2_in_Y_RHS": coef_B_psi_dphi_6_dchi_2_in_Y_RHS,

        "coef_B_psi_dphi_6_dchi_3_all_but_Y": coef_B_psi_dphi_6_dchi_3_all_but_Y,
        "coef_B_psi_dphi_6_dchi_3_in_Y_RHS": coef_B_psi_dphi_6_dchi_3_in_Y_RHS,

        "coef_B_psi_dphi_6_dchi_4_all_but_Y": coef_B_psi_dphi_6_dchi_4_all_but_Y,
        "coef_B_psi_dphi_6_dchi_4_in_Y_RHS": coef_B_psi_dphi_6_dchi_4_in_Y_RHS,

        "coef_B_psi_dphi_6_dchi_5_all_but_Y": coef_B_psi_dphi_6_dchi_5_all_but_Y,
        "coef_B_psi_dphi_6_dchi_5_in_Y_RHS": coef_B_psi_dphi_6_dchi_5_in_Y_RHS,

        "coef_B_psi_dphi_6_dchi_6_all_but_Y": coef_B_psi_dphi_6_dchi_6_all_but_Y,
        "coef_B_psi_dphi_6_dchi_6_in_Y_RHS": coef_B_psi_dphi_6_dchi_6_in_Y_RHS,

        "coef_B_psi_dphi_6_dchi_7_all_but_Y": coef_B_psi_dphi_6_dchi_7_all_but_Y,
        "coef_B_psi_dphi_6_dchi_7_in_Y_RHS": coef_B_psi_dphi_6_dchi_7_in_Y_RHS,

        "coef_B_psi_dphi_7_dchi_0_all_but_Y": coef_B_psi_dphi_7_dchi_0_all_but_Y,
        "coef_B_psi_dphi_7_dchi_0_in_Y_RHS": coef_B_psi_dphi_7_dchi_0_in_Y_RHS,

        "coef_B_psi_dphi_7_dchi_1_all_but_Y": coef_B_psi_dphi_7_dchi_1_all_but_Y,
        "coef_B_psi_dphi_7_dchi_1_in_Y_RHS": coef_B_psi_dphi_7_dchi_1_in_Y_RHS,

        "coef_B_psi_dphi_7_dchi_2_all_but_Y": coef_B_psi_dphi_7_dchi_2_all_but_Y,
        "coef_B_psi_dphi_7_dchi_2_in_Y_RHS": coef_B_psi_dphi_7_dchi_2_in_Y_RHS,

        "coef_B_psi_dphi_7_dchi_3_all_but_Y": coef_B_psi_dphi_7_dchi_3_all_but_Y,
        "coef_B_psi_dphi_7_dchi_3_in_Y_RHS": coef_B_psi_dphi_7_dchi_3_in_Y_RHS,

        "coef_B_psi_dphi_7_dchi_4_all_but_Y": coef_B_psi_dphi_7_dchi_4_all_but_Y,
        "coef_B_psi_dphi_7_dchi_4_in_Y_RHS": coef_B_psi_dphi_7_dchi_4_in_Y_RHS,

        "coef_B_psi_dphi_7_dchi_5_all_but_Y": coef_B_psi_dphi_7_dchi_5_all_but_Y,
        "coef_B_psi_dphi_7_dchi_5_in_Y_RHS": coef_B_psi_dphi_7_dchi_5_in_Y_RHS,

        "coef_B_psi_dphi_7_dchi_6_all_but_Y": coef_B_psi_dphi_7_dchi_6_all_but_Y,
        "coef_B_psi_dphi_7_dchi_6_in_Y_RHS": coef_B_psi_dphi_7_dchi_6_in_Y_RHS,

        "coef_B_psi_dphi_7_dchi_7_all_but_Y": coef_B_psi_dphi_7_dchi_7_all_but_Y,
        "coef_B_psi_dphi_7_dchi_7_in_Y_RHS": coef_B_psi_dphi_7_dchi_7_in_Y_RHS,
        "tensor_fft_op_B_psi_in_all_but_Y": (
            # for tracking B_theta dependence
            lambda n_eval, to_tensor_fft_op_multi_dim : (
                to_tensor_fft_op_multi_dim(coef_B_psi_dphi_0_dchi_0_all_but_Y(n_eval), dphi=0, dchi=0)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_0_dchi_1_all_but_Y(n_eval), dphi=0, dchi=1)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_0_dchi_2_all_but_Y(n_eval), dphi=0, dchi=2)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_0_dchi_3_all_but_Y(n_eval), dphi=0, dchi=3)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_0_dchi_4_all_but_Y(n_eval), dphi=0, dchi=4)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_0_dchi_5_all_but_Y(n_eval), dphi=0, dchi=5)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_0_dchi_6_all_but_Y(n_eval), dphi=0, dchi=6)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_0_dchi_7_all_but_Y(n_eval), dphi=0, dchi=7)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_1_dchi_0_all_but_Y(n_eval), dphi=1, dchi=0)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_1_dchi_1_all_but_Y(n_eval), dphi=1, dchi=1)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_1_dchi_2_all_but_Y(n_eval), dphi=1, dchi=2)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_1_dchi_3_all_but_Y(n_eval), dphi=1, dchi=3)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_1_dchi_4_all_but_Y(n_eval), dphi=1, dchi=4)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_1_dchi_5_all_but_Y(n_eval), dphi=1, dchi=5)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_1_dchi_6_all_but_Y(n_eval), dphi=1, dchi=6)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_1_dchi_7_all_but_Y(n_eval), dphi=1, dchi=7)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_2_dchi_0_all_but_Y(n_eval), dphi=2, dchi=0)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_2_dchi_1_all_but_Y(n_eval), dphi=2, dchi=1)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_2_dchi_2_all_but_Y(n_eval), dphi=2, dchi=2)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_2_dchi_3_all_but_Y(n_eval), dphi=2, dchi=3)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_2_dchi_4_all_but_Y(n_eval), dphi=2, dchi=4)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_2_dchi_5_all_but_Y(n_eval), dphi=2, dchi=5)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_2_dchi_6_all_but_Y(n_eval), dphi=2, dchi=6)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_2_dchi_7_all_but_Y(n_eval), dphi=2, dchi=7)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_3_dchi_0_all_but_Y(n_eval), dphi=3, dchi=0)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_3_dchi_1_all_but_Y(n_eval), dphi=3, dchi=1)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_3_dchi_2_all_but_Y(n_eval), dphi=3, dchi=2)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_3_dchi_3_all_but_Y(n_eval), dphi=3, dchi=3)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_3_dchi_4_all_but_Y(n_eval), dphi=3, dchi=4)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_3_dchi_5_all_but_Y(n_eval), dphi=3, dchi=5)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_3_dchi_6_all_but_Y(n_eval), dphi=3, dchi=6)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_3_dchi_7_all_but_Y(n_eval), dphi=3, dchi=7)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_4_dchi_0_all_but_Y(n_eval), dphi=4, dchi=0)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_4_dchi_1_all_but_Y(n_eval), dphi=4, dchi=1)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_4_dchi_2_all_but_Y(n_eval), dphi=4, dchi=2)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_4_dchi_3_all_but_Y(n_eval), dphi=4, dchi=3)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_4_dchi_4_all_but_Y(n_eval), dphi=4, dchi=4)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_4_dchi_5_all_but_Y(n_eval), dphi=4, dchi=5)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_4_dchi_6_all_but_Y(n_eval), dphi=4, dchi=6)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_4_dchi_7_all_but_Y(n_eval), dphi=4, dchi=7)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_5_dchi_0_all_but_Y(n_eval), dphi=5, dchi=0)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_5_dchi_1_all_but_Y(n_eval), dphi=5, dchi=1)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_5_dchi_2_all_but_Y(n_eval), dphi=5, dchi=2)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_5_dchi_3_all_but_Y(n_eval), dphi=5, dchi=3)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_5_dchi_4_all_but_Y(n_eval), dphi=5, dchi=4)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_5_dchi_5_all_but_Y(n_eval), dphi=5, dchi=5)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_5_dchi_6_all_but_Y(n_eval), dphi=5, dchi=6)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_5_dchi_7_all_but_Y(n_eval), dphi=5, dchi=7)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_6_dchi_0_all_but_Y(n_eval), dphi=6, dchi=0)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_6_dchi_1_all_but_Y(n_eval), dphi=6, dchi=1)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_6_dchi_2_all_but_Y(n_eval), dphi=6, dchi=2)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_6_dchi_3_all_but_Y(n_eval), dphi=6, dchi=3)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_6_dchi_4_all_but_Y(n_eval), dphi=6, dchi=4)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_6_dchi_5_all_but_Y(n_eval), dphi=6, dchi=5)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_6_dchi_6_all_but_Y(n_eval), dphi=6, dchi=6)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_6_dchi_7_all_but_Y(n_eval), dphi=6, dchi=7)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_7_dchi_0_all_but_Y(n_eval), dphi=7, dchi=0)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_7_dchi_1_all_but_Y(n_eval), dphi=7, dchi=1)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_7_dchi_2_all_but_Y(n_eval), dphi=7, dchi=2)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_7_dchi_3_all_but_Y(n_eval), dphi=7, dchi=3)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_7_dchi_4_all_but_Y(n_eval), dphi=7, dchi=4)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_7_dchi_5_all_but_Y(n_eval), dphi=7, dchi=5)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_7_dchi_6_all_but_Y(n_eval), dphi=7, dchi=6)
                +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_7_dchi_7_all_but_Y(n_eval), dphi=7, dchi=7)
                +0
            )
        )
    }
    def tensor_fft_op_B_psi_in_all_but_Y_parallel(n_eval, to_tensor_fft_op_multi_dim):
        expr = lambda i,j: to_tensor_fft_op_multi_dim(
            out_dict['coef_B_psi_dphi_'+str(i)+'_dchi_'+str(j)+'_all_but_Y'](n_eval),
            dphi=i,
            dchi=j
        )
        out_list = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(expr)(i,j)
                for i in range(7)
                for j in range(7)
        )
        out = 0
        for a in out_list:
            out = out+a
        return(out)

    out_dict['tensor_fft_op_B_psi_in_all_but_Y_parallel'] = tensor_fft_op_B_psi_in_all_but_Y_parallel
    return(out_dict)

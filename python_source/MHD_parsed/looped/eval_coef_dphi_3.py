# Evaluates coefficient of diff(B_psi_dummy,phi,3) 
from math import floor, ceil
from math_utilities import *
import chiphifunc
def eval_coef_dphi_3(n, B_psi_coef_cp, B_theta_coef_cp, 
    Delta_coef_cp, p_perp_coef_cp,
    X_coef_cp, Y_coef_cp, Z_coef_cp, 
    iota_coef, dl_p,
    B_denom_coef_c, B_alpha_coef, 
    kap_p, tau_p):    
    (((Delta_coef_cp[0]-1)*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)+(1-Delta_coef_cp[0])*Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1))**2)*n)/(B_alpha_coef[0]*dl_p*kap_p*n**2-B_alpha_coef[0]*dl_p*kap_p*n)+((1-Delta_coef_cp[0])*X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',2)+(1-Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)+(Delta_coef_cp[0]-1)*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+(Delta_coef_cp[0]-1)*Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1))**2)/(B_alpha_coef[0]*dl_p*kap_p*n**2-B_alpha_coef[0]*dl_p*kap_p*n)
    return(out)

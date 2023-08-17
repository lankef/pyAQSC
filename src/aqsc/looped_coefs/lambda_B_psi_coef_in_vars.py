from aqsc.math_utilities import *
lambda_B_psi_nm2_in_X_n = lambda n_eval, X_coef_cp, Y_coef_cp, B_theta_coef_cp, B_psi_coef_cp, iota_coef, p_perp_coef_cp, Delta_coef_cp, B_denom_coef_c, B_alpha_coef, kap_p, dl_p, tau_p: \
    -(diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],False,1)-iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],True,2)-X_coef_cp[1]*diff(Y_coef_cp[1],True,1,False,1)-diff(X_coef_cp[1],False,1)*diff(Y_coef_cp[1],True,1)+iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2)+Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1))/(dl_p*kap_p*n_eval-dl_p*kap_p)
lambda_dphi_B_psi_nm2_in_X_n = lambda n_eval, X_coef_cp, Y_coef_cp, B_theta_coef_cp, B_psi_coef_cp, iota_coef, p_perp_coef_cp, Delta_coef_cp, B_denom_coef_c, B_alpha_coef, kap_p, dl_p, tau_p: \
    (X_coef_cp[1]*diff(Y_coef_cp[1],True,1)-Y_coef_cp[1]*diff(X_coef_cp[1],True,1))/(dl_p*kap_p*n_eval-dl_p*kap_p)
lambda_dchi_B_psi_nm2_in_X_n = lambda n_eval, X_coef_cp, Y_coef_cp, B_theta_coef_cp, B_psi_coef_cp, iota_coef, p_perp_coef_cp, Delta_coef_cp, B_denom_coef_c, B_alpha_coef, kap_p, dl_p, tau_p: \
    (iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],True,1)-iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1))/(dl_p*kap_p*n_eval-dl_p*kap_p)
lambda_B_psi_nm2_in_Z_n = lambda n_eval, X_coef_cp, Y_coef_cp, B_theta_coef_cp, B_psi_coef_cp, iota_coef, p_perp_coef_cp, Delta_coef_cp, B_denom_coef_c, B_alpha_coef, kap_p, dl_p, tau_p: \
    (X_coef_cp[1]*diff(Y_coef_cp[1],True,1)-Y_coef_cp[1]*diff(X_coef_cp[1],True,1))/(n_eval-1)
lambda_B_psi_nm2_in_p_n = lambda n_eval, X_coef_cp, Y_coef_cp, B_theta_coef_cp, B_psi_coef_cp, iota_coef, p_perp_coef_cp, Delta_coef_cp, B_denom_coef_c, B_alpha_coef, kap_p, dl_p, tau_p: \
    -(2*diff(Delta_coef_cp[0],False,1))/(B_alpha_coef[0]*B_denom_coef_c[0]*n_eval-B_alpha_coef[0]*B_denom_coef_c[0])
lambda_dphi_B_psi_nm2_in_p_n = lambda n_eval, X_coef_cp, Y_coef_cp, B_theta_coef_cp, B_psi_coef_cp, iota_coef, p_perp_coef_cp, Delta_coef_cp, B_denom_coef_c, B_alpha_coef, kap_p, dl_p, tau_p: \
    -(2*Delta_coef_cp[0]-2)/(B_alpha_coef[0]*B_denom_coef_c[0]*n_eval-B_alpha_coef[0]*B_denom_coef_c[0])
lambda_dchi_B_psi_nm2_in_p_n = lambda n_eval, X_coef_cp, Y_coef_cp, B_theta_coef_cp, B_psi_coef_cp, iota_coef, p_perp_coef_cp, Delta_coef_cp, B_denom_coef_c, B_alpha_coef, kap_p, dl_p, tau_p: \
    -((2*Delta_coef_cp[0]-2)*iota_coef[0])/(B_alpha_coef[0]*B_denom_coef_c[0]*n_eval-B_alpha_coef[0]*B_denom_coef_c[0])
lambda_B_psi_nm2_in_Delta_n = lambda n_eval, X_coef_cp, Y_coef_cp, B_theta_coef_cp, B_psi_coef_cp, iota_coef, p_perp_coef_cp, Delta_coef_cp, B_denom_coef_c, B_alpha_coef, kap_p, dl_p, tau_p: \
    (2*diff(Delta_coef_cp[0],False,1))/(B_alpha_coef[0]*n_eval-B_alpha_coef[0])
lambda_dphi_B_psi_nm2_in_Delta_n = lambda n_eval, X_coef_cp, Y_coef_cp, B_theta_coef_cp, B_psi_coef_cp, iota_coef, p_perp_coef_cp, Delta_coef_cp, B_denom_coef_c, B_alpha_coef, kap_p, dl_p, tau_p: \
    (2*Delta_coef_cp[0]-2)/(B_alpha_coef[0]*n_eval-B_alpha_coef[0])
lambda_dchi_B_psi_nm2_in_Delta_n = lambda n_eval, X_coef_cp, Y_coef_cp, B_theta_coef_cp, B_psi_coef_cp, iota_coef, p_perp_coef_cp, Delta_coef_cp, B_denom_coef_c, B_alpha_coef, kap_p, dl_p, tau_p: \
    ((2*Delta_coef_cp[0]-2)*iota_coef[0])/(B_alpha_coef[0]*n_eval-B_alpha_coef[0])

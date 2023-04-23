from chiphifunc import *
from chiphifunc_test_suite import *
from equilibrium import *
from looped_solver import *

import time

# Loading the circular axis case

print('Current devices,', jax.devices())

debug_path = '../test_data_eduardo/'

B_psi_coef_cp, B_theta_coef_cp, \
    Delta_coef_cp, p_perp_coef_cp,\
    X_coef_cp, Y_coef_cp, Z_coef_cp, \
    iota_coef, dl_p,\
    nfp, Xi_0, eta, \
    B_denom_coef_c, B_alpha_coef, \
    kap_p, tau_p = read_first_three_orders(
        debug_path+'circ/',
        R_array=[2,0,1,2,0.0001,0],
        Z_array=[1,2,0,0.001]
    )

# Testing the speed if the second order
print('----- Speed test, 2nd order -----')
print('===== Compile speed =====')
start_time = time.time()
solution2 = iterate_looped(
    n_unknown=2, max_freq=50, target_len_phi=1000,
    X_coef_cp=X_coef_cp,
    Y_coef_cp=Y_coef_cp,
    Z_coef_cp=Z_coef_cp,
    p_perp_coef_cp=p_perp_coef_cp,
    Delta_coef_cp=Delta_coef_cp,
    B_psi_coef_cp=B_psi_coef_cp,
    B_theta_coef_cp=B_theta_coef_cp,
    B_alpha_coef=B_alpha_coef,
    B_denom_coef_c=B_denom_coef_c,
    kap_p=kap_p,
    tau_p=tau_p,
    dl_p=dl_p,
    iota_coef=iota_coef,
    nfp=nfp,
)
print('Compile done, time elapsed(s):')
print('===== Run speed =====')
start_time = time.time()
solution2 = iterate_looped(
    n_unknown=2, max_freq=50, target_len_phi=1000,
    X_coef_cp=X_coef_cp,
    Y_coef_cp=Y_coef_cp,
    Z_coef_cp=Z_coef_cp,
    p_perp_coef_cp=p_perp_coef_cp,
    Delta_coef_cp=Delta_coef_cp,
    B_psi_coef_cp=B_psi_coef_cp,
    B_theta_coef_cp=B_theta_coef_cp,
    B_alpha_coef=B_alpha_coef,
    B_denom_coef_c=B_denom_coef_c,
    kap_p=kap_p,
    tau_p=tau_p,
    dl_p=dl_p,
    iota_coef=iota_coef,
    nfp=nfp,
)
print('Run done, time elapsed(s):',(time.time() - start_time))

# Testing the speed if the second order
print('----- Speed test, 3rd order -----')
print('===== Compile speed =====')
start_time = time.time()
solution3 = iterate_looped(
    n_unknown=3, max_freq=50, target_len_phi=1000,
    X_coef_cp=X_coef_cp,
    Y_coef_cp=Y_coef_cp,
    Z_coef_cp=Z_coef_cp,
    p_perp_coef_cp=p_perp_coef_cp,
    Delta_coef_cp=Delta_coef_cp,
    B_psi_coef_cp=B_psi_coef_cp,
    B_theta_coef_cp=B_theta_coef_cp,
    B_alpha_coef=B_alpha_coef,
    B_denom_coef_c=B_denom_coef_c,
    kap_p=kap_p,
    tau_p=tau_p,
    dl_p=dl_p,
    iota_coef=iota_coef,
    nfp=nfp,
)
print('Compile done, time elapsed(s):',(time.time() - start_time))
print('===== Run speed =====')
start_time = time.time()
solution3 = iterate_looped(
    n_unknown=3, max_freq=50, target_len_phi=1000,
    X_coef_cp=X_coef_cp,
    Y_coef_cp=Y_coef_cp,
    Z_coef_cp=Z_coef_cp,
    p_perp_coef_cp=p_perp_coef_cp,
    Delta_coef_cp=Delta_coef_cp,
    B_psi_coef_cp=B_psi_coef_cp,
    B_theta_coef_cp=B_theta_coef_cp,
    B_alpha_coef=B_alpha_coef,
    B_denom_coef_c=B_denom_coef_c,
    kap_p=kap_p,
    tau_p=tau_p,
    dl_p=dl_p,
    iota_coef=iota_coef,
    nfp=nfp,
)
print('Run done, time elapsed(s):',(time.time() - start_time))

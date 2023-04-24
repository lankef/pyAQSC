from chiphifunc import *
from chiphifunc_test_suite import *
from equilibrium import *
from looped_solver import *

import numpy as np

import time

import os
# os.environ['JAX_LOG_COMPILES'] = "1"

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
start_time_tot = time.time()
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
solution2['Xn'].content
solution2['Yn'].content
solution2['Zn'].content
solution2['pn'].content
solution2['Deltan'].content
solution2['B_psi_nm2'].content
print('Compile done, time elapsed(s):', (time.time() - start_time))
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

solution2['Xn'].content
solution2['Yn'].content
solution2['Zn'].content
solution2['pn'].content
solution2['Deltan'].content
solution2['B_psi_nm2'].content
print('Run done, time elapsed(s):',(time.time() - start_time))
print('Test1 done, time elapsed(s):',(time.time() - start_time_tot))

# Testing the speed if the second order
print('----- Speed test, 3rd order -----')
print('===== Compile speed =====')
start_time = time.time()
start_time_tot = time.time()
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
solution3['Xn'].content
solution3['Yn'].content
solution3['Zn'].content
solution3['pn'].content
solution3['Deltan'].content
solution3['B_psi_nm2'].content
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
solution3['Xn'].content
solution3['Yn'].content
solution3['Zn'].content
solution3['pn'].content
solution3['Deltan'].content
solution3['B_psi_nm2'].content
print('Run done, time elapsed(s):',(time.time() - start_time))
print('Test2 done, time elapsed(s):',(time.time() - start_time_tot))

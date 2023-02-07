import numpy as np
from math import floor, ceil
from math_utilities import *
from recursion_relations import *
import chiphifunc


def eval_looped_lambdas(equilibrium):
    # Creating new ChiPhiEpsFunc's for the resulting Equilibrium
    X_coef_cp = equilibrium.unknown['X_coef_cp']
    Y_coef_cp = equilibrium.unknown['Y_coef_cp']
    Z_coef_cp = equilibrium.unknown['Z_coef_cp']
    B_theta_coef_cp = equilibrium.unknown['B_theta_coef_cp']
    B_psi_coef_cp = equilibrium.unknown['B_psi_coef_cp']
    iota_coef = equilibrium.unknown['iota_coef']
    p_perp_coef_cp = equilibrium.unknown['p_perp_coef_cp']
    Delta_coef_cp = equilibrium.unknown['Delta_coef_cp']
    B_denom_coef_c = equilibrium.constant['B_denom_coef_c']
    B_alpha_coef = equilibrium.constant['B_alpha_coef']
    kap_p = equilibrium.constant['kap_p']
    dl_p = equilibrium.constant['dl_p']
    tau_p = equilibrium.constant['tau_p']
    eta = equilibrium.constant['eta']
    ''' iota (n_unknown-1)/2 '''
    # For evaluating the free parameter iota (n-1)/2 at odd n
    # (or n_eval-2/2 at even 'n_eval'. recall that the looped equation is evaluated
    # at one order higher than its unknowns)
    # iota terms in the looped equations
    coef_iota_const_a = ( # *(2*n_eval-3)
        (Delta_coef_cp[0]-1)*iota_coef[0]*diff(Y_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',2)
        +(Delta_coef_cp[0]-1)*iota_coef[0]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',2)
    )
    coef_iota_const_b = (
        2*(Delta_coef_cp[0]-1)*diff(Y_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1,'phi',1)
        +diff(Delta_coef_cp[0],'phi',1)*(diff(Y_coef_cp[1],'chi',1))**2
        +2*(Delta_coef_cp[0]-1)*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',1)
        +diff(Delta_coef_cp[0],'phi',1)*(diff(X_coef_cp[1],'chi',1))**2
    )
    coef_iota_const_c = (
        +B_alpha_coef[0]*B_denom_coef_c[0]*(2*Delta_coef_cp[0]-2)*diff(B_theta_coef_cp[2],'chi',1)
        +(1-Delta_coef_cp[0])*diff(Y_coef_cp[1],'chi',2)*diff(Y_coef_cp[1],'phi',1)
        +(1-Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',3)
        +(1-Delta_coef_cp[0])*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',2,'phi',1)
        -Y_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(Y_coef_cp[1],'chi',2)
        +(1-Delta_coef_cp[0])*diff(X_coef_cp[1],'chi',2)*diff(X_coef_cp[1],'phi',1)
        +(1-Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',3)
        +(1-Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],'chi',2,'phi',1)
        -X_coef_cp[1]*diff(Delta_coef_cp[0],'phi',1)*diff(X_coef_cp[1],'chi',2)
        +(
            2*B_alpha_coef[0]*B_denom_coef_c[0]
          -2*B_alpha_coef[0]*B_denom_coef_c[0]*Delta_coef_cp[0]
        )*diff(B_psi_coef_cp[0],'chi',2)
    )
    # iota terms in Delta_n
    coef_Delta_const = ChiPhiFunc(
        np.array([
            np.mean(-(B_alpha_coef[0]*B_denom_coef_c[1].dchi()).content,axis=1)
        ]).T
    )
    coef_dchi_Delta_const = ChiPhiFunc(
        np.array([
            np.mean((B_alpha_coef[0]*B_denom_coef_c[1]).content, axis=1)
        ]).T
    )
    coef_iota_nm1b2_in_RHS_Delta = -(
        B_denom_coef_c[0]*(diff(p_perp_coef_cp[1],'chi',1))
        +(diff(Delta_coef_cp[1],'chi',1))
        -Delta_coef_cp[0]*(diff(B_denom_coef_c[1],'chi',1))/2/B_denom_coef_c[0]
    )
    coef_iota_nm1b2_in_Delta = ChiPhiFunc(solve_dphi_iota_dchi(
        iota_coef[0],
        coef_iota_nm1b2_in_RHS_Delta.content
    ))
    lambda_coef_iota = lambda n_eval : (
        (
            coef_iota_const_a*(2*n_eval-3)
            +coef_iota_const_b*(n_eval-1)
            +coef_iota_const_c
        )/(B_alpha_coef[0]*n_eval)
        + coef_Delta_const*(n_eval-1)/(2*n_eval) * coef_iota_nm1b2_in_Delta
        + coef_dchi_Delta_const/(2*n_eval) * coef_iota_nm1b2_in_Delta.dchi())

    ''' Avg Delta n0 '''
    # This coefficient is used for calculating the free parameter avg(Delta[n,0])
    lambda_coef_delta = lambda n_eval : coef_Delta_const*(n_eval-1)/(2*n_eval)

    return({
        'lambda_coef_iota':lambda_coef_iota,
        'lambda_coef_delta':lambda_coef_delta
    })

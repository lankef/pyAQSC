# Evaluates coefficient of diff(B_psi_dummy,phi,j) 
from math import floor, ceil
from math_utilities import *
import chiphifunc

def eval_B_psi_coefs_D3(n_eval, X_coef_cp, Y_coef_cp,
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,
    dl_p, tau_p, kap_p, iota_coef, to_tensor_fft_op_multi_dim):
    
    coef_B_psi_dphi_0_dchi_0_all_but_Y_D3 = (((diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)-Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2))*diff(Y_coef_cp[1],'phi',1)+iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',3)+X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',2,'phi',1)+((-iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],'chi',1))+Y_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',2)-X_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1,'phi',1)-diff(X_coef_cp[1],'phi',1)*(diff(Y_coef_cp[1],'chi',1))**2+Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1)*diff(Y_coef_cp[1],'chi',1)-iota_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',3)-Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',2,'phi',1))*dl_p*kap_p*tau_p+((diff(X_coef_cp[1],'chi',1))**2*diff(Y_coef_cp[1],'phi',1)-iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',2)-X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1,'phi',1)-diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)*diff(Y_coef_cp[1],'chi',1)+iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',2)+Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',1))*diff(kap_p,'phi',1)+((-X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',2))+X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)+X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)-Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1))**2)*dl_p**2*kap_p**3+((-(diff(X_coef_cp[1],'chi',1))**2*diff(Y_coef_cp[1],'phi',2))+((-diff(X_coef_cp[1],'chi',2)*diff(X_coef_cp[1],'phi',1))-3*iota_coef[0]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',2)-2*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',1))*diff(Y_coef_cp[1],'phi',1)+(iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+2*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',3)+(X_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+3*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',2,'phi',1)+((diff(X_coef_cp[1],'phi',1))**2+4*iota_coef[0]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)+2*iota_coef[0]**2*(diff(X_coef_cp[1],'chi',1))**2)*diff(Y_coef_cp[1],'chi',2)+X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1,'phi',2)+2*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)*diff(Y_coef_cp[1],'chi',1,'phi',1)+(diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',2)-iota_coef[0]*diff(X_coef_cp[1],'chi',2)*diff(X_coef_cp[1],'phi',1)-2*iota_coef[0]**2*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',2))*diff(Y_coef_cp[1],'chi',1)+((-iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',3))-Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2,'phi',1))*diff(X_coef_cp[1],'phi',1)-2*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',3)-3*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',2,'phi',1)-Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',2))*kap_p)/(dl_p*kap_p**2*n_eval-dl_p*kap_p**2)
    
    coef_B_psi_dphi_0_dchi_1_all_but_Y_D3 = -((Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'phi',1)-2*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',2)-X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',1,'phi',1)+iota_coef[0]*X_coef_cp[1]*(diff(Y_coef_cp[1],'chi',1))**2+((-Y_coef_cp[1]*diff(X_coef_cp[1],'phi',1))-iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1)+2*iota_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',2)+Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',1,'phi',1))*dl_p*kap_p*tau_p+(iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)-iota_coef[0]*Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1))**2)*diff(kap_p,'phi',1)+(X_coef_cp[1]**2*diff(Y_coef_cp[1],'chi',1)-X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1))*dl_p**2*kap_p**3+((diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)+3*iota_coef[0]*(diff(X_coef_cp[1],'chi',1))**2)*diff(Y_coef_cp[1],'phi',1)+((-2*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'phi',1))-4*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',2)+((-X_coef_cp[1]*diff(X_coef_cp[1],'phi',1))-3*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1,'phi',1)+((-(diff(X_coef_cp[1],'phi',1))**2)-3*iota_coef[0]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1))*diff(Y_coef_cp[1],'chi',1)+(2*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)+Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1,'phi',1))*diff(X_coef_cp[1],'phi',1)+4*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',2)+3*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',1))*kap_p)/(dl_p*kap_p**2*n_eval-dl_p*kap_p**2)
    
    coef_B_psi_dphi_0_dchi_2_all_but_Y_D3 = ((iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)-iota_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',1))*dl_p*tau_p+(iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+2*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1)-iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)-2*iota_coef[0]**2*Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1))**2)/(dl_p*kap_p*n_eval-dl_p*kap_p)
    
    coef_B_psi_dphi_0_dchi_3_all_but_Y_D3 = 0
    
    coef_B_psi_dphi_1_dchi_0_all_but_Y_D3 = ((X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',2)-X_coef_cp[1]*(diff(Y_coef_cp[1],'chi',1))**2+Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)-Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',2))*dl_p*kap_p*tau_p+(Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1))**2-X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1))*diff(kap_p,'phi',1)+((-2*(diff(X_coef_cp[1],'chi',1))**2*diff(Y_coef_cp[1],'phi',1))+(X_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+3*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',2)+2*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1,'phi',1)+2*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)*diff(Y_coef_cp[1],'chi',1)-Y_coef_cp[1]*diff(X_coef_cp[1],'chi',2)*diff(X_coef_cp[1],'phi',1)-3*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',2)-2*Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'chi',1,'phi',1))*kap_p)/(dl_p*kap_p**2*n_eval-dl_p*kap_p**2)
    
    coef_B_psi_dphi_1_dchi_1_all_but_Y_D3 = ((X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],'chi',1)-Y_coef_cp[1]**2*diff(X_coef_cp[1],'chi',1))*dl_p*tau_p+(X_coef_cp[1]*diff(X_coef_cp[1],'phi',1)+3*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],'chi',1))*diff(Y_coef_cp[1],'chi',1)-Y_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(X_coef_cp[1],'phi',1)-3*iota_coef[0]*Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1))**2)/(dl_p*kap_p*n_eval-dl_p*kap_p)
    
    coef_B_psi_dphi_1_dchi_2_all_but_Y_D3 = 0
    
    coef_B_psi_dphi_1_dchi_3_all_but_Y_D3 = 0
    
    coef_B_psi_dphi_2_dchi_0_all_but_Y_D3 = (X_coef_cp[1]*diff(X_coef_cp[1],'chi',1)*diff(Y_coef_cp[1],'chi',1)-Y_coef_cp[1]*(diff(X_coef_cp[1],'chi',1))**2)/(dl_p*kap_p*n_eval-dl_p*kap_p)
    
    coef_B_psi_dphi_2_dchi_1_all_but_Y_D3 = 0
    
    coef_B_psi_dphi_2_dchi_2_all_but_Y_D3 = 0
    
    coef_B_psi_dphi_2_dchi_3_all_but_Y_D3 = 0
    return(
        to_tensor_fft_op_multi_dim(coef_B_psi_dphi_0_dchi_0_all_but_Y_D3, dphi=0, dchi=0, cap_axis0=1)
        +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_0_dchi_1_all_but_Y_D3, dphi=0, dchi=1, cap_axis0=1)
        +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_0_dchi_2_all_but_Y_D3, dphi=0, dchi=2, cap_axis0=1)
        +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_0_dchi_3_all_but_Y_D3, dphi=0, dchi=3, cap_axis0=1)
        +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_1_dchi_0_all_but_Y_D3, dphi=1, dchi=0, cap_axis0=1)
        +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_1_dchi_1_all_but_Y_D3, dphi=1, dchi=1, cap_axis0=1)
        +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_1_dchi_2_all_but_Y_D3, dphi=1, dchi=2, cap_axis0=1)
        +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_1_dchi_3_all_but_Y_D3, dphi=1, dchi=3, cap_axis0=1)
        +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_2_dchi_0_all_but_Y_D3, dphi=2, dchi=0, cap_axis0=1)
        +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_2_dchi_1_all_but_Y_D3, dphi=2, dchi=1, cap_axis0=1)
        +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_2_dchi_2_all_but_Y_D3, dphi=2, dchi=2, cap_axis0=1)
        +to_tensor_fft_op_multi_dim(coef_B_psi_dphi_2_dchi_3_all_but_Y_D3, dphi=2, dchi=3, cap_axis0=1)
        +0
    )
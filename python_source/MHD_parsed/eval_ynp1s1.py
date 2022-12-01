# Ynp1s1 
# Used in (conv(a) + conv(b)@dchi)@Yn+1 = RHS - LHS(Yn+1 = 0) 
# Must run with Yn+1=0.# Depends on Xn+1, Yn, Zn, B_theta n, B_psi n-2
# iota (n-2)/2 or (n-3)/2, B_alpha n/2 or (n-1)/2.
# Ysn is the chi-indep component
from math import floor, ceil
from math_utilities import *
import chiphifunc
def evaluate_ynp1s1_full(n,
    X_coef_cp,
    Y_coef_cp,
    Z_coef_cp,
    B_psi_coef_cp,
    B_theta_coef_cp,
    B_alpha_coef,
    kap_p, dl_p, tau_p, eta,
    iota_coef):    
    def sum_arg_31(i802):
        # Child args for sum_arg_31    
        def sum_arg_30(i800):
            # Child args for sum_arg_30
            return(B_theta_coef_cp[i800]*X_coef_cp[n-i802-i800+2])
        
        return(i802*X_coef_cp[i802]*py_sum(sum_arg_30,0,n-i802+2))
    
    def sum_arg_29(i798):
        # Child args for sum_arg_29    
        def sum_arg_28(i796):
            # Child args for sum_arg_28
            return(B_theta_coef_cp[i796]*Y_coef_cp[n-i798-i796+2])
        
        return(i798*Y_coef_cp[i798]*py_sum(sum_arg_28,0,n-i798+2))
    
    def sum_arg_27(i794):
        # Child args for sum_arg_27    
        def sum_arg_26(i792):
            # Child args for sum_arg_26
            return(B_psi_coef_cp[i792]*X_coef_cp[n-i794-i792])
        
        return(diff(X_coef_cp[i794],'chi',1)*py_sum(sum_arg_26,0,n-i794))
    
    def sum_arg_25(i790):
        # Child args for sum_arg_25    
        def sum_arg_24(i788):
            # Child args for sum_arg_24
            return(B_psi_coef_cp[i788]*Y_coef_cp[n-i790-i788])
        
        return(diff(Y_coef_cp[i790],'chi',1)*py_sum(sum_arg_24,0,n-i790))
    
    def sum_arg_23(i980):
        # Child args for sum_arg_23    
        def sum_arg_22(i820):
            # Child args for sum_arg_22
            return(B_psi_coef_cp[i820]*diff(X_coef_cp[n-i980-i820],'phi',1))
        
        return(diff(Y_coef_cp[i980],'chi',1)*py_sum(sum_arg_22,0,n-i980))
    
    def sum_arg_21(i978):
        # Child args for sum_arg_21    
        def sum_arg_20(i816):
            # Child args for sum_arg_20
            return(B_psi_coef_cp[i816]*diff(X_coef_cp[n-i978-i816],'chi',1))
        
        return(diff(Y_coef_cp[i978],'phi',1)*py_sum(sum_arg_20,0,n-i978))
    
    def sum_arg_19(i981):
        # Child args for sum_arg_19    
        def sum_arg_18(i982):
            # Child args for sum_arg_18
            return((((-i982)+2*i981-2)*diff(Y_coef_cp[i982],'chi',1)*X_coef_cp[(-n)-i982+2*i981-2]-diff(Y_coef_cp[i982],'chi',1)*X_coef_cp[(-n)-i982+2*i981-2]*n)*is_seq(n-i981+2,i981-i982))
        
        return(is_seq(0,n-i981+2)*B_alpha_coef[n-i981+2]*is_integer(n-i981+2)*py_sum(sum_arg_18,0,i981))
    
    def sum_arg_17(i974):
        # Child args for sum_arg_17    
        def sum_arg_16(i818):
            # Child args for sum_arg_16
            return((B_theta_coef_cp[i818]*n-B_theta_coef_cp[i818]*i974+(2-i818)*B_theta_coef_cp[i818])*X_coef_cp[n-i974-i818+2])
        
        return(diff(Y_coef_cp[i974],'phi',1)*py_sum(sum_arg_16,0,n-i974+2))
    
    def sum_arg_15(i829):
        # Child args for sum_arg_15    
        def sum_arg_14(i830):
            # Child args for sum_arg_14
            return((((-i830)+2*i829-2)*diff(X_coef_cp[i830],'chi',1)*Y_coef_cp[(-n)-i830+2*i829-2]-diff(X_coef_cp[i830],'chi',1)*Y_coef_cp[(-n)-i830+2*i829-2]*n)*is_seq(n-i829+2,i829-i830))
        
        return(is_seq(0,n-i829+2)*B_alpha_coef[n-i829+2]*is_integer(n-i829+2)*py_sum(sum_arg_14,0,i829))
    
    def sum_arg_13(i826):
        # Child args for sum_arg_13    
        def sum_arg_12(i824):
            # Child args for sum_arg_12
            return((B_theta_coef_cp[i824]*n-B_theta_coef_cp[i824]*i826+(2-i824)*B_theta_coef_cp[i824])*Y_coef_cp[n-i826-i824+2])
        
        return(diff(X_coef_cp[i826],'phi',1)*py_sum(sum_arg_12,0,n-i826+2))
    
    def sum_arg_11(i810):
        # Child args for sum_arg_11    
        def sum_arg_10(i808):
            # Child args for sum_arg_10
            return((B_theta_coef_cp[i808]*n-B_theta_coef_cp[i808]*i810+(2-i808)*B_theta_coef_cp[i808])*Y_coef_cp[n-i810-i808+2])
        
        return(Z_coef_cp[i810]*py_sum(sum_arg_10,0,n-i810+2))
    
    def sum_arg_9(i806):
        # Child args for sum_arg_9    
        def sum_arg_8(i804):
            # Child args for sum_arg_8
            return(B_psi_coef_cp[i804]*Z_coef_cp[n-i806-i804])
        
        return(diff(Y_coef_cp[i806],'chi',1)*py_sum(sum_arg_8,0,n-i806))
    
    def sum_arg_7(i983):
        # Child args for sum_arg_7    
        def sum_arg_6(i984):
            # Child args for sum_arg_6    
            def sum_arg_5(i1439):
                # Child args for sum_arg_5
                return(i1439*Y_coef_cp[i1439]*diff(X_coef_cp[(-i984)+i983-i1439],'chi',1))
            
            return(is_seq(0,(-n)+i984+i983-2)*B_theta_coef_cp[(-n)+i984+i983-2]*is_integer((-n)+i984+i983-2)*is_seq((-n)+i984+i983-2,i984)*py_sum(sum_arg_5,0,i983-i984))
        
        return(iota_coef[n-i983+2]*py_sum(sum_arg_6,0,i983))
    
    def sum_arg_4(i975):
        # Child args for sum_arg_4    
        def sum_arg_3(i976):
            # Child args for sum_arg_3    
            def sum_arg_2(i1423):
                # Child args for sum_arg_2
                return(i1423*X_coef_cp[i1423]*diff(Y_coef_cp[(-i976)+i975-i1423],'chi',1))
            
            return(is_seq(0,(-n)+i976+i975-2)*B_theta_coef_cp[(-n)+i976+i975-2]*is_integer((-n)+i976+i975-2)*is_seq((-n)+i976+i975-2,i976)*py_sum(sum_arg_2,0,i975-i976))
        
        return(iota_coef[n-i975+2]*py_sum(sum_arg_3,0,i975))
    
    def sum_arg_1(i831):
        # Child args for sum_arg_1
        return(is_seq(0,n-i831)*diff(Z_coef_cp[2*i831-n],'chi',1)*iota_coef[n-i831]*is_integer(n-i831)*is_seq(n-i831,i831))
    
    
    out = -((2*is_seq(0,n+2)*dl_p*kap_p*is_integer(n+2)*py_sum_parallel(sum_arg_31,0,n+2)+2*is_seq(0,n+2)*dl_p*kap_p*is_integer(n+2)*py_sum_parallel(sum_arg_29,0,n+2)-4*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_27,0,n)-4*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_25,0,n))*tau_p-4*dl_p*kap_p**2*is_integer(n)*py_sum_parallel(sum_arg_9,0,n)+kap_p*(2*py_sum_parallel(sum_arg_7,ceil(n/2)+1,floor(n)+2)-2*py_sum_parallel(sum_arg_4,ceil(n/2)+1,floor(n)+2))-4*kap_p*is_integer(n)*py_sum_parallel(sum_arg_23,0,n)+4*kap_p*is_integer(n)*py_sum_parallel(sum_arg_21,0,n)+2*kap_p*py_sum_parallel(sum_arg_19,ceil(n/2)+1,floor(n)+2)-2*is_seq(0,n+2)*kap_p*is_integer(n+2)*py_sum_parallel(sum_arg_17,0,n+2)-2*kap_p*py_sum_parallel(sum_arg_15,ceil(n/2)+1,floor(n)+2)+2*is_seq(0,n+2)*kap_p*is_integer(n+2)*py_sum_parallel(sum_arg_13,0,n+2)+2*is_seq(0,n+2)*dl_p*kap_p**2*is_integer(n+2)*py_sum_parallel(sum_arg_11,0,n+2)-4*kap_p*py_sum_parallel(sum_arg_1,ceil(n/2),floor(n))-4*kap_p*is_integer(n)*diff(Z_coef_cp[n],'phi',1)+4*dl_p*kap_p**2*X_coef_cp[n]*is_integer(n))/(B_alpha_coef[0]*eta*n+2*B_alpha_coef[0]*eta)
    return(out)

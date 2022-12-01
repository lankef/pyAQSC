# Evaluating dchi B_psi_n-2. No masking needed. 
# Uses Xn-1, Yn-1, Zn-1,  B_theta_n, B_psi_n-3, iota_coef (n-1)/2 or (n-2)/2
# Must be evaluated with Z_coef_cp[n] = 0 and B_psi_coef_cp[n-2] = 0 
from math import floor, ceil
from math_utilities import *
import chiphifunc
def eval_dchi_B_psi_cp_nm2(n, X_coef_cp, Y_coef_cp, Z_coef_cp, \
    B_theta_coef_cp, B_psi_coef_cp, B_alpha_coef, B_denom_coef_c, \
    kap_p, dl_p, tau_p, iota_coef):    
    def sum_arg_12(i290):
        # Child args for sum_arg_12
        return(dl_p*Y_coef_cp[i290]*is_integer(n)*diff(X_coef_cp[n-i290],'chi',1)-dl_p*X_coef_cp[i290]*is_integer(n)*diff(Y_coef_cp[n-i290],'chi',1))
    
    def sum_arg_11(i258):
        # Child args for sum_arg_11
        return((2*dl_p*i258*X_coef_cp[i258]-dl_p*X_coef_cp[i258]*n)*is_integer(n)*diff(Y_coef_cp[n-i258],'chi',1)+(2*dl_p*i258*diff(X_coef_cp[i258],'chi',1)-dl_p*diff(X_coef_cp[i258],'chi',1)*n)*is_integer(n)*Y_coef_cp[n-i258])
    
    def sum_arg_10(i292):
        # Child args for sum_arg_10
        return((-diff(Z_coef_cp[i292],'chi',1)*is_integer(n)*diff(Z_coef_cp[n-i292],'phi',1))+dl_p*X_coef_cp[i292]*kap_p*is_integer(n)*diff(Z_coef_cp[n-i292],'chi',1)-diff(Y_coef_cp[i292],'chi',1)*is_integer(n)*diff(Y_coef_cp[n-i292],'phi',1)-diff(X_coef_cp[i292],'chi',1)*is_integer(n)*diff(X_coef_cp[n-i292],'phi',1)-dl_p*Z_coef_cp[i292]*kap_p*is_integer(n)*diff(X_coef_cp[n-i292],'chi',1))
    
    def sum_arg_9(i283):
        # Child args for sum_arg_9    
        def sum_arg_8(i284):
            # Child args for sum_arg_8
            return(((-is_seq(0,n-i283)*diff(Z_coef_cp[i284],'chi',1)*diff(Z_coef_cp[(-n)-i284+2*i283],'chi',1))-is_seq(0,n-i283)*diff(Y_coef_cp[i284],'chi',1)*diff(Y_coef_cp[(-n)-i284+2*i283],'chi',1)-is_seq(0,n-i283)*diff(X_coef_cp[i284],'chi',1)*diff(X_coef_cp[(-n)-i284+2*i283],'chi',1))*is_seq(n-i283,i283-i284))
        
        return(iota_coef[n-i283]*is_integer(n-i283)*py_sum(sum_arg_8,0,i283))
    
    def sum_arg_7(i271):
        # Child args for sum_arg_7    
        def sum_arg_6(i272):
            # Child args for sum_arg_6
            return((is_seq(0,n-i271)*i272*Z_coef_cp[i272]*diff(Z_coef_cp[(-n)-i272+2*i271],'chi',2)+is_seq(0,n-i271)*i272*diff(Z_coef_cp[i272],'chi',1)*diff(Z_coef_cp[(-n)-i272+2*i271],'chi',1)+is_seq(0,n-i271)*i272*Y_coef_cp[i272]*diff(Y_coef_cp[(-n)-i272+2*i271],'chi',2)+is_seq(0,n-i271)*i272*diff(Y_coef_cp[i272],'chi',1)*diff(Y_coef_cp[(-n)-i272+2*i271],'chi',1)+is_seq(0,n-i271)*i272*X_coef_cp[i272]*diff(X_coef_cp[(-n)-i272+2*i271],'chi',2)+is_seq(0,n-i271)*i272*diff(X_coef_cp[i272],'chi',1)*diff(X_coef_cp[(-n)-i272+2*i271],'chi',1))*is_seq(n-i271,i271-i272))
        
        return(iota_coef[n-i271]*is_integer(n-i271)*py_sum(sum_arg_6,0,i271))
    
    def sum_arg_5(i254):
        # Child args for sum_arg_5
        return((i254*diff(Z_coef_cp[i254],'chi',1)*is_integer(n)*diff(Z_coef_cp[n-i254],'phi',1))\
            +(i254*Z_coef_cp[i254]*is_integer(n)*diff(Z_coef_cp[n-i254],'chi',1,'phi',1))\
            +((2*dl_p*i254*X_coef_cp[i254]*kap_p-dl_p*X_coef_cp[i254]*kap_p*n)*is_integer(n)*diff(Z_coef_cp[n-i254],'chi',1))\
            +(i254*diff(Y_coef_cp[i254],'chi',1)*is_integer(n)*diff(Y_coef_cp[n-i254],'phi',1))\
            +(i254*Y_coef_cp[i254]*is_integer(n)*diff(Y_coef_cp[n-i254],'chi',1,'phi',1))\
            +(i254*diff(X_coef_cp[i254],'chi',1)*is_integer(n)*diff(X_coef_cp[n-i254],'phi',1))\
            +(i254*X_coef_cp[i254]*is_integer(n)*diff(X_coef_cp[n-i254],'chi',1,'phi',1))\
            +((2*dl_p*i254*diff(X_coef_cp[i254],'chi',1)*kap_p-dl_p*diff(X_coef_cp[i254],'chi',1)*kap_p*n)*is_integer(n)*Z_coef_cp[n-i254]))
    
    def sum_arg_4(i209):
        # Child args for sum_arg_4    
        def sum_arg_3(i210):
            # Child args for sum_arg_3
            return(B_theta_coef_cp[i210]*B_denom_coef_c[(-n)-i210+2*i209]*is_seq(n-i209,i209-i210))
        
        return(is_seq(0,n-i209)*B_alpha_coef[n-i209]*is_integer(n-i209)*py_sum(sum_arg_3,0,i209))
    
    def sum_arg_2(i201):
        # Child args for sum_arg_2    
        def sum_arg_1(i202):
            # Child args for sum_arg_1
            return((diff(B_psi_coef_cp[i202],'chi',1)*B_denom_coef_c[(-n)-i202+2*i201+2]+B_psi_coef_cp[i202]*diff(B_denom_coef_c[(-n)-i202+2*i201+2],'chi',1))*is_seq(n-i201-2,i201-i202))
        
        return(is_seq(0,n-i201-2)*B_alpha_coef[n-i201-2]*is_integer(n-i201-2)*py_sum(sum_arg_1,0,i201))
    
    
    out = -((n*py_sum_parallel(sum_arg_12,0,n)-py_sum_parallel(sum_arg_11,0,n))*tau_p-n*py_sum_parallel(sum_arg_9,ceil(n/2),floor(n))-py_sum_parallel(sum_arg_7,ceil(n/2),floor(n))-py_sum_parallel(sum_arg_5,0,n)-n*py_sum_parallel(sum_arg_4,ceil(n/2),floor(n))+2*py_sum_parallel(sum_arg_2,ceil(n/2)-1,floor(n)-2)-n*py_sum_parallel(sum_arg_10,0,n))/(2*B_alpha_coef[0]*B_denom_coef_c[0])
    return(out)

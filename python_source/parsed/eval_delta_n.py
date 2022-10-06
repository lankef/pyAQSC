# (dphi + iota_coef[0] * dchi) Delta_n = inhomogenous 
# This method evaluates the inhomogenous component, which serves as 
# the input to solve_dphi_iota_dchi(iota, f).
# Uses B_denom_coef_cp[n], p_perp_coef_cp[n], iota[(n-1)/2 or n/2].
# Must be evaluated with Delta_n=0 
from math import floor, ceil
from math_utilities import *
import chiphifunc
def eval_inhomogenous_Delta_n_cp(n,
    B_denom_coef_c,
    p_perp_coef_cp, 
    Delta_coef_cp,
    iota_coef):    
    def sum_arg_10(i1219):
        # Child args for sum_arg_10    
        def sum_arg_9(i1220):
            # Child args for sum_arg_9    
            def sum_arg_8(i1423):
                # Child args for sum_arg_8
                return(B_denom_coef_c[i1220-i1423]*B_denom_coef_c[i1423])
            
            return(diff(p_perp_coef_cp[(-n)-i1220+2*i1219],'chi',1)*is_seq(n-i1219,i1219-i1220)*py_sum(sum_arg_8,0,i1220))
        
        return(is_seq(0,n-i1219)*iota_coef[n-i1219]*is_integer(n-i1219)*py_sum(sum_arg_9,0,i1219))
    
    def sum_arg_7(i1218):
        # Child args for sum_arg_7    
        def sum_arg_6(i1178):
            # Child args for sum_arg_6
            return(B_denom_coef_c[i1178]*B_denom_coef_c[n-i1218-i1178])
        
        return(diff(p_perp_coef_cp[i1218],'phi',1)*py_sum(sum_arg_6,0,n-i1218))
    
    def sum_arg_5(i1215):
        # Child args for sum_arg_5    
        def sum_arg_4(i1216):
            # Child args for sum_arg_4
            return(B_denom_coef_c[i1216]*diff(Delta_coef_cp[(-n)-i1216+2*i1215],'chi',1)*is_seq(n-i1215,i1215-i1216))
        
        return(is_seq(0,n-i1215)*iota_coef[n-i1215]*is_integer(n-i1215)*py_sum(sum_arg_4,0,i1215))
    
    def sum_arg_3(i1214):
        # Child args for sum_arg_3
        return(B_denom_coef_c[i1214]*diff(Delta_coef_cp[n-i1214],'phi',1))
    
    def sum_arg_2(i1425):
        # Child args for sum_arg_2    
        def sum_arg_1(i1182):
            # Child args for sum_arg_1
            return(Delta_coef_cp[i1182]*diff(B_denom_coef_c[(-n)+2*i1425-i1182],'chi',1))
        
        return(is_seq(0,n-i1425)*iota_coef[n-i1425]*is_integer(n-i1425)*is_seq(n-i1425,i1425)*py_sum(sum_arg_1,0,2*i1425-n))
    
    
    out = ((-is_seq(0,n)*is_integer(n)*py_sum(sum_arg_7,0,n))-py_sum(sum_arg_5,ceil(0.5*n),floor(n))-is_seq(0,n)*is_integer(n)*py_sum(sum_arg_3,0,n)+py_sum(sum_arg_2,ceil(0.5*n),floor(n))/2-py_sum(sum_arg_10,ceil(0.5*n),floor(n)))/B_denom_coef_c[0]
    return(out)

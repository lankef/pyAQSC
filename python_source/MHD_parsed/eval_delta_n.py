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
    def sum_arg_7(i259):
        # Child args for sum_arg_7    
        def sum_arg_3(i224):
            # Child args for sum_arg_3
            return(Delta_coef_cp[i224]*diff(B_denom_coef_c[(-n)+2*i259-i224],'chi',1))
            
        def sum_arg_4(i256):
            # Child args for sum_arg_4
            return(B_denom_coef_c[i256]*diff(Delta_coef_cp[(-n)+2*i259-i256],'chi',1)*is_seq(n-i259,i259-i256))
            
        def sum_arg_6(i260):
            # Child args for sum_arg_6    
            def sum_arg_5(i279):
                # Child args for sum_arg_5
                return(B_denom_coef_c[i260-i279]*B_denom_coef_c[i279])
            
            return(diff(p_perp_coef_cp[(-n)-i260+2*i259],'chi',1)*is_seq(n-i259,i259-i260)*py_sum(sum_arg_5,0,i260))
        
        return(2*is_seq(0,n-i259)*iota_coef[n-i259]*is_integer(n-i259)*py_sum(sum_arg_6,0,i259)+2*is_seq(0,n-i259)*iota_coef[n-i259]*is_integer(n-i259)*py_sum(sum_arg_4,0,i259)-is_seq(0,n-i259)*iota_coef[n-i259]*is_integer(n-i259)*is_seq(n-i259,i259)*py_sum(sum_arg_3,0,2*i259-n))
    
    def sum_arg_2(i258):
        # Child args for sum_arg_2    
        def sum_arg_1(i220):
            # Child args for sum_arg_1
            return(B_denom_coef_c[i220]*B_denom_coef_c[n-i258-i220])
        
        return(diff(p_perp_coef_cp[i258],'phi',1)*py_sum(sum_arg_1,0,n-i258)+B_denom_coef_c[i258]*diff(Delta_coef_cp[n-i258],'phi',1))
    
    
    out = -(py_sum_parallel(sum_arg_7,ceil(n/2),floor(n))+2*is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_2,0,n))/(2*B_denom_coef_c[0])
    return(out)

# (dphi + iota_coef[0] * dchi) Delta_n = inhomogenous 
# This method evaluates the inhomogenous component, which serves as 
# the input to solve_dphi_iota_dchi(iota, f).# Must be evaluated with Delta_n=0 
from math import floor, ceil
from math_utilities import is_seq, py_sum, is_integer, diff
import chiphifunc
def eval_inhomogenous_Delta_n_cp(n,
    B_denom_coef_c,
    p_perp_coef_cp, 
    delta_coef_cp,
    iota_coef):    
    def sum_arg_10(i241):
        # Child args for sum_arg_10    
        def sum_arg_9(i242):
            # Child args for sum_arg_9    
            def sum_arg_8(i261):
                # Child args for sum_arg_8
                return(B_denom_coef_c[i242-i261]*B_denom_coef_c[i261])
            
            return(diff(p_perp_coef_cp[(-n)-i242+2*i241],'chi',1)*is_seq(n-i241,i241-i242)*py_sum(sum_arg_8,0,i242))
        
        return(is_seq(0,n-i241)*iota_coef[n-i241]*is_integer(n-i241)*py_sum(sum_arg_9,0,i241))
    
    def sum_arg_7(i240):
        # Child args for sum_arg_7    
        def sum_arg_6(i200):
            # Child args for sum_arg_6
            return(B_denom_coef_c[i200]*B_denom_coef_c[n-i240-i200])
        
        return(diff(p_perp_coef_cp[i240],'phi',1)*py_sum(sum_arg_6,0,n-i240))
    
    def sum_arg_5(i237):
        # Child args for sum_arg_5    
        def sum_arg_4(i238):
            # Child args for sum_arg_4
            return(B_denom_coef_c[i238]*diff(delta_coef_cp[(-n)-i238+2*i237],'chi',1)*is_seq(n-i237,i237-i238))
        
        return(is_seq(0,n-i237)*iota_coef[n-i237]*is_integer(n-i237)*py_sum(sum_arg_4,0,i237))
    
    def sum_arg_3(i236):
        # Child args for sum_arg_3
        return(B_denom_coef_c[i236]*diff(delta_coef_cp[n-i236],'phi',1))
    
    def sum_arg_2(i263):
        # Child args for sum_arg_2    
        def sum_arg_1(i204):
            # Child args for sum_arg_1
            return(delta_coef_cp[i204]*diff(B_denom_coef_c[(-n)+2*i263-i204],'chi',1))
        
        return(is_seq(0,n-i263)*iota_coef[n-i263]*is_integer(n-i263)*is_seq(n-i263,i263)*py_sum(sum_arg_1,0,2*i263-n))
    
    
    out = ((-is_seq(0,n)*is_integer(n)*py_sum(sum_arg_7,0,n))-py_sum(sum_arg_5,ceil(0.5*n),floor(n))-is_seq(0,n)*is_integer(n)*py_sum(sum_arg_3,0,n)+py_sum(sum_arg_2,ceil(0.5*n),floor(n))/2-py_sum(sum_arg_10,ceil(0.5*n),floor(n)))/B_denom_coef_c[0]
    return(out)

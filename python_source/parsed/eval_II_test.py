from math import floor, ceil
from math_utilities import *
import chiphifunc
def eval_II_lhs_test(n, X_coef_cp, Y_coef_cp, Z_coef_cp, \
    B_theta_coef_cp, B_psi_coef_cp, B_alpha_coef, B_denom_coef_c, \
    p_perp_coef_cp, Delta_coef_cp, kap_p, dl_p, tau_p, iota_coef):    
    def sum_arg_8(i347):
        # Child args for sum_arg_8    
        def sum_arg_7(i348):
            # Child args for sum_arg_7    
            def sum_arg_6(i346):
                # Child args for sum_arg_6
                return(Delta_coef_cp[i346]*diff(B_theta_coef_cp[(-n)-i348+2*i347-i346],'chi',1)*is_seq(n-i347,(-i348)+i347-i346))
            
            return(B_denom_coef_c[i348]*py_sum(sum_arg_6,0,i347-i348))
        
        return(is_seq(0,n-i347)*iota_coef[n-i347]*is_integer(n-i347)*py_sum(sum_arg_7,0,i347))
    
    def sum_arg_5(i344):
        # Child args for sum_arg_5    
        def sum_arg_4(i342):
            # Child args for sum_arg_4
            return(Delta_coef_cp[i342]*diff(B_theta_coef_cp[n-i344-i342],'phi',1))
        
        return(B_denom_coef_c[i344]*py_sum(sum_arg_4,0,n-i344))
    
    def sum_arg_3(i339):
        # Child args for sum_arg_3    
        def sum_arg_2(i340):
            # Child args for sum_arg_2
            return(B_denom_coef_c[i340]*diff(B_theta_coef_cp[(-n)-i340+2*i339],'chi',1)*is_seq(n-i339,i339-i340))
        
        return(is_seq(0,n-i339)*iota_coef[n-i339]*is_integer(n-i339)*py_sum(sum_arg_2,0,i339))
    
    def sum_arg_1(i338):
        # Child args for sum_arg_1
        return(B_denom_coef_c[i338]*diff(B_theta_coef_cp[n-i338],'phi',1))
    
    
    out = (-py_sum_parallel(sum_arg_8,ceil(0.5*n),floor(n)))-is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_5,0,n)+py_sum_parallel(sum_arg_3,ceil(0.5*n),floor(n))+is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_1,0,n)
    return(out)
def eval_II_rhs_test(n, X_coef_cp, Y_coef_cp, Z_coef_cp, \
    B_theta_coef_cp, B_psi_coef_cp, B_alpha_coef, B_denom_coef_c, \
    p_perp_coef_cp, Delta_coef_cp, kap_p, dl_p, tau_p, iota_coef):    
    def sum_arg_15(i350):
        # Child args for sum_arg_15    
        def sum_arg_14(i228):
            # Child args for sum_arg_14    
            def sum_arg_13(i226):
                # Child args for sum_arg_13
                return(B_denom_coef_c[i226]*B_denom_coef_c[n-i350-i228-i226])
            
            return(diff(p_perp_coef_cp[i228],'phi',1)*py_sum(sum_arg_13,0,n-i350-i228))
        
        return(B_theta_coef_cp[i350]*py_sum(sum_arg_14,0,n-i350))
    
    def sum_arg_12(i357):
        # Child args for sum_arg_12    
        def sum_arg_11(i358):
            # Child args for sum_arg_11    
            def sum_arg_10(i806):
                # Child args for sum_arg_10
                return(diff(B_denom_coef_c[(-i806)-i358+i357],'chi',1)*Delta_coef_cp[i806])
            
            return(is_seq(0,(-n)+i358+i357)*B_theta_coef_cp[(-n)+i358+i357]*is_integer((-n)+i358+i357)*is_seq((-n)+i358+i357,i358)*py_sum(sum_arg_10,0,i357-i358))
        
        return(iota_coef[n-i357]*py_sum(sum_arg_11,0,i357))
    
    def sum_arg_9(i355):
        # Child args for sum_arg_9    
        def sum_arg_8(i238):
            # Child args for sum_arg_8
            return(Delta_coef_cp[i238]*diff(B_denom_coef_c[(-n)+2*i355-i238],'chi',1))
        
        return(is_seq(0,n-i355)*B_alpha_coef[n-i355]*is_integer(n-i355)*is_seq(n-i355,i355)*py_sum(sum_arg_8,0,2*i355-n))
    
    def sum_arg_7(i353):
        # Child args for sum_arg_7    
        def sum_arg_6(i354):
            # Child args for sum_arg_6    
            def sum_arg_5(i767):
                # Child args for sum_arg_5    
                def sum_arg_4(i790):
                    # Child args for sum_arg_4
                    return(B_denom_coef_c[(-i790)-i767-i354+i353]*B_denom_coef_c[i790])
                
                return(diff(p_perp_coef_cp[i767],'chi',1)*py_sum(sum_arg_4,0,(-i767)-i354+i353))
            
            return(is_seq(0,(-n)+i354+i353)*B_theta_coef_cp[(-n)+i354+i353]*is_integer((-n)+i354+i353)*is_seq((-n)+i354+i353,i354)*py_sum(sum_arg_5,0,i353-i354))
        
        return(iota_coef[n-i353]*py_sum(sum_arg_6,0,i353))
    
    def sum_arg_3(i351):
        # Child args for sum_arg_3    
        def sum_arg_2(i234):
            # Child args for sum_arg_2    
            def sum_arg_1(i230):
                # Child args for sum_arg_1
                return(B_denom_coef_c[i230]*B_denom_coef_c[(-n)+2*i351-i234-i230])
            
            return(diff(p_perp_coef_cp[i234],'chi',1)*py_sum(sum_arg_1,0,(-n)+2*i351-i234))
        
        return(is_seq(0,n-i351)*B_alpha_coef[n-i351]*is_integer(n-i351)*is_seq(n-i351,i351)*py_sum(sum_arg_2,0,2*i351-n))
    
    
    out = (-py_sum_parallel(sum_arg_9,ceil(0.5*n),floor(n))/2)-py_sum_parallel(sum_arg_7,ceil(0.5*n),floor(n))+py_sum_parallel(sum_arg_3,ceil(0.5*n),floor(n))-is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_15,0,n)+py_sum_parallel(sum_arg_12,ceil(0.5*n),floor(n))/2
    return(out)

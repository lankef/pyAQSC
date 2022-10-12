from math import floor, ceil
from math_utilities import *
import chiphifunc
def eval_II_lhs_test(n, X_coef_cp, Y_coef_cp, Z_coef_cp, \
    B_theta_coef_cp, B_psi_coef_cp, B_alpha_coef, B_denom_coef_c, \
    p_perp_coef_cp, Delta_coef_cp, kap_p, dl_p, tau_p, iota_coef):    
    def sum_arg_8(i1069):
        # Child args for sum_arg_8    
        def sum_arg_7(i1070):
            # Child args for sum_arg_7    
            def sum_arg_6(i1068):
                # Child args for sum_arg_6
                return(Delta_coef_cp[i1068]*diff(B_theta_coef_cp[(-n)-i1070+2*i1069-i1068],'chi',1)*is_seq(n-i1069,(-i1070)+i1069-i1068))
            
            return(B_denom_coef_c[i1070]*py_sum(sum_arg_6,0,i1069-i1070))
        
        return(is_seq(0,n-i1069)*iota_coef[n-i1069]*is_integer(n-i1069)*py_sum(sum_arg_7,0,i1069))
    
    def sum_arg_5(i1066):
        # Child args for sum_arg_5    
        def sum_arg_4(i1064):
            # Child args for sum_arg_4
            return(Delta_coef_cp[i1064]*diff(B_theta_coef_cp[n-i1066-i1064],'phi',1))
        
        return(B_denom_coef_c[i1066]*py_sum(sum_arg_4,0,n-i1066))
    
    def sum_arg_3(i1061):
        # Child args for sum_arg_3    
        def sum_arg_2(i1062):
            # Child args for sum_arg_2
            return(B_denom_coef_c[i1062]*diff(B_theta_coef_cp[(-n)-i1062+2*i1061],'chi',1)*is_seq(n-i1061,i1061-i1062))
        
        return(is_seq(0,n-i1061)*iota_coef[n-i1061]*is_integer(n-i1061)*py_sum(sum_arg_2,0,i1061))
    
    def sum_arg_1(i1060):
        # Child args for sum_arg_1
        return(B_denom_coef_c[i1060]*diff(B_theta_coef_cp[n-i1060],'phi',1))
    
    
    out = (-py_sum_parallel(sum_arg_8,ceil(0.5*n),floor(n)))-is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_5,0,n)+py_sum_parallel(sum_arg_3,ceil(0.5*n),floor(n))+is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_1,0,n)
    return(out)
def eval_II_rhs_test(n, X_coef_cp, Y_coef_cp, Z_coef_cp, \
    B_theta_coef_cp, B_psi_coef_cp, B_alpha_coef, B_denom_coef_c, \
    p_perp_coef_cp, Delta_coef_cp, kap_p, dl_p, tau_p, iota_coef):    
    def sum_arg_15(i1072):
        # Child args for sum_arg_15    
        def sum_arg_14(i1034):
            # Child args for sum_arg_14    
            def sum_arg_13(i1032):
                # Child args for sum_arg_13
                return(B_denom_coef_c[i1032]*B_denom_coef_c[n-i1072-i1034-i1032])
            
            return(diff(p_perp_coef_cp[i1034],'phi',1)*py_sum(sum_arg_13,0,n-i1072-i1034))
        
        return(B_theta_coef_cp[i1072]*py_sum(sum_arg_14,0,n-i1072))
    
    def sum_arg_12(i1079):
        # Child args for sum_arg_12    
        def sum_arg_11(i1080):
            # Child args for sum_arg_11    
            def sum_arg_10(i1528):
                # Child args for sum_arg_10
                return(diff(B_denom_coef_c[(-i1528)-i1080+i1079],'chi',1)*Delta_coef_cp[i1528])
            
            return(is_seq(0,(-n)+i1080+i1079)*B_theta_coef_cp[(-n)+i1080+i1079]*is_integer((-n)+i1080+i1079)*is_seq((-n)+i1080+i1079,i1080)*py_sum(sum_arg_10,0,i1079-i1080))
        
        return(iota_coef[n-i1079]*py_sum(sum_arg_11,0,i1079))
    
    def sum_arg_9(i1077):
        # Child args for sum_arg_9    
        def sum_arg_8(i1044):
            # Child args for sum_arg_8
            return(Delta_coef_cp[i1044]*diff(B_denom_coef_c[(-n)+2*i1077-i1044],'chi',1))
        
        return(is_seq(0,n-i1077)*B_alpha_coef[n-i1077]*is_integer(n-i1077)*is_seq(n-i1077,i1077)*py_sum(sum_arg_8,0,2*i1077-n))
    
    def sum_arg_7(i1075):
        # Child args for sum_arg_7    
        def sum_arg_6(i1076):
            # Child args for sum_arg_6    
            def sum_arg_5(i1489):
                # Child args for sum_arg_5    
                def sum_arg_4(i1512):
                    # Child args for sum_arg_4
                    return(B_denom_coef_c[(-i1512)-i1489-i1076+i1075]*B_denom_coef_c[i1512])
                
                return(diff(p_perp_coef_cp[i1489],'chi',1)*py_sum(sum_arg_4,0,(-i1489)-i1076+i1075))
            
            return(is_seq(0,(-n)+i1076+i1075)*B_theta_coef_cp[(-n)+i1076+i1075]*is_integer((-n)+i1076+i1075)*is_seq((-n)+i1076+i1075,i1076)*py_sum(sum_arg_5,0,i1075-i1076))
        
        return(iota_coef[n-i1075]*py_sum(sum_arg_6,0,i1075))
    
    def sum_arg_3(i1073):
        # Child args for sum_arg_3    
        def sum_arg_2(i1040):
            # Child args for sum_arg_2    
            def sum_arg_1(i1036):
                # Child args for sum_arg_1
                return(B_denom_coef_c[i1036]*B_denom_coef_c[(-n)+2*i1073-i1040-i1036])
            
            return(diff(p_perp_coef_cp[i1040],'chi',1)*py_sum(sum_arg_1,0,(-n)+2*i1073-i1040))
        
        return(is_seq(0,n-i1073)*B_alpha_coef[n-i1073]*is_integer(n-i1073)*is_seq(n-i1073,i1073)*py_sum(sum_arg_2,0,2*i1073-n))
    
    
    out = (-py_sum_parallel(sum_arg_9,ceil(0.5*n),floor(n))/2)-py_sum_parallel(sum_arg_7,ceil(0.5*n),floor(n))+py_sum_parallel(sum_arg_3,ceil(0.5*n),floor(n))-is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_15,0,n)+py_sum_parallel(sum_arg_12,ceil(0.5*n),floor(n))/2
    return(out)

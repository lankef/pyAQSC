# Evaluates p_perp_n 
# Uses n, B_theta_coef_cp[n-2], B_psi_coef_cp[n-2], 
# B_alpha_coef[first-order terms], B_denom_coef_c[n],
# p_perp_coef_cp[n-1], Delta_coef_cp[n-1],iota_coef[(n-3)/2 or (n-2)/2] 
# Must be evaluated with p_perp_n=0 
from math import floor, ceil
from math_utilities import *
import chiphifunc
def eval_p_perp_n_cp(n,
    B_theta_coef_cp,
    B_psi_coef_cp,
    B_alpha_coef,
    B_denom_coef_c,
    p_perp_coef_cp, 
    Delta_coef_cp,
    iota_coef):    
    def sum_arg_26(i592):
        # Child args for sum_arg_26    
        def sum_arg_25(i554):
            # Child args for sum_arg_25
            return(B_psi_coef_cp[i554]*diff(Delta_coef_cp[n-i592-i554-2],'phi',1))
        
        return(B_denom_coef_c[i592]*py_sum(sum_arg_25,0,n-i592-2))
    
    def sum_arg_24(i588):
        # Child args for sum_arg_24    
        def sum_arg_23(i586):
            # Child args for sum_arg_23
            return(Delta_coef_cp[i586]*diff(B_psi_coef_cp[n-i588-i586-2],'phi',1))
        
        return(B_denom_coef_c[i588]*py_sum(sum_arg_23,0,n-i588-2))
    
    def sum_arg_22(i583):
        # Child args for sum_arg_22    
        def sum_arg_21(i584):
            # Child args for sum_arg_21    
            def sum_arg_20(i556):
                # Child args for sum_arg_20
                return(B_psi_coef_cp[i556]*diff(Delta_coef_cp[(-n)-i584+2*i583-i556+2],'chi',1))
            
            return(B_denom_coef_c[i584]*is_seq(n-i583-2,i583-i584)*py_sum(sum_arg_20,0,(-n)-i584+2*i583+2))
        
        return(is_seq(0,n-i583-2)*iota_coef[n-i583-2]*is_integer(n-i583-2)*py_sum(sum_arg_21,0,i583))
    
    def sum_arg_19(i579):
        # Child args for sum_arg_19    
        def sum_arg_18(i580):
            # Child args for sum_arg_18    
            def sum_arg_17(i576):
                # Child args for sum_arg_17
                return(Delta_coef_cp[i576]*diff(B_psi_coef_cp[(-n)-i580+2*i579-i576+2],'chi',1))
            
            return(B_denom_coef_c[i580]*is_seq(n-i579-2,i579-i580)*py_sum(sum_arg_17,0,(-n)-i580+2*i579+2))
        
        return(is_seq(0,n-i579-2)*iota_coef[n-i579-2]*is_integer(n-i579-2)*py_sum(sum_arg_18,0,i579))
    
    def sum_arg_16(i569):
        # Child args for sum_arg_16    
        def sum_arg_15(i570):
            # Child args for sum_arg_15
            return(B_denom_coef_c[i570]*Delta_coef_cp[(-n)-i570+2*i569]*is_seq(n-i569,i569-i570))
        
        return(is_seq(0,n-i569)*(n-i569)*B_alpha_coef[n-i569]*is_integer(n-i569)*py_sum(sum_arg_15,0,i569))
    
    def sum_arg_14(i563):
        # Child args for sum_arg_14    
        def sum_arg_13(i564):
            # Child args for sum_arg_13
            return(B_denom_coef_c[i564]*diff(B_psi_coef_cp[(-n)-i564+2*i563+2],'chi',1)*is_seq(n-i563-2,i563-i564))
        
        return(is_seq(0,n-i563-2)*iota_coef[n-i563-2]*is_integer(n-i563-2)*py_sum(sum_arg_13,0,i563))
    
    def sum_arg_12(i562):
        # Child args for sum_arg_12
        return(B_denom_coef_c[i562]*diff(B_psi_coef_cp[n-i562-2],'phi',1))
    
    def sum_arg_11(i557):
        # Child args for sum_arg_11    
        def sum_arg_10(i558):
            # Child args for sum_arg_10
            return(i558*B_denom_coef_c[i558]*Delta_coef_cp[(-n)-i558+2*i557]*is_seq(n-i557,i557-i558))
        
        return(is_seq(0,n-i557)*B_alpha_coef[n-i557]*is_integer(n-i557)*py_sum(sum_arg_10,0,i557))
    
    def sum_arg_9(i549):
        # Child args for sum_arg_9    
        def sum_arg_8(i550):
            # Child args for sum_arg_8    
            def sum_arg_7(i500):
                # Child args for sum_arg_7
                return(B_denom_coef_c[i500]*B_denom_coef_c[(-n)-i550+2*i549-i500])
            
            return(i550*p_perp_coef_cp[i550]*is_seq(n-i549,i549-i550)*py_sum(sum_arg_7,0,(-n)-i550+2*i549))
        
        return(is_seq(0,n-i549)*B_alpha_coef[n-i549]*is_integer(n-i549)*py_sum(sum_arg_8,0,i549))
    
    def sum_arg_6(i589):
        # Child args for sum_arg_6    
        def sum_arg_5(i590):
            # Child args for sum_arg_5
            return(is_seq(0,(-n)-i590+2*i589)*B_denom_coef_c[i590]*B_theta_coef_cp[(-n)-i590+2*i589]*is_integer((-n)-i590+2*i589)*is_seq((-n)-i590+2*i589,i589-i590))
        
        return((n-i589)*iota_coef[n-i589]*py_sum(sum_arg_5,0,i589))
    
    def sum_arg_4(i573):
        # Child args for sum_arg_4    
        def sum_arg_3(i574):
            # Child args for sum_arg_3    
            def sum_arg_2(i572):
                # Child args for sum_arg_2
                return(is_seq(0,(-n)-i574+2*i573-i572)*Delta_coef_cp[i572]*B_theta_coef_cp[(-n)-i574+2*i573-i572]*is_integer((-n)-i574+2*i573-i572)*is_seq((-n)-i574+2*i573-i572,(-i574)+i573-i572))
            
            return(B_denom_coef_c[i574]*py_sum(sum_arg_2,0,i573-i574))
        
        return((n-i573)*iota_coef[n-i573]*py_sum(sum_arg_3,0,i573))
    
    def sum_arg_1(i565):
        # Child args for sum_arg_1
        return((is_seq(0,n-i565))\
            *(B_denom_coef_c[2*i565-n])\
            *(n-i565)\
            *(B_alpha_coef[n-i565])\
            *(is_integer(n-i565))\
            *(is_seq(n-i565,i565)))
    
    
    out = (2*((-py_sum(sum_arg_9,ceil(0.5*n),floor(n))/2)+py_sum(sum_arg_6,ceil(0.5*n),floor(n))-py_sum(sum_arg_4,ceil(0.5*n),floor(n))-is_seq(0,n-2)*is_integer(n-2)*py_sum(sum_arg_26,0,n-2)-is_seq(0,n-2)*is_integer(n-2)*py_sum(sum_arg_24,0,n-2)-py_sum(sum_arg_22,ceil(0.5*n)-1,floor(n)-2)-py_sum(sum_arg_19,ceil(0.5*n)-1,floor(n)-2)+py_sum(sum_arg_16,ceil(0.5*n),floor(n))+py_sum(sum_arg_14,ceil(0.5*n)-1,floor(n)-2)+is_seq(0,n-2)*is_integer(n-2)*py_sum(sum_arg_12,0,n-2)+py_sum(sum_arg_11,ceil(0.5*n),floor(n))/4-py_sum(sum_arg_1,ceil(0.5*n),floor(n))))/(B_alpha_coef[0]*B_denom_coef_c[0]**2*n)
    return(out)

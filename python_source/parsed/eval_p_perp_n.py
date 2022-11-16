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
    def sum_arg_29(i296):
        # Child args for sum_arg_29    
        def sum_arg_28(i258):
            # Child args for sum_arg_28
            return(B_psi_coef_cp[i258]*diff(Delta_coef_cp[n-i296-i258-2],'phi',1))
        
        return(B_denom_coef_c[i296]*py_sum(sum_arg_28,0,n-i296-2))
    
    def sum_arg_27(i292):
        # Child args for sum_arg_27    
        def sum_arg_26(i290):
            # Child args for sum_arg_26
            return(Delta_coef_cp[i290]*diff(B_psi_coef_cp[n-i292-i290-2],'phi',1))
        
        return(B_denom_coef_c[i292]*py_sum(sum_arg_26,0,n-i292-2))
    
    def sum_arg_25(i287):
        # Child args for sum_arg_25    
        def sum_arg_24(i288):
            # Child args for sum_arg_24    
            def sum_arg_23(i260):
                # Child args for sum_arg_23
                return(B_psi_coef_cp[i260]*diff(Delta_coef_cp[(-n)-i288+2*i287-i260+2],'chi',1))
            
            return(B_denom_coef_c[i288]*is_seq(n-i287-2,i287-i288)*py_sum(sum_arg_23,0,(-n)-i288+2*i287+2))
        
        return(is_seq(0,n-i287-2)*iota_coef[n-i287-2]*is_integer(n-i287-2)*py_sum(sum_arg_24,0,i287))
    
    def sum_arg_22(i283):
        # Child args for sum_arg_22    
        def sum_arg_21(i284):
            # Child args for sum_arg_21    
            def sum_arg_20(i280):
                # Child args for sum_arg_20
                return(Delta_coef_cp[i280]*diff(B_psi_coef_cp[(-n)-i284+2*i283-i280+2],'chi',1))
            
            return(B_denom_coef_c[i284]*is_seq(n-i283-2,i283-i284)*py_sum(sum_arg_20,0,(-n)-i284+2*i283+2))
        
        return(is_seq(0,n-i283-2)*iota_coef[n-i283-2]*is_integer(n-i283-2)*py_sum(sum_arg_21,0,i283))
    
    def sum_arg_19(i273):
        # Child args for sum_arg_19    
        def sum_arg_18(i274):
            # Child args for sum_arg_18
            return(B_denom_coef_c[i274]*Delta_coef_cp[(-n)-i274+2*i273]*is_seq(n-i273,i273-i274))
        
        return((is_seq(0,n-i273)*n-is_seq(0,n-i273)*i273)*B_alpha_coef[n-i273]*is_integer(n-i273)*py_sum(sum_arg_18,0,i273))
    
    def sum_arg_17(i267):
        # Child args for sum_arg_17    
        def sum_arg_16(i268):
            # Child args for sum_arg_16
            return(B_denom_coef_c[i268]*diff(B_psi_coef_cp[(-n)-i268+2*i267+2],'chi',1)*is_seq(n-i267-2,i267-i268))
        
        return(is_seq(0,n-i267-2)*iota_coef[n-i267-2]*is_integer(n-i267-2)*py_sum(sum_arg_16,0,i267))
    
    def sum_arg_15(i266):
        # Child args for sum_arg_15
        return(B_denom_coef_c[i266]*diff(B_psi_coef_cp[n-i266-2],'phi',1))
    
    def sum_arg_14(i261):
        # Child args for sum_arg_14    
        def sum_arg_13(i262):
            # Child args for sum_arg_13
            return(i262*B_denom_coef_c[i262]*Delta_coef_cp[(-n)-i262+2*i261]*is_seq(n-i261,i261-i262))
        
        return(is_seq(0,n-i261)*B_alpha_coef[n-i261]*is_integer(n-i261)*py_sum(sum_arg_13,0,i261))
    
    def sum_arg_12(i253):
        # Child args for sum_arg_12    
        def sum_arg_11(i254):
            # Child args for sum_arg_11    
            def sum_arg_10(i248):
                # Child args for sum_arg_10
                return(B_denom_coef_c[i248]*B_denom_coef_c[(-n)-i254+2*i253-i248])
            
            return(i254*p_perp_coef_cp[i254]*is_seq(n-i253,i253-i254)*py_sum(sum_arg_10,0,(-n)-i254+2*i253))
        
        return(is_seq(0,n-i253)*B_alpha_coef[n-i253]*is_integer(n-i253)*py_sum(sum_arg_11,0,i253))
    
    def sum_arg_9(i293):
        # Child args for sum_arg_9    
        def sum_arg_7(i294):
            # Child args for sum_arg_7
            return(is_seq(0,(-n)-i294+2*i293)*B_denom_coef_c[i294]*B_theta_coef_cp[(-n)-i294+2*i293]*is_integer((-n)-i294+2*i293)*is_seq((-n)-i294+2*i293,i293-i294))
            
        def sum_arg_8(i294):
            # Child args for sum_arg_8
            return(is_seq(0,(-n)-i294+2*i293)*B_denom_coef_c[i294]*B_theta_coef_cp[(-n)-i294+2*i293]*is_integer((-n)-i294+2*i293)*is_seq((-n)-i294+2*i293,i293-i294))
        
        return(iota_coef[n-i293]*(n*py_sum(sum_arg_8,0,i293)-i293*py_sum(sum_arg_7,0,i293)))
    
    def sum_arg_6(i277):
        # Child args for sum_arg_6    
        def sum_arg_3(i278):
            # Child args for sum_arg_3    
            def sum_arg_2(i276):
                # Child args for sum_arg_2
                return(is_seq(0,(-n)-i278+2*i277-i276)*Delta_coef_cp[i276]*B_theta_coef_cp[(-n)-i278+2*i277-i276]*is_integer((-n)-i278+2*i277-i276)*is_seq((-n)-i278+2*i277-i276,(-i278)+i277-i276))
            
            return(B_denom_coef_c[i278]*py_sum(sum_arg_2,0,i277-i278))
            
        def sum_arg_5(i278):
            # Child args for sum_arg_5    
            def sum_arg_4(i276):
                # Child args for sum_arg_4
                return(is_seq(0,(-n)-i278+2*i277-i276)*Delta_coef_cp[i276]*B_theta_coef_cp[(-n)-i278+2*i277-i276]*is_integer((-n)-i278+2*i277-i276)*is_seq((-n)-i278+2*i277-i276,(-i278)+i277-i276))
            
            return(B_denom_coef_c[i278]*py_sum(sum_arg_4,0,i277-i278))
        
        return(iota_coef[n-i277]*(n*py_sum(sum_arg_5,0,i277)-i277*py_sum(sum_arg_3,0,i277)))
    
    def sum_arg_1(i269):
        # Child args for sum_arg_1
        return((is_seq(0,n-i269)*B_denom_coef_c[2*i269-n]*n-is_seq(0,n-i269)*i269*B_denom_coef_c[2*i269-n])*B_alpha_coef[n-i269]*is_integer(n-i269)*is_seq(n-i269,i269))
    
    
    out = -((-4*py_sum_parallel(sum_arg_9,ceil(n/2),floor(n)))+4*py_sum_parallel(sum_arg_6,ceil(n/2),floor(n))+4*is_seq(0,n-2)*is_integer(n-2)*py_sum_parallel(sum_arg_29,0,n-2)+4*is_seq(0,n-2)*is_integer(n-2)*py_sum_parallel(sum_arg_27,0,n-2)+4*py_sum_parallel(sum_arg_25,ceil(n/2)-1,floor(n)-2)+4*py_sum_parallel(sum_arg_22,ceil(n/2)-1,floor(n)-2)-4*py_sum_parallel(sum_arg_19,ceil(n/2),floor(n))-4*py_sum_parallel(sum_arg_17,ceil(n/2)-1,floor(n)-2)-4*is_seq(0,n-2)*is_integer(n-2)*py_sum_parallel(sum_arg_15,0,n-2)-py_sum_parallel(sum_arg_14,ceil(n/2),floor(n))+2*py_sum_parallel(sum_arg_12,ceil(n/2),floor(n))+4*py_sum_parallel(sum_arg_1,ceil(n/2),floor(n)))/(2*B_alpha_coef[0]*B_denom_coef_c[0]**2*n)
    return(out)

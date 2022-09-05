# Evaluates p_perp_n 
# Uses n, B_theta_coef_cp, B_psi_coef_cp, 
# B_alpha_coef, B_denom_coef_c,
# p_perp_coef_cp, delta_coef_cp,iota_coef 
# Must be evaluated with p_perp_n=0 
from math import floor, ceil
from math_utilities import *
import chiphifunc
def eval_p_perp_n_cp(arguments):    
    def sum_arg_26(i294):
        # Child args for sum_arg_26    
        def sum_arg_25(i256):
            # Child args for sum_arg_25
            return(B_psi_coef_cp[i256]*diff(delta_coef_cp[n-i294-i256-2],'phi',1))
        
        return(B_denom_coef_c[i294]*py_sum(sum_arg_25,0,n-i294-2))
    
    def sum_arg_24(i290):
        # Child args for sum_arg_24    
        def sum_arg_23(i288):
            # Child args for sum_arg_23
            return(delta_coef_cp[i288]*diff(B_psi_coef_cp[n-i290-i288-2],'phi',1))
        
        return(B_denom_coef_c[i290]*py_sum(sum_arg_23,0,n-i290-2))
    
    def sum_arg_22(i285):
        # Child args for sum_arg_22    
        def sum_arg_21(i286):
            # Child args for sum_arg_21    
            def sum_arg_20(i258):
                # Child args for sum_arg_20
                return(B_psi_coef_cp[i258]*diff(delta_coef_cp[(-n)-i286+2*i285-i258+2],'chi',1))
            
            return(B_denom_coef_c[i286]*is_seq(n-i285-2,i285-i286)*py_sum(sum_arg_20,0,(-n)-i286+2*i285+2))
        
        return(is_seq(0,n-i285-2)*iota_coef[n-i285-2]*is_integer(n-i285-2)*py_sum(sum_arg_21,0,i285))
    
    def sum_arg_19(i281):
        # Child args for sum_arg_19    
        def sum_arg_18(i282):
            # Child args for sum_arg_18    
            def sum_arg_17(i278):
                # Child args for sum_arg_17
                return(delta_coef_cp[i278]*diff(B_psi_coef_cp[(-n)-i282+2*i281-i278+2],'chi',1))
            
            return(B_denom_coef_c[i282]*is_seq(n-i281-2,i281-i282)*py_sum(sum_arg_17,0,(-n)-i282+2*i281+2))
        
        return(is_seq(0,n-i281-2)*iota_coef[n-i281-2]*is_integer(n-i281-2)*py_sum(sum_arg_18,0,i281))
    
    def sum_arg_16(i271):
        # Child args for sum_arg_16    
        def sum_arg_15(i272):
            # Child args for sum_arg_15
            return(B_denom_coef_c[i272]*delta_coef_cp[(-n)-i272+2*i271]*is_seq(n-i271,i271-i272))
        
        return(is_seq(0,n-i271)*(n-i271)*B_alpha_coef[n-i271]*is_integer(n-i271)*py_sum(sum_arg_15,0,i271))
    
    def sum_arg_14(i265):
        # Child args for sum_arg_14    
        def sum_arg_13(i266):
            # Child args for sum_arg_13
            return(B_denom_coef_c[i266]*diff(B_psi_coef_cp[(-n)-i266+2*i265+2],'chi',1)*is_seq(n-i265-2,i265-i266))
        
        return(is_seq(0,n-i265-2)*iota_coef[n-i265-2]*is_integer(n-i265-2)*py_sum(sum_arg_13,0,i265))
    
    def sum_arg_12(i264):
        # Child args for sum_arg_12
        return(B_denom_coef_c[i264]*diff(B_psi_coef_cp[n-i264-2],'phi',1))
    
    def sum_arg_11(i259):
        # Child args for sum_arg_11    
        def sum_arg_10(i260):
            # Child args for sum_arg_10
            return(i260*B_denom_coef_c[i260]*delta_coef_cp[(-n)-i260+2*i259]*is_seq(n-i259,i259-i260))
        
        return(is_seq(0,n-i259)*B_alpha_coef[n-i259]*is_integer(n-i259)*py_sum(sum_arg_10,0,i259))
    
    def sum_arg_9(i251):
        # Child args for sum_arg_9    
        def sum_arg_8(i252):
            # Child args for sum_arg_8    
            def sum_arg_7(i246):
                # Child args for sum_arg_7
                return(B_denom_coef_c[i246]*B_denom_coef_c[(-n)-i252+2*i251-i246])
            
            return(i252*p_perp_coef_cp[i252]*is_seq(n-i251,i251-i252)*py_sum(sum_arg_7,0,(-n)-i252+2*i251))
        
        return(is_seq(0,n-i251)*B_alpha_coef[n-i251]*is_integer(n-i251)*py_sum(sum_arg_8,0,i251))
    
    def sum_arg_6(i291):
        # Child args for sum_arg_6    
        def sum_arg_5(i292):
            # Child args for sum_arg_5
            return(is_seq(0,(-n)-i292+2*i291)*B_denom_coef_c[i292]*B_theta_coef_cp[(-n)-i292+2*i291]*is_integer((-n)-i292+2*i291)*is_seq((-n)-i292+2*i291,i291-i292))
        
        return((n-i291)*iota_coef[n-i291]*py_sum(sum_arg_5,0,i291))
    
    def sum_arg_4(i275):
        # Child args for sum_arg_4    
        def sum_arg_3(i276):
            # Child args for sum_arg_3    
            def sum_arg_2(i274):
                # Child args for sum_arg_2
                return(is_seq(0,(-n)-i276+2*i275-i274)*delta_coef_cp[i274]*B_theta_coef_cp[(-n)-i276+2*i275-i274]*is_integer((-n)-i276+2*i275-i274)*is_seq((-n)-i276+2*i275-i274,(-i276)+i275-i274))
            
            return(B_denom_coef_c[i276]*py_sum(sum_arg_2,0,i275-i276))
        
        return((n-i275)*iota_coef[n-i275]*py_sum(sum_arg_3,0,i275))
    
    def sum_arg_1(i267):
        # Child args for sum_arg_1
        return((is_seq(0,n-i267))\
            *(B_denom_coef_c[2*i267-n])\
            *(n-i267)\
            *(B_alpha_coef[n-i267])\
            *(is_integer(n-i267))\
            *(is_seq(n-i267,i267)))
    
    
    out = (2*((-py_sum(sum_arg_9,ceil(0.5*n),floor(n))/2)+py_sum(sum_arg_6,ceil(0.5*n),floor(n))-py_sum(sum_arg_4,ceil(0.5*n),floor(n))-is_seq(0,n-2)*is_integer(n-2)*py_sum(sum_arg_26,0,n-2)-is_seq(0,n-2)*is_integer(n-2)*py_sum(sum_arg_24,0,n-2)-py_sum(sum_arg_22,ceil(0.5*n)-1,floor(n)-2)-py_sum(sum_arg_19,ceil(0.5*n)-1,floor(n)-2)+py_sum(sum_arg_16,ceil(0.5*n),floor(n))+py_sum(sum_arg_14,ceil(0.5*n)-1,floor(n)-2)+is_seq(0,n-2)*is_integer(n-2)*py_sum(sum_arg_12,0,n-2)+py_sum(sum_arg_11,ceil(0.5*n),floor(n))/4-py_sum(sum_arg_1,ceil(0.5*n),floor(n))))/(B_alpha_coef[0]*B_denom_coef_c[0]**2*n)
    return(out)

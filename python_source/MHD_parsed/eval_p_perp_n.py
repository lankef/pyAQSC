# Evaluates p_perp_n 
# Uses n, B_theta_coef_cp[n-2], B_psi_coef_cp[n-2], 
# B_alpha_coef[first-order terms], B_denom_coef_c[n],
# p_perp_coef_cp[n-1], Delta_coef_cp[n-1],iota_coef[(n-3)/2 or (n-2)/2] 
# Must be evaluated with p_perp_n=0 
# The zeroth component contains B_psi_0: 
# coeff = -2diff(Delta_coef_cp[0],'phi',1)/(n*B_alpha_coef[0]*B_denom_coef_c[0]) 
# coeff_dphi = -2(Delta_coef_cp[0]-1)/(n*B_alpha_coef[0]*B_denom_coef_c[0]) 
from math import floor, ceil
from math_utilities import *
import chiphifunc
def eval_p_perp_n_cp(arguments):    
    def sum_arg_29(i340):
        # Child args for sum_arg_29    
        def sum_arg_28(i302):
            # Child args for sum_arg_28
            return(B_psi_coef_cp[i302]*diff(Delta_coef_cp[n-i340-i302-2],'phi',1))
        
        return(B_denom_coef_c[i340]*py_sum(sum_arg_28,0,n-i340-2))
    
    def sum_arg_27(i336):
        # Child args for sum_arg_27    
        def sum_arg_26(i334):
            # Child args for sum_arg_26
            return(Delta_coef_cp[i334]*diff(B_psi_coef_cp[n-i336-i334-2],'phi',1))
        
        return(B_denom_coef_c[i336]*py_sum(sum_arg_26,0,n-i336-2))
    
    def sum_arg_25(i331):
        # Child args for sum_arg_25    
        def sum_arg_24(i332):
            # Child args for sum_arg_24    
            def sum_arg_23(i304):
                # Child args for sum_arg_23
                return(B_psi_coef_cp[i304]*diff(Delta_coef_cp[(-n)-i332+2*i331-i304+2],'chi',1))
            
            return(B_denom_coef_c[i332]*is_seq(n-i331-2,i331-i332)*py_sum(sum_arg_23,0,(-n)-i332+2*i331+2))
        
        return(is_seq(0,n-i331-2)*iota_coef[n-i331-2]*is_integer(n-i331-2)*py_sum(sum_arg_24,0,i331))
    
    def sum_arg_22(i327):
        # Child args for sum_arg_22    
        def sum_arg_21(i328):
            # Child args for sum_arg_21    
            def sum_arg_20(i324):
                # Child args for sum_arg_20
                return(Delta_coef_cp[i324]*diff(B_psi_coef_cp[(-n)-i328+2*i327-i324+2],'chi',1))
            
            return(B_denom_coef_c[i328]*is_seq(n-i327-2,i327-i328)*py_sum(sum_arg_20,0,(-n)-i328+2*i327+2))
        
        return(is_seq(0,n-i327-2)*iota_coef[n-i327-2]*is_integer(n-i327-2)*py_sum(sum_arg_21,0,i327))
    
    def sum_arg_19(i317):
        # Child args for sum_arg_19    
        def sum_arg_18(i318):
            # Child args for sum_arg_18
            return(B_denom_coef_c[i318]*Delta_coef_cp[(-n)-i318+2*i317]*is_seq(n-i317,i317-i318))
        
        return((is_seq(0,n-i317)*n-is_seq(0,n-i317)*i317)*B_alpha_coef[n-i317]*is_integer(n-i317)*py_sum(sum_arg_18,0,i317))
    
    def sum_arg_17(i311):
        # Child args for sum_arg_17    
        def sum_arg_16(i312):
            # Child args for sum_arg_16
            return(B_denom_coef_c[i312]*diff(B_psi_coef_cp[(-n)-i312+2*i311+2],'chi',1)*is_seq(n-i311-2,i311-i312))
        
        return(is_seq(0,n-i311-2)*iota_coef[n-i311-2]*is_integer(n-i311-2)*py_sum(sum_arg_16,0,i311))
    
    def sum_arg_15(i310):
        # Child args for sum_arg_15
        return(B_denom_coef_c[i310]*diff(B_psi_coef_cp[n-i310-2],'phi',1))
    
    def sum_arg_14(i305):
        # Child args for sum_arg_14    
        def sum_arg_13(i306):
            # Child args for sum_arg_13
            return(i306*B_denom_coef_c[i306]*Delta_coef_cp[(-n)-i306+2*i305]*is_seq(n-i305,i305-i306))
        
        return(is_seq(0,n-i305)*B_alpha_coef[n-i305]*is_integer(n-i305)*py_sum(sum_arg_13,0,i305))
    
    def sum_arg_12(i297):
        # Child args for sum_arg_12    
        def sum_arg_11(i298):
            # Child args for sum_arg_11    
            def sum_arg_10(i248):
                # Child args for sum_arg_10
                return(B_denom_coef_c[i248]*B_denom_coef_c[(-n)-i298+2*i297-i248])
            
            return(i298*p_perp_coef_cp[i298]*is_seq(n-i297,i297-i298)*py_sum(sum_arg_10,0,(-n)-i298+2*i297))
        
        return(is_seq(0,n-i297)*B_alpha_coef[n-i297]*is_integer(n-i297)*py_sum(sum_arg_11,0,i297))
    
    def sum_arg_9(i337):
        # Child args for sum_arg_9    
        def sum_arg_7(i338):
            # Child args for sum_arg_7
            return(is_seq(0,(-n)-i338+2*i337)*B_denom_coef_c[i338]*B_theta_coef_cp[(-n)-i338+2*i337]*is_integer((-n)-i338+2*i337)*is_seq((-n)-i338+2*i337,i337-i338))
            
        def sum_arg_8(i338):
            # Child args for sum_arg_8
            return(is_seq(0,(-n)-i338+2*i337)*B_denom_coef_c[i338]*B_theta_coef_cp[(-n)-i338+2*i337]*is_integer((-n)-i338+2*i337)*is_seq((-n)-i338+2*i337,i337-i338))
        
        return(iota_coef[n-i337]*(n*py_sum(sum_arg_8,0,i337)-i337*py_sum(sum_arg_7,0,i337)))
    
    def sum_arg_6(i321):
        # Child args for sum_arg_6    
        def sum_arg_3(i322):
            # Child args for sum_arg_3    
            def sum_arg_2(i320):
                # Child args for sum_arg_2
                return(is_seq(0,(-n)-i322+2*i321-i320)*Delta_coef_cp[i320]*B_theta_coef_cp[(-n)-i322+2*i321-i320]*is_integer((-n)-i322+2*i321-i320)*is_seq((-n)-i322+2*i321-i320,(-i322)+i321-i320))
            
            return(B_denom_coef_c[i322]*py_sum(sum_arg_2,0,i321-i322))
            
        def sum_arg_5(i322):
            # Child args for sum_arg_5    
            def sum_arg_4(i320):
                # Child args for sum_arg_4
                return(is_seq(0,(-n)-i322+2*i321-i320)*Delta_coef_cp[i320]*B_theta_coef_cp[(-n)-i322+2*i321-i320]*is_integer((-n)-i322+2*i321-i320)*is_seq((-n)-i322+2*i321-i320,(-i322)+i321-i320))
            
            return(B_denom_coef_c[i322]*py_sum(sum_arg_4,0,i321-i322))
        
        return(iota_coef[n-i321]*(n*py_sum(sum_arg_5,0,i321)-i321*py_sum(sum_arg_3,0,i321)))
    
    def sum_arg_1(i313):
        # Child args for sum_arg_1
        return((is_seq(0,n-i313)*B_denom_coef_c[2*i313-n]*n-is_seq(0,n-i313)*i313*B_denom_coef_c[2*i313-n])*B_alpha_coef[n-i313]*is_integer(n-i313)*is_seq(n-i313,i313))
    
    
    out = -((-4*py_sum_parallel(sum_arg_9,ceil(n/2),floor(n)))+4*py_sum_parallel(sum_arg_6,ceil(n/2),floor(n))+4*is_seq(0,n-2)*is_integer(n-2)*py_sum_parallel(sum_arg_29,0,n-2)+4*is_seq(0,n-2)*is_integer(n-2)*py_sum_parallel(sum_arg_27,0,n-2)+4*py_sum_parallel(sum_arg_25,ceil(n/2)-1,floor(n)-2)+4*py_sum_parallel(sum_arg_22,ceil(n/2)-1,floor(n)-2)-4*py_sum_parallel(sum_arg_19,ceil(n/2),floor(n))-4*py_sum_parallel(sum_arg_17,ceil(n/2)-1,floor(n)-2)-4*is_seq(0,n-2)*is_integer(n-2)*py_sum_parallel(sum_arg_15,0,n-2)-py_sum_parallel(sum_arg_14,ceil(n/2),floor(n))+2*py_sum_parallel(sum_arg_12,ceil(n/2),floor(n))+4*py_sum_parallel(sum_arg_1,ceil(n/2),floor(n)))/(2*B_alpha_coef[0]*B_denom_coef_c[0]**2*n)
    return(out)

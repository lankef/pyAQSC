# Coefficient for Yn+1
from math import floor, ceil
from math_utilities import is_seq, py_sum, is_integer, diff
import chiphifunc
def coef_for_y(n, B_alpha_coef[0], X_coef_cp):    
    
    out = (B_alpha_coef[0]*diff(X_coef_cp[1],'chi',1)*((-n)-1))/2
    return(out)


# Coefficient for dchi Yn+1
def coef_for_dchi_y(B_alpha_coef[0], X_coef_cp):    
    
    out = (B_alpha_coef[0]*X_coef_cp[1])/2
    return(out)


# RHS - LHS 
def rhs_minus_lhs(n,
    X_coef_cp,
    Y_coef_cp,
    Z_coef_cp,
    B_psi_coef_cp,
    B_theta_coef_cp,
    B_alpha_coef,
    kap_p, dl_p, tau_p,
    iota_coef):    
    def sum_arg_31(i194):
        # Child args for sum_arg_31    
        def sum_arg_30(i192):
            # Child args for sum_arg_30
            return(B_theta_coef_cp[i192]*X_coef_cp[n-i194-i192+2])
        
        return(i194*X_coef_cp[i194]*py_sum(sum_arg_30,0,n-i194+2))
    
    def sum_arg_29(i190):
        # Child args for sum_arg_29    
        def sum_arg_28(i188):
            # Child args for sum_arg_28
            return(B_theta_coef_cp[i188]*Y_coef_cp[n-i190-i188+2])
        
        return(i190*Y_coef_cp[i190]*py_sum(sum_arg_28,0,n-i190+2))
    
    def sum_arg_27(i186):
        # Child args for sum_arg_27    
        def sum_arg_26(i184):
            # Child args for sum_arg_26
            return(B_psi_coef_cp[i184]*X_coef_cp[n-i186-i184])
        
        return(diff(X_coef_cp[i186],'chi',1)*py_sum(sum_arg_26,0,n-i186))
    
    def sum_arg_25(i182):
        # Child args for sum_arg_25    
        def sum_arg_24(i180):
            # Child args for sum_arg_24
            return(B_psi_coef_cp[i180]*Y_coef_cp[n-i182-i180])
        
        return(diff(Y_coef_cp[i182],'chi',1)*py_sum(sum_arg_24,0,n-i182))
    
    def sum_arg_23(i320):
        # Child args for sum_arg_23    
        def sum_arg_22(i212):
            # Child args for sum_arg_22
            return(B_psi_coef_cp[i212]*diff(X_coef_cp[n-i320-i212],'phi',1))
        
        return(diff(Y_coef_cp[i320],'chi',1)*py_sum(sum_arg_22,0,n-i320))
    
    def sum_arg_21(i318):
        # Child args for sum_arg_21    
        def sum_arg_20(i208):
            # Child args for sum_arg_20
            return(B_psi_coef_cp[i208]*diff(X_coef_cp[n-i318-i208],'chi',1))
        
        return(diff(Y_coef_cp[i318],'phi',1)*py_sum(sum_arg_20,0,n-i318))
    
    def sum_arg_19(i321):
        # Child args for sum_arg_19    
        def sum_arg_18(i322):
            # Child args for sum_arg_18
            return(diff(Y_coef_cp[i322],'chi',1)*X_coef_cp[(-n)-i322+2*i321-2]*((-n)-i322+2*i321-2)*is_seq(n-i321+2,i321-i322))
        
        return(is_seq(0,n-i321+2)*B_alpha_coef[n-i321+2]*is_integer(n-i321+2)*py_sum(sum_arg_18,0,i321))
    
    def sum_arg_17(i314):
        # Child args for sum_arg_17    
        def sum_arg_16(i210):
            # Child args for sum_arg_16
            return(B_theta_coef_cp[i210]*(n-i314-i210+2)*X_coef_cp[n-i314-i210+2])
        
        return(diff(Y_coef_cp[i314],'phi',1)*py_sum(sum_arg_16,0,n-i314+2))
    
    def sum_arg_15(i221):
        # Child args for sum_arg_15    
        def sum_arg_14(i222):
            # Child args for sum_arg_14
            return(diff(X_coef_cp[i222],'chi',1)*Y_coef_cp[(-n)-i222+2*i221-2]*((-n)-i222+2*i221-2)*is_seq(n-i221+2,i221-i222))
        
        return(is_seq(0,n-i221+2)*B_alpha_coef[n-i221+2]*is_integer(n-i221+2)*py_sum(sum_arg_14,0,i221))
    
    def sum_arg_13(i218):
        # Child args for sum_arg_13    
        def sum_arg_12(i216):
            # Child args for sum_arg_12
            return(B_theta_coef_cp[i216]*(n-i218-i216+2)*Y_coef_cp[n-i218-i216+2])
        
        return(diff(X_coef_cp[i218],'phi',1)*py_sum(sum_arg_12,0,n-i218+2))
    
    def sum_arg_11(i202):
        # Child args for sum_arg_11    
        def sum_arg_10(i200):
            # Child args for sum_arg_10
            return(B_theta_coef_cp[i200]*(n-i202-i200+2)*Y_coef_cp[n-i202-i200+2])
        
        return(Z_coef_cp[i202]*py_sum(sum_arg_10,0,n-i202+2))
    
    def sum_arg_9(i198):
        # Child args for sum_arg_9    
        def sum_arg_8(i196):
            # Child args for sum_arg_8
            return(B_psi_coef_cp[i196]*Z_coef_cp[n-i198-i196])
        
        return(diff(Y_coef_cp[i198],'chi',1)*py_sum(sum_arg_8,0,n-i198))
    
    def sum_arg_7(i323):
        # Child args for sum_arg_7    
        def sum_arg_6(i324):
            # Child args for sum_arg_6    
            def sum_arg_5(i780):
                # Child args for sum_arg_5
                return(diff(X_coef_cp[(-i780)-i324+i323],'chi',1)*i780*Y_coef_cp[i780])
            
            return(is_seq(0,(-n)+i324+i323-2)*B_theta_coef_cp[(-n)+i324+i323-2]*is_integer((-n)+i324+i323-2)*is_seq((-n)+i324+i323-2,i324)*py_sum(sum_arg_5,0,i323-i324))
        
        return(iota_coef[n-i323+2]*py_sum(sum_arg_6,0,i323))
    
    def sum_arg_4(i315):
        # Child args for sum_arg_4    
        def sum_arg_3(i316):
            # Child args for sum_arg_3    
            def sum_arg_2(i764):
                # Child args for sum_arg_2
                return(diff(Y_coef_cp[(-i764)-i316+i315],'chi',1)*i764*X_coef_cp[i764])
            
            return(is_seq(0,(-n)+i316+i315-2)*B_theta_coef_cp[(-n)+i316+i315-2]*is_integer((-n)+i316+i315-2)*is_seq((-n)+i316+i315-2,i316)*py_sum(sum_arg_2,0,i315-i316))
        
        return(iota_coef[n-i315+2]*py_sum(sum_arg_3,0,i315))
    
    def sum_arg_1(i223):
        # Child args for sum_arg_1
        return(is_seq(0,n-i223)*diff(Z_coef_cp[2*i223-n],'chi',1)*iota_coef[n-i223]*is_integer(n-i223)*is_seq(n-i223,i223))
    
    
    out = (-(is_seq(0,n+2)*dl_p*is_integer(n+2)*py_sum(sum_arg_31,0,n+2)*tau_p)/2)\
        +(-(is_seq(0,n+2)*dl_p*is_integer(n+2)*py_sum(sum_arg_29,0,n+2)*tau_p)/2)\
        +(is_seq(0,n)*dl_p*is_integer(n)*py_sum(sum_arg_27,0,n)*tau_p)\
        +(is_seq(0,n)*dl_p*is_integer(n)*py_sum(sum_arg_25,0,n)*tau_p)\
        +(is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum(sum_arg_9,0,n))\
        +(-py_sum(sum_arg_7,ceil(0.5*n)+1,floor(n)+2)/2)\
        +(py_sum(sum_arg_4,ceil(0.5*n)+1,floor(n)+2)/2)\
        +(is_seq(0,n)*is_integer(n)*py_sum(sum_arg_23,0,n))\
        +(-is_seq(0,n)*is_integer(n)*py_sum(sum_arg_21,0,n))\
        +(-py_sum(sum_arg_19,ceil(0.5*n)+1,floor(n)+2)/2)\
        +((is_seq(0,n+2)*is_integer(n+2)*py_sum(sum_arg_17,0,n+2))/2)\
        +(py_sum(sum_arg_15,ceil(0.5*n)+1,floor(n)+2)/2)\
        +(-(is_seq(0,n+2)*is_integer(n+2)*py_sum(sum_arg_13,0,n+2))/2)\
        +(-(is_seq(0,n+2)*dl_p*kap_p*is_integer(n+2)*py_sum(sum_arg_11,0,n+2))/2)\
        +(py_sum(sum_arg_1,ceil(0.5*n),floor(n)))\
        +(is_seq(0,n)*is_integer(n)*diff(Z_coef_cp[n],'phi',1))\
        +(-is_seq(0,n)*dl_p*kap_p*X_coef_cp[n]*is_integer(n))
    return(out)

# Coefficient for Yn+1 - only X1 and Balpha0 needed
from math import floor, ceil
from math_utilities import *
import chiphifunc
def coef_a(n, B_alpha_coef, X_coef_cp):    
    
    out = (B_alpha_coef[0]*diff(X_coef_cp[1],'chi',1)*((-n)-1))/2
    return(out)


# Coefficient for dchi Yn+1  - only X1 and Balpha0 needed
def coef_b(B_alpha_coef, X_coef_cp):    
    
    out = (B_alpha_coef[0]*X_coef_cp[1])/2
    return(out)


# RHS - LHS 
# Used in (conv(a) + conv(b)@dchi)@Yn+1 = RHS - LHS(Yn+1 = 0) 
# Must run with Yn+1=0.# Depends on Xn+1, Yn, Zn, B_theta n, B_psi n-2
# iota (n-2)/2 or (n-3)/2, B_alpha n/2 or (n-1)/2.
def rhs_minus_lhs(n,
    X_coef_cp,
    Y_coef_cp,
    Z_coef_cp,
    B_psi_coef_cp,
    B_theta_coef_cp,
    B_alpha_coef,
    kap_p, dl_p, tau_p,
    iota_coef):    
    def sum_arg_31(i78):
        # Child args for sum_arg_31    
        def sum_arg_30(i76):
            # Child args for sum_arg_30
            return(B_theta_coef_cp[i76]*X_coef_cp[n-i78-i76+2])
        
        return(i78*X_coef_cp[i78]*py_sum(sum_arg_30,0,n-i78+2))
    
    def sum_arg_29(i74):
        # Child args for sum_arg_29    
        def sum_arg_28(i72):
            # Child args for sum_arg_28
            return(B_theta_coef_cp[i72]*Y_coef_cp[n-i74-i72+2])
        
        return(i74*Y_coef_cp[i74]*py_sum(sum_arg_28,0,n-i74+2))
    
    def sum_arg_27(i70):
        # Child args for sum_arg_27    
        def sum_arg_26(i68):
            # Child args for sum_arg_26
            return(B_psi_coef_cp[i68]*X_coef_cp[n-i70-i68])
        
        return(diff(X_coef_cp[i70],'chi',1)*py_sum(sum_arg_26,0,n-i70))
    
    def sum_arg_25(i66):
        # Child args for sum_arg_25    
        def sum_arg_24(i64):
            # Child args for sum_arg_24
            return(B_psi_coef_cp[i64]*Y_coef_cp[n-i66-i64])
        
        return(diff(Y_coef_cp[i66],'chi',1)*py_sum(sum_arg_24,0,n-i66))
    
    def sum_arg_23(i258):
        # Child args for sum_arg_23    
        def sum_arg_22(i96):
            # Child args for sum_arg_22
            return(B_psi_coef_cp[i96]*diff(X_coef_cp[n-i96-i258],'phi',1))
        
        return(diff(Y_coef_cp[i258],'chi',1)*py_sum(sum_arg_22,0,n-i258))
    
    def sum_arg_21(i256):
        # Child args for sum_arg_21    
        def sum_arg_20(i92):
            # Child args for sum_arg_20
            return(B_psi_coef_cp[i92]*diff(X_coef_cp[n-i92-i256],'chi',1))
        
        return(diff(Y_coef_cp[i256],'phi',1)*py_sum(sum_arg_20,0,n-i256))
    
    def sum_arg_19(i252):
        # Child args for sum_arg_19    
        def sum_arg_18(i94):
            # Child args for sum_arg_18
            return(B_theta_coef_cp[i94]*(n-i94-i252+2)*X_coef_cp[n-i94-i252+2])
        
        return(diff(Y_coef_cp[i252],'phi',1)*py_sum(sum_arg_18,0,n-i252+2))
    
    def sum_arg_17(i86):
        # Child args for sum_arg_17    
        def sum_arg_16(i84):
            # Child args for sum_arg_16
            return(B_theta_coef_cp[i84]*(n-i86-i84+2)*Y_coef_cp[n-i86-i84+2])
        
        return(Z_coef_cp[i86]*py_sum(sum_arg_16,0,n-i86+2))
    
    def sum_arg_15(i82):
        # Child args for sum_arg_15    
        def sum_arg_14(i80):
            # Child args for sum_arg_14
            return(B_psi_coef_cp[i80]*Z_coef_cp[n-i82-i80])
        
        return(diff(Y_coef_cp[i82],'chi',1)*py_sum(sum_arg_14,0,n-i82))
    
    def sum_arg_13(i259):
        # Child args for sum_arg_13    
        def sum_arg_12(i260):
            # Child args for sum_arg_12
            return(diff(Y_coef_cp[i260],'chi',1)*X_coef_cp[(-n)-i260+2*i259-2]*((-n)-i260+2*i259-2)*is_seq(n-i259+2,i259-i260))
        
        return(is_seq(0,n-i259+2)*B_alpha_coef[n-i259+2]*is_integer(n-i259+2)*py_sum(sum_arg_12,0,i259))
    
    def sum_arg_11(i105):
        # Child args for sum_arg_11    
        def sum_arg_10(i106):
            # Child args for sum_arg_10
            return(diff(X_coef_cp[i106],'chi',1)*Y_coef_cp[(-n)-i106+2*i105-2]*((-n)-i106+2*i105-2)*is_seq(n-i105+2,i105-i106))
        
        return(is_seq(0,n-i105+2)*B_alpha_coef[n-i105+2]*is_integer(n-i105+2)*py_sum(sum_arg_10,0,i105))
    
    def sum_arg_9(i102):
        # Child args for sum_arg_9    
        def sum_arg_8(i100):
            # Child args for sum_arg_8
            return(B_theta_coef_cp[i100]*(n-i102-i100+2)*Y_coef_cp[n-i102-i100+2])
        
        return(diff(X_coef_cp[i102],'phi',1)*py_sum(sum_arg_8,0,n-i102+2))
    
    def sum_arg_7(i261):
        # Child args for sum_arg_7    
        def sum_arg_6(i262):
            # Child args for sum_arg_6    
            def sum_arg_5(i718):
                # Child args for sum_arg_5
                return(diff(X_coef_cp[(-i718)-i262+i261],'chi',1)*i718*Y_coef_cp[i718])
            
            return(is_seq(0,(-n)+i262+i261-2)*B_theta_coef_cp[(-n)+i262+i261-2]*is_integer((-n)+i262+i261-2)*is_seq((-n)+i262+i261-2,i262)*py_sum(sum_arg_5,0,i261-i262))
        
        return(iota_coef[n-i261+2]*py_sum(sum_arg_6,0,i261))
    
    def sum_arg_4(i253):
        # Child args for sum_arg_4    
        def sum_arg_3(i254):
            # Child args for sum_arg_3    
            def sum_arg_2(i702):
                # Child args for sum_arg_2
                return(diff(Y_coef_cp[(-i702)-i254+i253],'chi',1)*i702*X_coef_cp[i702])
            
            return(is_seq(0,(-n)+i254+i253-2)*B_theta_coef_cp[(-n)+i254+i253-2]*is_integer((-n)+i254+i253-2)*is_seq((-n)+i254+i253-2,i254)*py_sum(sum_arg_2,0,i253-i254))
        
        return(iota_coef[n-i253+2]*py_sum(sum_arg_3,0,i253))
    
    def sum_arg_1(i107):
        # Child args for sum_arg_1
        return(is_seq(0,n-i107)*diff(Z_coef_cp[2*i107-n],'chi',1)*iota_coef[n-i107]*is_integer(n-i107)*is_seq(n-i107,i107))
    
    
    out = (-(is_seq(0,n+2)*dl_p*is_integer(n+2)*py_sum(sum_arg_31,0,n+2)*tau_p)/2)\
        +(-(is_seq(0,n+2)*dl_p*is_integer(n+2)*py_sum(sum_arg_29,0,n+2)*tau_p)/2)\
        +(is_seq(0,n)*dl_p*is_integer(n)*py_sum(sum_arg_27,0,n)*tau_p)\
        +(is_seq(0,n)*dl_p*is_integer(n)*py_sum(sum_arg_25,0,n)*tau_p)\
        +(-(is_seq(0,n+2)*is_integer(n+2)*py_sum(sum_arg_9,0,n+2))/2)\
        +(-py_sum(sum_arg_7,ceil(0.5*n)+1,floor(n)+2)/2)\
        +(py_sum(sum_arg_4,ceil(0.5*n)+1,floor(n)+2)/2)\
        +(is_seq(0,n)*is_integer(n)*py_sum(sum_arg_23,0,n))\
        +(-is_seq(0,n)*is_integer(n)*py_sum(sum_arg_21,0,n))\
        +((is_seq(0,n+2)*is_integer(n+2)*py_sum(sum_arg_19,0,n+2))/2)\
        +(-(is_seq(0,n+2)*dl_p*kap_p*is_integer(n+2)*py_sum(sum_arg_17,0,n+2))/2)\
        +(is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum(sum_arg_15,0,n))\
        +(-py_sum(sum_arg_13,ceil(0.5*n)+1,floor(n)+2)/2)\
        +(py_sum(sum_arg_11,ceil(0.5*n)+1,floor(n)+2)/2)\
        +(py_sum(sum_arg_1,ceil(0.5*n),floor(n)))\
        +((B_alpha_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[n+1],'chi',1))/2)\
        +(is_seq(0,n)*is_integer(n)*diff(Z_coef_cp[n],'phi',1))\
        +((B_alpha_coef[0]*diff(X_coef_cp[1],'chi',1)*((-n)-1)*Y_coef_cp[n+1])/2)\
        +(-is_seq(0,n)*dl_p*kap_p*X_coef_cp[n]*is_integer(n))
    return(out)

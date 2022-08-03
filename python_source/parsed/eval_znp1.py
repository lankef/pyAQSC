# Evaluates Zn+1. Requires X[..., n], Y[..., n], Z[..., n], 
# B_theta_coef_cp[..., n+1], B_psi_coef_cp[..., n], 
# B_alpha_coef [..., (n-1)/2 or (n-2)/2] 
# iota_coef [..., (n-1)/2 or (n-2)/2] 
# kap_p, dl_p, tau_p
from math import floor, ceil
from math_utilities import is_seq, py_sum, is_integer, diff
import chiphifunc
def eval_Znp1_cp(n, X_coef_cp, Y_coef_cp, Z_coef_cp, \
    B_theta_coef_cp, B_psi_coef_cp, \
    B_alpha_coef, \
    kap_p, dl_p, tau_p, iota_coef):    
    def sum_arg_62(i158):
        # Child args for sum_arg_62    
        def sum_arg_61(i156):
            # Child args for sum_arg_61
            return(B_theta_coef_cp[i156]*Y_coef_cp[n-i158-i156+2])
        
        return(i158*Z_coef_cp[i158]*py_sum(sum_arg_61,0,n-i158+2))
    
    def sum_arg_60(i154):
        # Child args for sum_arg_60    
        def sum_arg_59(i152):
            # Child args for sum_arg_59
            return(B_psi_coef_cp[i152]*Y_coef_cp[n-i154-i152])
        
        return(diff(Z_coef_cp[i154],'chi',1)*py_sum(sum_arg_59,0,n-i154))
    
    def sum_arg_58(i682):
        # Child args for sum_arg_58    
        def sum_arg_57(i180):
            # Child args for sum_arg_57
            return(B_psi_coef_cp[i180]*diff(X_coef_cp[n-i682-i180],'chi',1))
        
        return(diff(Z_coef_cp[i682],'phi',1)*py_sum(sum_arg_57,0,n-i682))
    
    def sum_arg_56(i676):
        # Child args for sum_arg_56    
        def sum_arg_55(i184):
            # Child args for sum_arg_55
            return(B_psi_coef_cp[i184]*diff(X_coef_cp[n-i676-i184],'phi',1))
        
        return(diff(Z_coef_cp[i676],'chi',1)*py_sum(sum_arg_55,0,n-i676))
    
    def sum_arg_54(i679):
        # Child args for sum_arg_54    
        def sum_arg_53(i680):
            # Child args for sum_arg_53
            return(diff(Z_coef_cp[i680],'chi',1)*X_coef_cp[(-n)-i680+2*i679-2]*((-n)-i680+2*i679-2)*is_seq(n-i679+2,i679-i680))
        
        return(is_seq(0,n-i679+2)*B_alpha_coef[n-i679+2]*is_integer(n-i679+2)*py_sum(sum_arg_53,0,i679))
    
    def sum_arg_52(i678):
        # Child args for sum_arg_52
        return(B_psi_coef_cp[i678]*diff(X_coef_cp[n-i678],'chi',1))
    
    def sum_arg_51(i672):
        # Child args for sum_arg_51    
        def sum_arg_50(i182):
            # Child args for sum_arg_50
            return(B_theta_coef_cp[i182]*(n-i672-i182+2)*X_coef_cp[n-i672-i182+2])
        
        return(diff(Z_coef_cp[i672],'phi',1)*py_sum(sum_arg_50,0,n-i672+2))
    
    def sum_arg_49(i669):
        # Child args for sum_arg_49    
        def sum_arg_48(i670):
            # Child args for sum_arg_48
            return(diff(X_coef_cp[i670],'chi',1)*Z_coef_cp[(-n)-i670+2*i669-2]*((-n)-i670+2*i669-2)*is_seq(n-i669+2,i669-i670))
        
        return(is_seq(0,n-i669+2)*B_alpha_coef[n-i669+2]*is_integer(n-i669+2)*py_sum(sum_arg_48,0,i669))
    
    def sum_arg_47(i190):
        # Child args for sum_arg_47    
        def sum_arg_46(i188):
            # Child args for sum_arg_46
            return(B_theta_coef_cp[i188]*(n-i190-i188+2)*Z_coef_cp[n-i190-i188+2])
        
        return(diff(X_coef_cp[i190],'phi',1)*py_sum(sum_arg_46,0,n-i190+2))
    
    def sum_arg_45(i174):
        # Child args for sum_arg_45    
        def sum_arg_44(i172):
            # Child args for sum_arg_44
            return(B_theta_coef_cp[i172]*X_coef_cp[n-i174-i172+2])
        
        return(i174*X_coef_cp[i174]*py_sum(sum_arg_44,0,n-i174+2))
    
    def sum_arg_43(i170):
        # Child args for sum_arg_43    
        def sum_arg_42(i168):
            # Child args for sum_arg_42
            return(B_theta_coef_cp[i168]*Z_coef_cp[n-i170-i168+2])
        
        return(i170*Z_coef_cp[i170]*py_sum(sum_arg_42,0,n-i170+2))
    
    def sum_arg_41(i166):
        # Child args for sum_arg_41    
        def sum_arg_40(i164):
            # Child args for sum_arg_40
            return(B_psi_coef_cp[i164]*X_coef_cp[n-i166-i164])
        
        return(diff(X_coef_cp[i166],'chi',1)*py_sum(sum_arg_40,0,n-i166))
    
    def sum_arg_39(i162):
        # Child args for sum_arg_39    
        def sum_arg_38(i160):
            # Child args for sum_arg_38
            return(B_psi_coef_cp[i160]*Z_coef_cp[n-i162-i160])
        
        return(diff(Z_coef_cp[i162],'chi',1)*py_sum(sum_arg_38,0,n-i162))
    
    def sum_arg_37(i683):
        # Child args for sum_arg_37    
        def sum_arg_36(i684):
            # Child args for sum_arg_36    
            def sum_arg_35(i1139):
                # Child args for sum_arg_35
                return(i1139*Z_coef_cp[i1139]*diff(X_coef_cp[(-i684)+i683-i1139],'chi',1))
            
            return(is_seq(0,(-n)+i684+i683-2)*B_theta_coef_cp[(-n)+i684+i683-2]*is_integer((-n)+i684+i683-2)*is_seq((-n)+i684+i683-2,i684)*py_sum(sum_arg_35,0,i683-i684))
        
        return(iota_coef[n-i683+2]*py_sum(sum_arg_36,0,i683))
    
    def sum_arg_34(i673):
        # Child args for sum_arg_34    
        def sum_arg_33(i674):
            # Child args for sum_arg_33    
            def sum_arg_32(i1123):
                # Child args for sum_arg_32
                return(i1123*X_coef_cp[i1123]*diff(Z_coef_cp[(-i674)+i673-i1123],'chi',1))
            
            return(is_seq(0,(-n)+i674+i673-2)*B_theta_coef_cp[(-n)+i674+i673-2]*is_integer((-n)+i674+i673-2)*is_seq((-n)+i674+i673-2,i674)*py_sum(sum_arg_32,0,i673-i674))
        
        return(iota_coef[n-i673+2]*py_sum(sum_arg_33,0,i673))
    
    def sum_arg_31(i194):
        # Child args for sum_arg_31
        return(B_theta_coef_cp[i194]*(n-i194+2)*X_coef_cp[n-i194+2])
    
    def sum_arg_30(i118):
        # Child args for sum_arg_30    
        def sum_arg_29(i116):
            # Child args for sum_arg_29
            return(B_theta_coef_cp[i116]*X_coef_cp[n-i118-i116+2])
        
        return(i118*Z_coef_cp[i118]*py_sum(sum_arg_29,0,n-i118+2))
    
    def sum_arg_28(i114):
        # Child args for sum_arg_28    
        def sum_arg_27(i112):
            # Child args for sum_arg_27
            return(B_psi_coef_cp[i112]*X_coef_cp[n-i114-i112])
        
        return(diff(Z_coef_cp[i114],'chi',1)*py_sum(sum_arg_27,0,n-i114))
    
    def sum_arg_26(i208):
        # Child args for sum_arg_26    
        def sum_arg_25(i136):
            # Child args for sum_arg_25
            return(B_psi_coef_cp[i136]*diff(Y_coef_cp[n-i208-i136],'phi',1))
        
        return(diff(Z_coef_cp[i208],'chi',1)*py_sum(sum_arg_25,0,n-i208))
    
    def sum_arg_24(i202):
        # Child args for sum_arg_24    
        def sum_arg_23(i132):
            # Child args for sum_arg_23
            return(B_psi_coef_cp[i132]*diff(Y_coef_cp[n-i202-i132],'chi',1))
        
        return(diff(Z_coef_cp[i202],'phi',1)*py_sum(sum_arg_23,0,n-i202))
    
    def sum_arg_22(i210):
        # Child args for sum_arg_22    
        def sum_arg_21(i134):
            # Child args for sum_arg_21
            return(B_theta_coef_cp[i134]*(n-i210-i134+2)*Y_coef_cp[n-i210-i134+2])
        
        return(diff(Z_coef_cp[i210],'phi',1)*py_sum(sum_arg_21,0,n-i210+2))
    
    def sum_arg_20(i205):
        # Child args for sum_arg_20    
        def sum_arg_19(i206):
            # Child args for sum_arg_19
            return(diff(Y_coef_cp[i206],'chi',1)*Z_coef_cp[(-n)-i206+2*i205-2]*((-n)-i206+2*i205-2)*is_seq(n-i205+2,i205-i206))
        
        return(is_seq(0,n-i205+2)*B_alpha_coef[n-i205+2]*is_integer(n-i205+2)*py_sum(sum_arg_19,0,i205))
    
    def sum_arg_18(i204):
        # Child args for sum_arg_18
        return(B_psi_coef_cp[i204]*diff(Y_coef_cp[n-i204],'chi',1))
    
    def sum_arg_17(i197):
        # Child args for sum_arg_17    
        def sum_arg_16(i198):
            # Child args for sum_arg_16
            return(diff(Z_coef_cp[i198],'chi',1)*Y_coef_cp[(-n)-i198+2*i197-2]*((-n)-i198+2*i197-2)*is_seq(n-i197+2,i197-i198))
        
        return(is_seq(0,n-i197+2)*B_alpha_coef[n-i197+2]*is_integer(n-i197+2)*py_sum(sum_arg_16,0,i197))
    
    def sum_arg_15(i142):
        # Child args for sum_arg_15    
        def sum_arg_14(i140):
            # Child args for sum_arg_14
            return(B_theta_coef_cp[i140]*(n-i142-i140+2)*Z_coef_cp[n-i142-i140+2])
        
        return(diff(Y_coef_cp[i142],'phi',1)*py_sum(sum_arg_14,0,n-i142+2))
    
    def sum_arg_13(i126):
        # Child args for sum_arg_13    
        def sum_arg_12(i124):
            # Child args for sum_arg_12
            return(B_theta_coef_cp[i124]*X_coef_cp[n-i126-i124+2])
        
        return(i126*Y_coef_cp[i126]*py_sum(sum_arg_12,0,n-i126+2))
    
    def sum_arg_11(i122):
        # Child args for sum_arg_11    
        def sum_arg_10(i120):
            # Child args for sum_arg_10
            return(B_psi_coef_cp[i120]*X_coef_cp[n-i122-i120])
        
        return(diff(Y_coef_cp[i122],'chi',1)*py_sum(sum_arg_10,0,n-i122))
    
    def sum_arg_9(i211):
        # Child args for sum_arg_9    
        def sum_arg_8(i212):
            # Child args for sum_arg_8    
            def sum_arg_7(i668):
                # Child args for sum_arg_7
                return(diff(Z_coef_cp[(-i668)-i212+i211],'chi',1)*i668*Y_coef_cp[i668])
            
            return(is_seq(0,(-n)+i212+i211-2)*B_theta_coef_cp[(-n)+i212+i211-2]*is_integer((-n)+i212+i211-2)*is_seq((-n)+i212+i211-2,i212)*py_sum(sum_arg_7,0,i211-i212))
        
        return(iota_coef[n-i211+2]*py_sum(sum_arg_8,0,i211))
    
    def sum_arg_6(i199):
        # Child args for sum_arg_6    
        def sum_arg_5(i200):
            # Child args for sum_arg_5    
            def sum_arg_4(i652):
                # Child args for sum_arg_4
                return(diff(Y_coef_cp[(-i652)-i200+i199],'chi',1)*i652*Z_coef_cp[i652])
            
            return(is_seq(0,(-n)+i200+i199-2)*B_theta_coef_cp[(-n)+i200+i199-2]*is_integer((-n)+i200+i199-2)*is_seq((-n)+i200+i199-2,i200)*py_sum(sum_arg_4,0,i199-i200))
        
        return(iota_coef[n-i199+2]*py_sum(sum_arg_5,0,i199))
    
    def sum_arg_3(i146):
        # Child args for sum_arg_3
        return(B_theta_coef_cp[i146]*(n-i146+2)*Y_coef_cp[n-i146+2])
    
    def sum_arg_2(i147):
        # Child args for sum_arg_2
        return(is_seq(0,n-i147)*diff(X_coef_cp[2*i147-n],'chi',1)*iota_coef[n-i147]*is_integer(n-i147)*is_seq(n-i147,i147))
    
    def sum_arg_1(i195):
        # Child args for sum_arg_1
        return(is_seq(0,n-i195)*diff(Y_coef_cp[2*i195-n],'chi',1)*iota_coef[n-i195]*is_integer(n-i195)*is_seq(n-i195,i195))
    
    
    out = ((-Y_coef_cp[1]*((-(is_seq(0,n+2)*dl_p*is_integer(n+2)*py_sum(sum_arg_62,0,n+2)*tau_p)/2)+is_seq(0,n)*dl_p*is_integer(n)*py_sum(sum_arg_60,0,n)*tau_p-is_seq(0,n)*is_integer(n)*py_sum(sum_arg_58,0,n)+is_seq(0,n)*is_integer(n)*py_sum(sum_arg_56,0,n)-py_sum(sum_arg_54,ceil(0.5*n)+1,floor(n)+2)/2-is_seq(0,n)*dl_p*is_integer(n)*py_sum(sum_arg_52,0,n)+(is_seq(0,n+2)*is_integer(n+2)*py_sum(sum_arg_51,0,n+2))/2+py_sum(sum_arg_49,ceil(0.5*n)+1,floor(n)+2)/2-(is_seq(0,n+2)*is_integer(n+2)*py_sum(sum_arg_47,0,n+2))/2-(is_seq(0,n+2)*dl_p*kap_p*is_integer(n+2)*py_sum(sum_arg_45,0,n+2))/2-(is_seq(0,n+2)*dl_p*kap_p*is_integer(n+2)*py_sum(sum_arg_43,0,n+2))/2+is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum(sum_arg_41,0,n)+is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum(sum_arg_39,0,n)-py_sum(sum_arg_37,ceil(0.5*n)+1,floor(n)+2)/2+py_sum(sum_arg_34,ceil(0.5*n)+1,floor(n)+2)/2+(is_seq(0,n+2)*dl_p*is_integer(n+2)*py_sum(sum_arg_31,0,n+2))/2))-X_coef_cp[1]*((-(is_seq(0,n+2)*dl_p*is_integer(n+2)*py_sum(sum_arg_30,0,n+2)*tau_p)/2)+is_seq(0,n)*dl_p*is_integer(n)*py_sum(sum_arg_28,0,n)*tau_p-py_sum(sum_arg_9,ceil(0.5*n)+1,floor(n)+2)/2+py_sum(sum_arg_6,ceil(0.5*n)+1,floor(n)+2)/2-(is_seq(0,n+2)*dl_p*is_integer(n+2)*py_sum(sum_arg_3,0,n+2))/2-is_seq(0,n)*is_integer(n)*py_sum(sum_arg_26,0,n)+is_seq(0,n)*is_integer(n)*py_sum(sum_arg_24,0,n)-(is_seq(0,n+2)*is_integer(n+2)*py_sum(sum_arg_22,0,n+2))/2-py_sum(sum_arg_20,ceil(0.5*n)+1,floor(n)+2)/2+is_seq(0,n)*dl_p*is_integer(n)*py_sum(sum_arg_18,0,n)+py_sum(sum_arg_17,ceil(0.5*n)+1,floor(n)+2)/2+(is_seq(0,n+2)*is_integer(n+2)*py_sum(sum_arg_15,0,n+2))/2+(is_seq(0,n+2)*dl_p*kap_p*is_integer(n+2)*py_sum(sum_arg_13,0,n+2))/2-is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum(sum_arg_11,0,n))+X_coef_cp[1]*(is_seq(0,n)*dl_p*Y_coef_cp[n]*is_integer(n)*tau_p+py_sum(sum_arg_2,ceil(0.5*n),floor(n))+is_seq(0,n)*is_integer(n)*diff(X_coef_cp[n],'phi',1)+is_seq(0,n)*dl_p*kap_p*Z_coef_cp[n]*is_integer(n))+Y_coef_cp[1]*((-is_seq(0,n)*dl_p*X_coef_cp[n]*is_integer(n)*tau_p)+py_sum(sum_arg_1,ceil(0.5*n),floor(n))+is_seq(0,n)*is_integer(n)*diff(Y_coef_cp[n],'phi',1)))/(dl_p*((-n)-1))
    return(out)

# Evaluates Zn. See Zn+1 for requirements.
def eval_Zn_cp(n, X_coef_cp, Y_coef_cp, Z_coef_cp, 
    B_theta_coef_cp, B_psi_coef_cp, 
    B_alpha_coef, 
    kap_p, dl_p, tau_p, iota_coef):
    
    return(eval_Znp1_cp(n-1, X_coef_cp, Y_coef_cp, Z_coef_cp, 
        B_theta_coef_cp, B_psi_coef_cp, 
        B_alpha_coef, 
        kap_p, dl_p, tau_p, iota_coef))


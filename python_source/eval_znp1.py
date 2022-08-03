# Coefficient for Zn+1
from math import floor, ceil
from math_utilities import is_seq, py_sum, is_integer, diff
import chiphifunc
def coef_for_z(n, dl_p):    
    
    out = dl_p*((-n)-1)
    return(out)


# t*Y1+k*X1, RHS - LHS 
def kt_rhs_minus_lhs(n, X_coef_cp, Y_coef_cp, Z_coef_cp, \
    B_denom_coef_c, B_alpha_coef, \
    kap_p, dl_p, tau_p, iota_coef):    
    def sum_arg_62(i274):
        # Child args for sum_arg_62    
        def sum_arg_61(i272):
            # Child args for sum_arg_61
            return(B_theta_coef_cp[i272]*Y_coef_cp[n-i274-i272+2])
        
        return(i274*Z_coef_cp[i274]*py_sum(sum_arg_61,0,n-i274+2))
    
    def sum_arg_60(i270):
        # Child args for sum_arg_60    
        def sum_arg_59(i268):
            # Child args for sum_arg_59
            return(B_psi_coef_cp[i268]*Y_coef_cp[n-i270-i268])
        
        return(diff(Z_coef_cp[i270],'chi',1)*py_sum(sum_arg_59,0,n-i270))
    
    def sum_arg_58(i1260):
        # Child args for sum_arg_58    
        def sum_arg_57(i300):
            # Child args for sum_arg_57
            return(B_psi_coef_cp[i300]*diff(X_coef_cp[n-i300-i1260],'phi',1))
        
        return(diff(Z_coef_cp[i1260],'chi',1)*py_sum(sum_arg_57,0,n-i1260))
    
    def sum_arg_56(i1266):
        # Child args for sum_arg_56    
        def sum_arg_55(i296):
            # Child args for sum_arg_55
            return(B_psi_coef_cp[i296]*diff(X_coef_cp[n-i296-i1266],'chi',1))
        
        return(diff(Z_coef_cp[i1266],'phi',1)*py_sum(sum_arg_55,0,n-i1266))
    
    def sum_arg_54(i306):
        # Child args for sum_arg_54    
        def sum_arg_53(i304):
            # Child args for sum_arg_53
            return(B_theta_coef_cp[i304]*(n-i306-i304+2)*Z_coef_cp[n-i306-i304+2])
        
        return(diff(X_coef_cp[i306],'phi',1)*py_sum(sum_arg_53,0,n-i306+2))
    
    def sum_arg_52(i1256):
        # Child args for sum_arg_52    
        def sum_arg_51(i298):
            # Child args for sum_arg_51
            return(B_theta_coef_cp[i298]*(n-i298-i1256+2)*X_coef_cp[n-i298-i1256+2])
        
        return(diff(Z_coef_cp[i1256],'phi',1)*py_sum(sum_arg_51,0,n-i1256+2))
    
    def sum_arg_50(i290):
        # Child args for sum_arg_50    
        def sum_arg_49(i288):
            # Child args for sum_arg_49
            return(B_theta_coef_cp[i288]*X_coef_cp[n-i290-i288+2])
        
        return(i290*X_coef_cp[i290]*py_sum(sum_arg_49,0,n-i290+2))
    
    def sum_arg_48(i286):
        # Child args for sum_arg_48    
        def sum_arg_47(i284):
            # Child args for sum_arg_47
            return(B_theta_coef_cp[i284]*Z_coef_cp[n-i286-i284+2])
        
        return(i286*Z_coef_cp[i286]*py_sum(sum_arg_47,0,n-i286+2))
    
    def sum_arg_46(i282):
        # Child args for sum_arg_46    
        def sum_arg_45(i280):
            # Child args for sum_arg_45
            return(B_psi_coef_cp[i280]*X_coef_cp[n-i282-i280])
        
        return(diff(X_coef_cp[i282],'chi',1)*py_sum(sum_arg_45,0,n-i282))
    
    def sum_arg_44(i278):
        # Child args for sum_arg_44    
        def sum_arg_43(i276):
            # Child args for sum_arg_43
            return(B_psi_coef_cp[i276]*Z_coef_cp[n-i278-i276])
        
        return(diff(Z_coef_cp[i278],'chi',1)*py_sum(sum_arg_43,0,n-i278))
    
    def sum_arg_42(i1263):
        # Child args for sum_arg_42    
        def sum_arg_41(i1264):
            # Child args for sum_arg_41
            return(diff(Z_coef_cp[i1264],'chi',1)*X_coef_cp[(-n)-i1264+2*i1263-2]*((-n)-i1264+2*i1263-2)*is_seq(n-i1263+2,i1263-i1264))
        
        return(is_seq(0,n-i1263+2)*B_alpha_coef[n-i1263+2]*is_integer(n-i1263+2)*py_sum(sum_arg_41,0,i1263))
    
    def sum_arg_40(i1262):
        # Child args for sum_arg_40
        return(B_psi_coef_cp[i1262]*diff(X_coef_cp[n-i1262],'chi',1))
    
    def sum_arg_39(i1253):
        # Child args for sum_arg_39    
        def sum_arg_38(i1254):
            # Child args for sum_arg_38
            return(diff(X_coef_cp[i1254],'chi',1)*Z_coef_cp[(-n)-i1254+2*i1253-2]*((-n)-i1254+2*i1253-2)*is_seq(n-i1253+2,i1253-i1254))
        
        return(is_seq(0,n-i1253+2)*B_alpha_coef[n-i1253+2]*is_integer(n-i1253+2)*py_sum(sum_arg_38,0,i1253))
    
    def sum_arg_37(i310):
        # Child args for sum_arg_37
        return(B_theta_coef_cp[i310]*(n-i310+2)*X_coef_cp[n-i310+2])
    
    def sum_arg_36(i1267):
        # Child args for sum_arg_36    
        def sum_arg_35(i1268):
            # Child args for sum_arg_35    
            def sum_arg_34(i1724):
                # Child args for sum_arg_34
                return(diff(X_coef_cp[(-i1724)-i1268+i1267],'chi',1)*i1724*Z_coef_cp[i1724])
            
            return(is_seq(0,(-n)+i1268+i1267-2)*B_theta_coef_cp[(-n)+i1268+i1267-2]*is_integer((-n)+i1268+i1267-2)*is_seq((-n)+i1268+i1267-2,i1268)*py_sum(sum_arg_34,0,i1267-i1268))
        
        return(iota_coef[n-i1267+2]*py_sum(sum_arg_35,0,i1267))
    
    def sum_arg_33(i1257):
        # Child args for sum_arg_33    
        def sum_arg_32(i1258):
            # Child args for sum_arg_32    
            def sum_arg_31(i1708):
                # Child args for sum_arg_31
                return(diff(Z_coef_cp[(-i1708)-i1258+i1257],'chi',1)*i1708*X_coef_cp[i1708])
            
            return(is_seq(0,(-n)+i1258+i1257-2)*B_theta_coef_cp[(-n)+i1258+i1257-2]*is_integer((-n)+i1258+i1257-2)*is_seq((-n)+i1258+i1257-2,i1258)*py_sum(sum_arg_31,0,i1257-i1258))
        
        return(iota_coef[n-i1257+2]*py_sum(sum_arg_32,0,i1257))
    
    def sum_arg_30(i234):
        # Child args for sum_arg_30    
        def sum_arg_29(i232):
            # Child args for sum_arg_29
            return(B_theta_coef_cp[i232]*X_coef_cp[n-i234-i232+2])
        
        return(i234*Z_coef_cp[i234]*py_sum(sum_arg_29,0,n-i234+2))
    
    def sum_arg_28(i230):
        # Child args for sum_arg_28    
        def sum_arg_27(i228):
            # Child args for sum_arg_27
            return(B_psi_coef_cp[i228]*X_coef_cp[n-i230-i228])
        
        return(diff(Z_coef_cp[i230],'chi',1)*py_sum(sum_arg_27,0,n-i230))
    
    def sum_arg_26(i792):
        # Child args for sum_arg_26    
        def sum_arg_25(i252):
            # Child args for sum_arg_25
            return(B_psi_coef_cp[i252]*diff(Y_coef_cp[n-i792-i252],'phi',1))
        
        return(diff(Z_coef_cp[i792],'chi',1)*py_sum(sum_arg_25,0,n-i792))
    
    def sum_arg_24(i786):
        # Child args for sum_arg_24    
        def sum_arg_23(i248):
            # Child args for sum_arg_23
            return(B_psi_coef_cp[i248]*diff(Y_coef_cp[n-i786-i248],'chi',1))
        
        return(diff(Z_coef_cp[i786],'phi',1)*py_sum(sum_arg_23,0,n-i786))
    
    def sum_arg_22(i794):
        # Child args for sum_arg_22    
        def sum_arg_21(i250):
            # Child args for sum_arg_21
            return(B_theta_coef_cp[i250]*(n-i794-i250+2)*Y_coef_cp[n-i794-i250+2])
        
        return(diff(Z_coef_cp[i794],'phi',1)*py_sum(sum_arg_21,0,n-i794+2))
    
    def sum_arg_20(i789):
        # Child args for sum_arg_20    
        def sum_arg_19(i790):
            # Child args for sum_arg_19
            return(diff(Y_coef_cp[i790],'chi',1)*Z_coef_cp[(-n)-i790+2*i789-2]*((-n)-i790+2*i789-2)*is_seq(n-i789+2,i789-i790))
        
        return(is_seq(0,n-i789+2)*B_alpha_coef[n-i789+2]*is_integer(n-i789+2)*py_sum(sum_arg_19,0,i789))
    
    def sum_arg_18(i788):
        # Child args for sum_arg_18
        return(B_psi_coef_cp[i788]*diff(Y_coef_cp[n-i788],'chi',1))
    
    def sum_arg_17(i781):
        # Child args for sum_arg_17    
        def sum_arg_16(i782):
            # Child args for sum_arg_16
            return(diff(Z_coef_cp[i782],'chi',1)*Y_coef_cp[(-n)-i782+2*i781-2]*((-n)-i782+2*i781-2)*is_seq(n-i781+2,i781-i782))
        
        return(is_seq(0,n-i781+2)*B_alpha_coef[n-i781+2]*is_integer(n-i781+2)*py_sum(sum_arg_16,0,i781))
    
    def sum_arg_15(i258):
        # Child args for sum_arg_15    
        def sum_arg_14(i256):
            # Child args for sum_arg_14
            return(B_theta_coef_cp[i256]*(n-i258-i256+2)*Z_coef_cp[n-i258-i256+2])
        
        return(diff(Y_coef_cp[i258],'phi',1)*py_sum(sum_arg_14,0,n-i258+2))
    
    def sum_arg_13(i242):
        # Child args for sum_arg_13    
        def sum_arg_12(i240):
            # Child args for sum_arg_12
            return(B_theta_coef_cp[i240]*X_coef_cp[n-i242-i240+2])
        
        return(i242*Y_coef_cp[i242]*py_sum(sum_arg_12,0,n-i242+2))
    
    def sum_arg_11(i238):
        # Child args for sum_arg_11    
        def sum_arg_10(i236):
            # Child args for sum_arg_10
            return(B_psi_coef_cp[i236]*X_coef_cp[n-i238-i236])
        
        return(diff(Y_coef_cp[i238],'chi',1)*py_sum(sum_arg_10,0,n-i238))
    
    def sum_arg_9(i795):
        # Child args for sum_arg_9    
        def sum_arg_8(i796):
            # Child args for sum_arg_8    
            def sum_arg_7(i1251):
                # Child args for sum_arg_7
                return(i1251*Y_coef_cp[i1251]*diff(Z_coef_cp[(-i796)+i795-i1251],'chi',1))
            
            return(is_seq(0,(-n)+i796+i795-2)*B_theta_coef_cp[(-n)+i796+i795-2]*is_integer((-n)+i796+i795-2)*is_seq((-n)+i796+i795-2,i796)*py_sum(sum_arg_7,0,i795-i796))
        
        return(iota_coef[n-i795+2]*py_sum(sum_arg_8,0,i795))
    
    def sum_arg_6(i783):
        # Child args for sum_arg_6    
        def sum_arg_5(i784):
            # Child args for sum_arg_5    
            def sum_arg_4(i1235):
                # Child args for sum_arg_4
                return(i1235*Z_coef_cp[i1235]*diff(Y_coef_cp[(-i784)+i783-i1235],'chi',1))
            
            return(is_seq(0,(-n)+i784+i783-2)*B_theta_coef_cp[(-n)+i784+i783-2]*is_integer((-n)+i784+i783-2)*is_seq((-n)+i784+i783-2,i784)*py_sum(sum_arg_4,0,i783-i784))
        
        return(iota_coef[n-i783+2]*py_sum(sum_arg_5,0,i783))
    
    def sum_arg_3(i262):
        # Child args for sum_arg_3
        return(B_theta_coef_cp[i262]*(n-i262+2)*Y_coef_cp[n-i262+2])
    
    def sum_arg_2(i263):
        # Child args for sum_arg_2
        return(is_seq(0,n-i263)*diff(X_coef_cp[2*i263-n],'chi',1)*iota_coef[n-i263]*is_integer(n-i263)*is_seq(n-i263,i263))
    
    def sum_arg_1(i311):
        # Child args for sum_arg_1
        return(is_seq(0,n-i311)*diff(Y_coef_cp[2*i311-n],'chi',1)*iota_coef[n-i311]*is_integer(n-i311)*is_seq(n-i311,i311))
    
    
    out = (-Y_coef_cp[1]*((-(is_seq(0,n+2)*dl_p*is_integer(n+2)*py_sum(sum_arg_62,0,n+2)*tau_p)/2)+is_seq(0,n)*dl_p*is_integer(n)*py_sum(sum_arg_60,0,n)*tau_p+is_seq(0,n)*is_integer(n)*py_sum(sum_arg_58,0,n)-is_seq(0,n)*is_integer(n)*py_sum(sum_arg_56,0,n)-(is_seq(0,n+2)*is_integer(n+2)*py_sum(sum_arg_54,0,n+2))/2+(is_seq(0,n+2)*is_integer(n+2)*py_sum(sum_arg_52,0,n+2))/2-(is_seq(0,n+2)*dl_p*kap_p*is_integer(n+2)*py_sum(sum_arg_50,0,n+2))/2-(is_seq(0,n+2)*dl_p*kap_p*is_integer(n+2)*py_sum(sum_arg_48,0,n+2))/2+is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum(sum_arg_46,0,n)+is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum(sum_arg_44,0,n)-py_sum(sum_arg_42,ceil(0.5*n)+1,floor(n)+2)/2-is_seq(0,n)*dl_p*is_integer(n)*py_sum(sum_arg_40,0,n)+py_sum(sum_arg_39,ceil(0.5*n)+1,floor(n)+2)/2+(is_seq(0,n+2)*dl_p*is_integer(n+2)*py_sum(sum_arg_37,0,n+2))/2-py_sum(sum_arg_36,ceil(0.5*n)+1,floor(n)+2)/2+py_sum(sum_arg_33,ceil(0.5*n)+1,floor(n)+2)/2))-X_coef_cp[1]*((-(is_seq(0,n+2)*dl_p*is_integer(n+2)*py_sum(sum_arg_30,0,n+2)*tau_p)/2)+is_seq(0,n)*dl_p*is_integer(n)*py_sum(sum_arg_28,0,n)*tau_p-py_sum(sum_arg_9,ceil(0.5*n)+1,floor(n)+2)/2+py_sum(sum_arg_6,ceil(0.5*n)+1,floor(n)+2)/2-(is_seq(0,n+2)*dl_p*is_integer(n+2)*py_sum(sum_arg_3,0,n+2))/2-is_seq(0,n)*is_integer(n)*py_sum(sum_arg_26,0,n)+is_seq(0,n)*is_integer(n)*py_sum(sum_arg_24,0,n)-(is_seq(0,n+2)*is_integer(n+2)*py_sum(sum_arg_22,0,n+2))/2-py_sum(sum_arg_20,ceil(0.5*n)+1,floor(n)+2)/2+is_seq(0,n)*dl_p*is_integer(n)*py_sum(sum_arg_18,0,n)+py_sum(sum_arg_17,ceil(0.5*n)+1,floor(n)+2)/2+(is_seq(0,n+2)*is_integer(n+2)*py_sum(sum_arg_15,0,n+2))/2+(is_seq(0,n+2)*dl_p*kap_p*is_integer(n+2)*py_sum(sum_arg_13,0,n+2))/2-is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum(sum_arg_11,0,n))+X_coef_cp[1]*(is_seq(0,n)*dl_p*Y_coef_cp[n]*is_integer(n)*tau_p+py_sum(sum_arg_2,ceil(0.5*n),floor(n))+is_seq(0,n)*is_integer(n)*diff(X_coef_cp[n],'phi',1)+is_seq(0,n)*dl_p*kap_p*Z_coef_cp[n]*is_integer(n))+Y_coef_cp[1]*((-is_seq(0,n)*dl_p*X_coef_cp[n]*is_integer(n)*tau_p)+py_sum(sum_arg_1,ceil(0.5*n),floor(n))+is_seq(0,n)*is_integer(n)*diff(Y_coef_cp[n],'phi',1))
    return(out)

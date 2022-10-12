# Ynp1s1
# Used in (conv(a) + conv(b)@dchi)@Yn+1 = RHS - LHS(Yn+1 = 0)
# Must run with Yn+1=0.# Depends on Xn+1, Yn, Zn, B_theta n, B_psi n-2
# iota (n-2)/2 or (n-3)/2, B_alpha n/2 or (n-1)/2.
# Ysn is the chi-indep component
from math import floor, ceil
from math_utilities import *
import chiphifunc
def evaluate_ynp1s1_full(n,
    X_coef_cp,
    Y_coef_cp,
    Z_coef_cp,
    B_psi_coef_cp,
    B_theta_coef_cp,
    B_alpha_coef,
    kap_p, dl_p, tau_p, eta,
    iota_coef):
    def sum_arg_31(i82):
        # Child args for sum_arg_31
        def sum_arg_30(i80):
            # Child args for sum_arg_30
            return(B_theta_coef_cp[i80]*X_coef_cp[n-i82-i80+2])

        return(i82*X_coef_cp[i82]*py_sum(sum_arg_30,0,n-i82+2))

    def sum_arg_29(i78):
        # Child args for sum_arg_29
        def sum_arg_28(i76):
            # Child args for sum_arg_28
            return(B_theta_coef_cp[i76]*Y_coef_cp[n-i78-i76+2])

        return(i78*Y_coef_cp[i78]*py_sum(sum_arg_28,0,n-i78+2))

    def sum_arg_27(i74):
        # Child args for sum_arg_27
        def sum_arg_26(i72):
            # Child args for sum_arg_26
            return(B_psi_coef_cp[i72]*X_coef_cp[n-i74-i72])

        return(diff(X_coef_cp[i74],'chi',1)*py_sum(sum_arg_26,0,n-i74))

    def sum_arg_25(i70):
        # Child args for sum_arg_25
        def sum_arg_24(i68):
            # Child args for sum_arg_24
            return(B_psi_coef_cp[i68]*Y_coef_cp[n-i70-i68])

        return(diff(Y_coef_cp[i70],'chi',1)*py_sum(sum_arg_24,0,n-i70))

    def sum_arg_23(i726):
        # Child args for sum_arg_23
        def sum_arg_22(i96):
            # Child args for sum_arg_22
            return(B_psi_coef_cp[i96]*diff(X_coef_cp[n-i96-i726],'chi',1))

        return(diff(Y_coef_cp[i726],'phi',1)*py_sum(sum_arg_22,0,n-i726))

    def sum_arg_21(i730):
        # Child args for sum_arg_21
        def sum_arg_20(i100):
            # Child args for sum_arg_20
            return(B_psi_coef_cp[i100]*diff(X_coef_cp[n-i730-i100],'phi',1))

        return(diff(Y_coef_cp[i730],'chi',1)*py_sum(sum_arg_20,0,n-i730))

    def sum_arg_19(i722):
        # Child args for sum_arg_19
        def sum_arg_18(i98):
            # Child args for sum_arg_18
            return((B_theta_coef_cp[i98]*n+((-i98)-i722+2)*B_theta_coef_cp[i98])*X_coef_cp[n-i98-i722+2])

        return(diff(Y_coef_cp[i722],'phi',1)*py_sum(sum_arg_18,0,n-i722+2))

    def sum_arg_17(i90):
        # Child args for sum_arg_17
        def sum_arg_16(i88):
            # Child args for sum_arg_16
            return((B_theta_coef_cp[i88]*n-B_theta_coef_cp[i88]*i90+(2-i88)*B_theta_coef_cp[i88])*Y_coef_cp[n-i90-i88+2])

        return(Z_coef_cp[i90]*py_sum(sum_arg_16,0,n-i90+2))

    def sum_arg_15(i86):
        # Child args for sum_arg_15
        def sum_arg_14(i84):
            # Child args for sum_arg_14
            return(B_psi_coef_cp[i84]*Z_coef_cp[n-i86-i84])

        return(diff(Y_coef_cp[i86],'chi',1)*py_sum(sum_arg_14,0,n-i86))

    def sum_arg_13(i727):
        # Child args for sum_arg_13
        def sum_arg_12(i728):
            # Child args for sum_arg_12
            return((((-i728)+2*i727-2)*diff(Y_coef_cp[i728],'chi',1)*X_coef_cp[(-n)-i728+2*i727-2]-diff(Y_coef_cp[i728],'chi',1)*X_coef_cp[(-n)-i728+2*i727-2]*n)*is_seq(n-i727+2,i727-i728))

        return(is_seq(0,n-i727+2)*B_alpha_coef[n-i727+2]*is_integer(n-i727+2)*py_sum(sum_arg_12,0,i727))

    def sum_arg_11(i109):
        # Child args for sum_arg_11
        def sum_arg_10(i110):
            # Child args for sum_arg_10
            return((((-i110)+2*i109-2)*diff(X_coef_cp[i110],'chi',1)*Y_coef_cp[(-n)-i110+2*i109-2]-diff(X_coef_cp[i110],'chi',1)*Y_coef_cp[(-n)-i110+2*i109-2]*n)*is_seq(n-i109+2,i109-i110))

        return(is_seq(0,n-i109+2)*B_alpha_coef[n-i109+2]*is_integer(n-i109+2)*py_sum(sum_arg_10,0,i109))

    def sum_arg_9(i106):
        # Child args for sum_arg_9
        def sum_arg_8(i104):
            # Child args for sum_arg_8
            return((B_theta_coef_cp[i104]*n-B_theta_coef_cp[i104]*i106+(2-i104)*B_theta_coef_cp[i104])*Y_coef_cp[n-i106-i104+2])

        return(diff(X_coef_cp[i106],'phi',1)*py_sum(sum_arg_8,0,n-i106+2))

    def sum_arg_7(i731):
        # Child args for sum_arg_7
        def sum_arg_6(i732):
            # Child args for sum_arg_6
            def sum_arg_5(i1187):
                # Child args for sum_arg_5
                return(i1187*Y_coef_cp[i1187]*diff(X_coef_cp[(-i732)+i731-i1187],'chi',1))

            return(is_seq(0,(-n)+i732+i731-2)*B_theta_coef_cp[(-n)+i732+i731-2]*is_integer((-n)+i732+i731-2)*is_seq((-n)+i732+i731-2,i732)*py_sum(sum_arg_5,0,i731-i732))

        return(iota_coef[n-i731+2]*py_sum(sum_arg_6,0,i731))

    def sum_arg_4(i723):
        # Child args for sum_arg_4
        def sum_arg_3(i724):
            # Child args for sum_arg_3
            def sum_arg_2(i1171):
                # Child args for sum_arg_2
                return(i1171*X_coef_cp[i1171]*diff(Y_coef_cp[(-i724)+i723-i1171],'chi',1))

            return(is_seq(0,(-n)+i724+i723-2)*B_theta_coef_cp[(-n)+i724+i723-2]*is_integer((-n)+i724+i723-2)*is_seq((-n)+i724+i723-2,i724)*py_sum(sum_arg_2,0,i723-i724))

        return(iota_coef[n-i723+2]*py_sum(sum_arg_3,0,i723))

    def sum_arg_1(i111):
        # Child args for sum_arg_1
        return(is_seq(0,n-i111)*diff(Z_coef_cp[2*i111-n],'chi',1)*iota_coef[n-i111]*is_integer(n-i111)*is_seq(n-i111,i111))


    out = -((2*is_seq(0,n+2)*dl_p*kap_p*is_integer(n+2)*py_sum_parallel(sum_arg_31,0,n+2)+2*is_seq(0,n+2)*dl_p*kap_p*is_integer(n+2)*py_sum_parallel(sum_arg_29,0,n+2)-4*is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_27,0,n)-4*is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_25,0,n))*tau_p+2*is_seq(0,n+2)*kap_p*is_integer(n+2)*py_sum_parallel(sum_arg_9,0,n+2)+kap_p*(2*py_sum_parallel(sum_arg_7,ceil(n/2)+1,floor(n)+2)-2*py_sum_parallel(sum_arg_4,ceil(n/2)+1,floor(n)+2))+4*is_seq(0,n)*kap_p*is_integer(n)*py_sum_parallel(sum_arg_23,0,n)-4*is_seq(0,n)*kap_p*is_integer(n)*py_sum_parallel(sum_arg_21,0,n)-2*is_seq(0,n+2)*kap_p*is_integer(n+2)*py_sum_parallel(sum_arg_19,0,n+2)+2*is_seq(0,n+2)*dl_p*kap_p**2*is_integer(n+2)*py_sum_parallel(sum_arg_17,0,n+2)-4*is_seq(0,n)*dl_p*kap_p**2*is_integer(n)*py_sum_parallel(sum_arg_15,0,n)+2*kap_p*py_sum_parallel(sum_arg_13,ceil(n/2)+1,floor(n)+2)-2*kap_p*py_sum_parallel(sum_arg_11,ceil(n/2)+1,floor(n)+2)-4*kap_p*py_sum_parallel(sum_arg_1,ceil(n/2),floor(n))-4*is_seq(0,n)*kap_p*is_integer(n)*diff(Z_coef_cp[n],'phi',1)+4*is_seq(0,n)*dl_p*kap_p**2*X_coef_cp[n]*is_integer(n))/(B_alpha_coef[0]*eta*n+2*B_alpha_coef[0]*eta)
    return(out)

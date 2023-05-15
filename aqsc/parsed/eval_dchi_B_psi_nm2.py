# Evaluating dchi B_psi_n-2. No masking needed.
# Uses Xn-1, Yn-1, Zn-1,  B_theta_n, B_psi_n-3, iota_coef (n-1)/2 or (n-2)/2
# Must be evaluated with Z_coef_cp[n] = 0 and B_psi_coef_cp[n-2] = 0
from math import floor, ceil
from aqsc.math_utilities import *
def eval_dchi_B_psi_cp_nm2(n, X_coef_cp, Y_coef_cp, Z_coef_cp, \
    B_theta_coef_cp, B_psi_coef_cp, B_alpha_coef, B_denom_coef_c, \
    kap_p, dl_p, tau_p, iota_coef):
    def sum_arg_30(i288):
        # Child args for sum_arg_30
        return(X_coef_cp[i288]*diff(Y_coef_cp[n-i288],True,1))

    def sum_arg_29(i284):
        # Child args for sum_arg_29
        return(Y_coef_cp[i284]*diff(X_coef_cp[n-i284],True,1))

    def sum_arg_28(i256):
        # Child args for sum_arg_28
        return(i256*X_coef_cp[i256]*diff(Y_coef_cp[n-i256],True,1)+i256*diff(X_coef_cp[i256],True,1)*Y_coef_cp[n-i256])

    def sum_arg_27(i254):
        # Child args for sum_arg_27
        return((X_coef_cp[i254]*n-i254*X_coef_cp[i254])*diff(Y_coef_cp[n-i254],True,1)+(diff(X_coef_cp[i254],True,1)*n-i254*diff(X_coef_cp[i254],True,1))*Y_coef_cp[n-i254])

    def sum_arg_26(i290):
        # Child args for sum_arg_26
        return(X_coef_cp[i290]*diff(Z_coef_cp[n-i290],True,1))

    def sum_arg_25(i286):
        # Child args for sum_arg_25
        return(Z_coef_cp[i286]*diff(X_coef_cp[n-i286],True,1))

    def sum_arg_24(i281):
        # Child args for sum_arg_24
        def sum_arg_23(i282):
            # Child args for sum_arg_23
            return(diff(Z_coef_cp[i282],True,1)*diff(Z_coef_cp[(-n)-i282+2*i281],True,1)*is_seq(n-i281,i281-i282))

        return(is_seq(0,n-i281)*iota_coef[n-i281]*is_integer(n-i281)*py_sum(sum_arg_23,0,i281))

    def sum_arg_22(i280):
        # Child args for sum_arg_22
        return(diff(Z_coef_cp[i280],True,1)*diff(Z_coef_cp[n-i280],False,1))

    def sum_arg_21(i277):
        # Child args for sum_arg_21
        def sum_arg_20(i278):
            # Child args for sum_arg_20
            return(diff(Y_coef_cp[i278],True,1)*diff(Y_coef_cp[(-n)-i278+2*i277],True,1)*is_seq(n-i277,i277-i278))

        return(is_seq(0,n-i277)*iota_coef[n-i277]*is_integer(n-i277)*py_sum(sum_arg_20,0,i277))

    def sum_arg_19(i276):
        # Child args for sum_arg_19
        return(diff(Y_coef_cp[i276],True,1)*diff(Y_coef_cp[n-i276],False,1))

    def sum_arg_18(i273):
        # Child args for sum_arg_18
        def sum_arg_17(i274):
            # Child args for sum_arg_17
            return(diff(X_coef_cp[i274],True,1)*diff(X_coef_cp[(-n)-i274+2*i273],True,1)*is_seq(n-i273,i273-i274))

        return(is_seq(0,n-i273)*iota_coef[n-i273]*is_integer(n-i273)*py_sum(sum_arg_17,0,i273))

    def sum_arg_16(i272):
        # Child args for sum_arg_16
        return(diff(X_coef_cp[i272],True,1)*diff(X_coef_cp[n-i272],False,1))

    def sum_arg_15(i269):
        # Child args for sum_arg_15
        def sum_arg_14(i270):
            # Child args for sum_arg_14
            return((i270*Z_coef_cp[i270]*diff(Z_coef_cp[(-n)-i270+2*i269],True,2)+i270*diff(Z_coef_cp[i270],True,1)*diff(Z_coef_cp[(-n)-i270+2*i269],True,1))*is_seq(n-i269,i269-i270))

        return(is_seq(0,n-i269)*iota_coef[n-i269]*is_integer(n-i269)*py_sum(sum_arg_14,0,i269))

    def sum_arg_13(i267):
        # Child args for sum_arg_13
        def sum_arg_12(i268):
            # Child args for sum_arg_12
            return((i268*Y_coef_cp[i268]*diff(Y_coef_cp[(-n)-i268+2*i267],True,2)+i268*diff(Y_coef_cp[i268],True,1)*diff(Y_coef_cp[(-n)-i268+2*i267],True,1))*is_seq(n-i267,i267-i268))

        return(is_seq(0,n-i267)*iota_coef[n-i267]*is_integer(n-i267)*py_sum(sum_arg_12,0,i267))

    def sum_arg_11(i265):
        # Child args for sum_arg_11
        def sum_arg_10(i266):
            # Child args for sum_arg_10
            return((i266*X_coef_cp[i266]*diff(X_coef_cp[(-n)-i266+2*i265],True,2)+i266*diff(X_coef_cp[i266],True,1)*diff(X_coef_cp[(-n)-i266+2*i265],True,1))*is_seq(n-i265,i265-i266))

        return(is_seq(0,n-i265)*iota_coef[n-i265]*is_integer(n-i265)*py_sum(sum_arg_10,0,i265))

    def sum_arg_9(i264):
        # Child args for sum_arg_9
        return(i264*diff(Z_coef_cp[i264],True,1)*diff(Z_coef_cp[n-i264],False,1)+i264*Z_coef_cp[i264]*diff(Z_coef_cp[n-i264],True,1,False,1))

    def sum_arg_8(i262):
        # Child args for sum_arg_8
        return(i262*diff(Y_coef_cp[i262],True,1)*diff(Y_coef_cp[n-i262],False,1)+i262*Y_coef_cp[i262]*diff(Y_coef_cp[n-i262],True,1,False,1))

    def sum_arg_7(i260):
        # Child args for sum_arg_7
        return(i260*diff(X_coef_cp[i260],True,1)*diff(X_coef_cp[n-i260],False,1)+i260*X_coef_cp[i260]*diff(X_coef_cp[n-i260],True,1,False,1))

    def sum_arg_6(i258):
        # Child args for sum_arg_6
        return(i258*X_coef_cp[i258]*diff(Z_coef_cp[n-i258],True,1)+i258*diff(X_coef_cp[i258],True,1)*Z_coef_cp[n-i258])

    def sum_arg_5(i252):
        # Child args for sum_arg_5
        return((X_coef_cp[i252]*n-i252*X_coef_cp[i252])*diff(Z_coef_cp[n-i252],True,1)+(diff(X_coef_cp[i252],True,1)*n-i252*diff(X_coef_cp[i252],True,1))*Z_coef_cp[n-i252])

    def sum_arg_4(i209):
        # Child args for sum_arg_4
        def sum_arg_3(i210):
            # Child args for sum_arg_3
            return(B_theta_coef_cp[i210]*B_denom_coef_c[(-n)-i210+2*i209]*is_seq(n-i209,i209-i210))

        return(is_seq(0,n-i209)*B_alpha_coef[n-i209]*is_integer(n-i209)*py_sum(sum_arg_3,0,i209))

    def sum_arg_2(i201):
        # Child args for sum_arg_2
        def sum_arg_1(i202):
            # Child args for sum_arg_1
            return((diff(B_psi_coef_cp[i202],True,1)*B_denom_coef_c[(-n)-i202+2*i201+2]+B_psi_coef_cp[i202]*diff(B_denom_coef_c[(-n)-i202+2*i201+2],True,1))*is_seq(n-i201-2,i201-i202))

        return(is_seq(0,n-i201-2)*B_alpha_coef[n-i201-2]*is_integer(n-i201-2)*py_sum(sum_arg_1,0,i201))


    out = ((is_seq(0,n)*dl_p*n*is_integer(n)*py_sum_parallel(sum_arg_30,0,n)-is_seq(0,n)*dl_p*n*is_integer(n)*py_sum_parallel(sum_arg_29,0,n)+is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_28,0,n)-is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_27,0,n))*tau_p+is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_9,0,n)+is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_8,0,n)+is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_7,0,n)+is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_6,0,n)-is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_5,0,n)+n*py_sum_parallel(sum_arg_4,ceil(n/2),floor(n))+is_seq(0,n)*dl_p*kap_p*n*is_integer(n)*py_sum_parallel(sum_arg_26,0,n)-is_seq(0,n)*dl_p*kap_p*n*is_integer(n)*py_sum_parallel(sum_arg_25,0,n)-n*py_sum_parallel(sum_arg_24,ceil(n/2),floor(n))-is_seq(0,n)*n*is_integer(n)*py_sum_parallel(sum_arg_22,0,n)-n*py_sum_parallel(sum_arg_21,ceil(n/2),floor(n))-2*py_sum_parallel(sum_arg_2,ceil(n/2)-1,floor(n)-2)-is_seq(0,n)*n*is_integer(n)*py_sum_parallel(sum_arg_19,0,n)-n*py_sum_parallel(sum_arg_18,ceil(n/2),floor(n))-is_seq(0,n)*n*is_integer(n)*py_sum_parallel(sum_arg_16,0,n)+py_sum_parallel(sum_arg_15,ceil(n/2),floor(n))+py_sum_parallel(sum_arg_13,ceil(n/2),floor(n))+py_sum_parallel(sum_arg_11,ceil(n/2),floor(n)))/(2*B_alpha_coef[0]*B_denom_coef_c[0])
    return(out)

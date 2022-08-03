# This script calculates Xn. Values of:
# n,
# X_coef_cp[0, ... , n-1],
# Y_coef_cp[0, ... , n-1],
# Z_coef_cp[0, ... , n],
# B_denom_coef_c[0, ... , n],
# B_alpha_coef[0, ... , n-2],
# kap_p, dl_p, tau_p, iota[0, ... , (n-2)/2 or (n-3)/2]
# must be provided.
#
# X_coef_cp[n]=0 must be provided using ChPhiEpsFunc.zero_append().
from math import floor, ceil
from math_utilities import is_seq, py_sum, is_integer, diff
import chiphifunc
def eval_Xn_cp(n,
    X_coef_cp,
    Y_coef_cp,
    Z_coef_cp,
    B_denom_coef_c,
    B_alpha_coef,
    kap_p, dl_p, tau_p,
    iota_coef):
    def sum_arg_38(i214):
        # Child args for sum_arg_38
        return(X_coef_cp[i214]*X_coef_cp[n-i214])

    def sum_arg_37(i212):
        # Child args for sum_arg_37
        return(Y_coef_cp[i212]*Y_coef_cp[n-i212])

    def sum_arg_36(i224):
        # Child args for sum_arg_36
        return(Y_coef_cp[i224]*diff(X_coef_cp[n-i224],'phi',1))

    def sum_arg_35(i222):
        # Child args for sum_arg_35
        return(X_coef_cp[i222]*diff(Y_coef_cp[n-i222],'phi',1))

    def sum_arg_34(i219):
        # Child args for sum_arg_34
        def sum_arg_33(i220):
            # Child args for sum_arg_33
            return(Y_coef_cp[i220]*diff(X_coef_cp[(-n)-i220+2*i219],'chi',1)*is_seq(n-i219,i219-i220))

        return(is_seq(0,n-i219)*iota_coef[n-i219]*is_integer(n-i219)*py_sum(sum_arg_33,0,i219))

    def sum_arg_32(i217):
        # Child args for sum_arg_32
        def sum_arg_31(i218):
            # Child args for sum_arg_31
            return(X_coef_cp[i218]*diff(Y_coef_cp[(-n)-i218+2*i217],'chi',1)*is_seq(n-i217,i217-i218))

        return(is_seq(0,n-i217)*iota_coef[n-i217]*is_integer(n-i217)*py_sum(sum_arg_31,0,i217))

    def sum_arg_30(i216):
        # Child args for sum_arg_30
        return(Y_coef_cp[i216]*Z_coef_cp[n-i216])

    def sum_arg_29(i247):
        # Child args for sum_arg_29
        def sum_arg_28(i248):
            # Child args for sum_arg_28
            def sum_arg_27(i249):
                # Child args for sum_arg_27
                return((is_seq(0,n-i249-i247))\
                    *(diff(X_coef_cp[(-i249)-i248+i247],'chi',1))\
                    *(iota_coef[i249])\
                    *(diff(X_coef_cp[(-n)+i249+i248+i247],'chi',1))\
                    *(iota_coef[n-i249-i247])\
                    *(is_integer(n-i249-i247))\
                    *(is_seq(n-i249-i247,i248)))

            return(py_sum(sum_arg_27,0,i247-i248))

        return(py_sum(sum_arg_28,0,i247))

    def sum_arg_26(i241):
        # Child args for sum_arg_26
        def sum_arg_25(i242):
            # Child args for sum_arg_25
            def sum_arg_24(i243):
                # Child args for sum_arg_24
                return((is_seq(0,n-i243-i241))\
                    *(diff(Y_coef_cp[(-i243)-i242+i241],'chi',1))\
                    *(iota_coef[i243])\
                    *(diff(Y_coef_cp[(-n)+i243+i242+i241],'chi',1))\
                    *(iota_coef[n-i243-i241])\
                    *(is_integer(n-i243-i241))\
                    *(is_seq(n-i243-i241,i242)))

            return(py_sum(sum_arg_24,0,i241-i242))

        return(py_sum(sum_arg_25,0,i241))

    def sum_arg_23(i237):
        # Child args for sum_arg_23
        def sum_arg_22(i238):
            # Child args for sum_arg_22
            def sum_arg_21(i239):
                # Child args for sum_arg_21
                return((is_seq(0,n-i239-i237))\
                    *(diff(Z_coef_cp[(-i239)-i238+i237],'chi',1))\
                    *(iota_coef[i239])\
                    *(diff(Z_coef_cp[(-n)+i239+i238+i237],'chi',1))\
                    *(iota_coef[n-i239-i237])\
                    *(is_integer(n-i239-i237))\
                    *(is_seq(n-i239-i237,i238)))

            return(py_sum(sum_arg_21,0,i237-i238))

        return(py_sum(sum_arg_22,0,i237))

    def sum_arg_20(i395):
        # Child args for sum_arg_20
        def sum_arg_19(i396):
            # Child args for sum_arg_19
            return(diff(Z_coef_cp[i396],'phi',1)*diff(Z_coef_cp[(-n)-i396+2*i395],'chi',1)*is_seq(n-i395,i395-i396))

        return(is_seq(0,n-i395)*iota_coef[n-i395]*is_integer(n-i395)*py_sum(sum_arg_19,0,i395))

    def sum_arg_18(i258):
        # Child args for sum_arg_18
        return(diff(X_coef_cp[i258],'phi',1)*diff(X_coef_cp[n-i258],'phi',1))

    def sum_arg_17(i256):
        # Child args for sum_arg_17
        return(diff(Y_coef_cp[i256],'phi',1)*diff(Y_coef_cp[n-i256],'phi',1))

    def sum_arg_16(i254):
        # Child args for sum_arg_16
        return(diff(Z_coef_cp[i254],'phi',1)*diff(Z_coef_cp[n-i254],'phi',1))

    def sum_arg_15(i251):
        # Child args for sum_arg_15
        def sum_arg_14(i252):
            # Child args for sum_arg_14
            return(diff(X_coef_cp[i252],'phi',1)*diff(X_coef_cp[(-n)-i252+2*i251],'chi',1)*is_seq(n-i251,i251-i252))

        return(is_seq(0,n-i251)*iota_coef[n-i251]*is_integer(n-i251)*py_sum(sum_arg_14,0,i251))

    def sum_arg_13(i245):
        # Child args for sum_arg_13
        def sum_arg_12(i246):
            # Child args for sum_arg_12
            return(diff(Y_coef_cp[i246],'phi',1)*diff(Y_coef_cp[(-n)-i246+2*i245],'chi',1)*is_seq(n-i245,i245-i246))

        return(is_seq(0,n-i245)*iota_coef[n-i245]*is_integer(n-i245)*py_sum(sum_arg_12,0,i245))

    def sum_arg_11(i236):
        # Child args for sum_arg_11
        return(Z_coef_cp[i236]*diff(X_coef_cp[n-i236],'phi',1))

    def sum_arg_10(i234):
        # Child args for sum_arg_10
        return(X_coef_cp[i234]*diff(Z_coef_cp[n-i234],'phi',1))

    def sum_arg_9(i231):
        # Child args for sum_arg_9
        def sum_arg_8(i232):
            # Child args for sum_arg_8
            return(Z_coef_cp[i232]*diff(X_coef_cp[(-n)-i232+2*i231],'chi',1)*is_seq(n-i231,i231-i232))

        return(is_seq(0,n-i231)*iota_coef[n-i231]*is_integer(n-i231)*py_sum(sum_arg_8,0,i231))

    def sum_arg_7(i229):
        # Child args for sum_arg_7
        def sum_arg_6(i230):
            # Child args for sum_arg_6
            return(X_coef_cp[i230]*diff(Z_coef_cp[(-n)-i230+2*i229],'chi',1)*is_seq(n-i229,i229-i230))

        return(is_seq(0,n-i229)*iota_coef[n-i229]*is_integer(n-i229)*py_sum(sum_arg_6,0,i229))

    def sum_arg_5(i203):
        # Child args for sum_arg_5
        def sum_arg_4(i202):
            # Child args for sum_arg_4
            return(B_alpha_coef[i202]*B_alpha_coef[n-i203-i202])

        return(is_seq(0,2*i203-n)*B_denom_coef_c[2*i203-n]*is_integer(2*i203-n)*is_seq(2*i203-n,i203)*py_sum(sum_arg_4,0,n-i203))

    def sum_arg_3(i228):
        # Child args for sum_arg_3
        return(X_coef_cp[i228]*X_coef_cp[n-i228])

    def sum_arg_2(i226):
        # Child args for sum_arg_2
        return(Z_coef_cp[i226]*Z_coef_cp[n-i226])

    def sum_arg_1(i209):
        # Child args for sum_arg_1
        return(is_seq(0,n-i209)*diff(Z_coef_cp[2*i209-n],'chi',1)*iota_coef[n-i209]*is_integer(n-i209)*is_seq(n-i209,i209))


    out = (is_seq(0,n)*dl_p**2*is_integer(n)*py_sum(sum_arg_38,0,n)*tau_p**2+is_seq(0,n)*dl_p**2*is_integer(n)*py_sum(sum_arg_37,0,n)*tau_p**2+2*is_seq(0,n)*dl_p*is_integer(n)*py_sum(sum_arg_36,0,n)*tau_p-2*is_seq(0,n)*dl_p*is_integer(n)*py_sum(sum_arg_35,0,n)*tau_p+2*dl_p*py_sum(sum_arg_34,ceil(0.5*n),floor(n))*tau_p-2*dl_p*py_sum(sum_arg_32,ceil(0.5*n),floor(n))*tau_p+2*is_seq(0,n)*dl_p**2*kap_p*is_integer(n)*py_sum(sum_arg_30,0,n)*tau_p+2*dl_p*kap_p*py_sum(sum_arg_9,ceil(0.5*n),floor(n))-2*dl_p*kap_p*py_sum(sum_arg_7,ceil(0.5*n),floor(n))-py_sum(sum_arg_5,ceil(0.5*n),floor(n))+is_seq(0,n)*dl_p**2*kap_p**2*is_integer(n)*py_sum(sum_arg_3,0,n)+py_sum(sum_arg_29,ceil(0.5*n),floor(n))+py_sum(sum_arg_26,ceil(0.5*n),floor(n))+py_sum(sum_arg_23,ceil(0.5*n),floor(n))+2*py_sum(sum_arg_20,ceil(0.5*n),floor(n))+is_seq(0,n)*dl_p**2*kap_p**2*is_integer(n)*py_sum(sum_arg_2,0,n)+is_seq(0,n)*is_integer(n)*py_sum(sum_arg_18,0,n)+is_seq(0,n)*is_integer(n)*py_sum(sum_arg_17,0,n)+is_seq(0,n)*is_integer(n)*py_sum(sum_arg_16,0,n)+2*py_sum(sum_arg_15,ceil(0.5*n),floor(n))+2*py_sum(sum_arg_13,ceil(0.5*n),floor(n))+2*is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum(sum_arg_11,0,n)-2*is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum(sum_arg_10,0,n)+2*dl_p*py_sum(sum_arg_1,ceil(0.5*n),floor(n))+2*is_seq(0,n)*dl_p*is_integer(n)*diff(Z_coef_cp[n],'phi',1)-2*is_seq(0,n)*dl_p**2*kap_p*X_coef_cp[n]*is_integer(n))/(2*dl_p**2*kap_p)
    return(out)

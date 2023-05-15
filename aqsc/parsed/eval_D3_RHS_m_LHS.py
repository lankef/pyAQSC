# D3_RHS[n] - D3_LHS[n] for calculating Y[n,+1].
# The constant component is what we need.
# Must evaluate with Yn, Zn+1=0
from math import floor, ceil
from aqsc.math_utilities import *
def eval_D3_RHS_m_LHS(n, X_coef_cp, Y_coef_cp, Z_coef_cp, B_theta_coef_cp,
    B_denom_coef_c, B_alpha_coef, iota_coef, dl_p, tau_p, kap_p):
    def sum_arg_15(i268):
        # Child args for sum_arg_15
        return(X_coef_cp[i268]*diff(Y_coef_cp[n-i268],True,1))

    def sum_arg_14(i264):
        # Child args for sum_arg_14
        return(Y_coef_cp[i264]*diff(X_coef_cp[n-i264],True,1))

    def sum_arg_13(i270):
        # Child args for sum_arg_13
        return(X_coef_cp[i270]*diff(Z_coef_cp[n-i270],True,1))

    def sum_arg_12(i266):
        # Child args for sum_arg_12
        return(Z_coef_cp[i266]*diff(X_coef_cp[n-i266],True,1))

    def sum_arg_11(i261):
        # Child args for sum_arg_11
        def sum_arg_10(i262):
            # Child args for sum_arg_10
            return(diff(Z_coef_cp[i262],True,1)*diff(Z_coef_cp[(-n)-i262+2*i261],True,1)*is_seq(n-i261,i261-i262))

        return(is_seq(0,n-i261)*iota_coef[n-i261]*is_integer(n-i261)*py_sum(sum_arg_10,0,i261))

    def sum_arg_9(i260):
        # Child args for sum_arg_9
        return(diff(Z_coef_cp[i260],True,1)*diff(Z_coef_cp[n-i260],False,1))

    def sum_arg_8(i257):
        # Child args for sum_arg_8
        def sum_arg_7(i258):
            # Child args for sum_arg_7
            return(diff(Y_coef_cp[i258],True,1)*diff(Y_coef_cp[(-n)-i258+2*i257],True,1)*is_seq(n-i257,i257-i258))

        return(is_seq(0,n-i257)*iota_coef[n-i257]*is_integer(n-i257)*py_sum(sum_arg_7,0,i257))

    def sum_arg_6(i256):
        # Child args for sum_arg_6
        return(diff(Y_coef_cp[i256],True,1)*diff(Y_coef_cp[n-i256],False,1))

    def sum_arg_5(i253):
        # Child args for sum_arg_5
        def sum_arg_4(i254):
            # Child args for sum_arg_4
            return(diff(X_coef_cp[i254],True,1)*diff(X_coef_cp[(-n)-i254+2*i253],True,1)*is_seq(n-i253,i253-i254))

        return(is_seq(0,n-i253)*iota_coef[n-i253]*is_integer(n-i253)*py_sum(sum_arg_4,0,i253))

    def sum_arg_3(i252):
        # Child args for sum_arg_3
        return(diff(X_coef_cp[i252],True,1)*diff(X_coef_cp[n-i252],False,1))

    def sum_arg_2(i209):
        # Child args for sum_arg_2
        def sum_arg_1(i210):
            # Child args for sum_arg_1
            return(B_theta_coef_cp[i210]*B_denom_coef_c[(-n)-i210+2*i209]*is_seq(n-i209,i209-i210))

        return(is_seq(0,n-i209)*B_alpha_coef[n-i209]*is_integer(n-i209)*py_sum(sum_arg_1,0,i209))


    out = ((is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_14,0,n)-is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_15,0,n))*tau_p)\
        +(is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_9,0,n))\
        +(py_sum_parallel(sum_arg_8,ceil(n/2),floor(n)))\
        +(is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_6,0,n))\
        +(py_sum_parallel(sum_arg_5,ceil(n/2),floor(n)))\
        +(is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_3,0,n))\
        +(-py_sum_parallel(sum_arg_2,ceil(n/2),floor(n)))\
        +(-is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_13,0,n))\
        +(is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_12,0,n))\
        +(py_sum_parallel(sum_arg_11,ceil(n/2),floor(n)))\
        +(is_seq(0,n)*dl_p*is_integer(n)*diff(Z_coef_cp[n],True,1))
    return(out)

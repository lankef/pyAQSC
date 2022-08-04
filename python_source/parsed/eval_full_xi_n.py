# Xi_n, but without doing the 0th order matching. 
# The constant component is the actual sigma_n(phi).
# Must evaluate with Yn, Zn+1=0
from math import floor, ceil
from math_utilities import is_seq, py_sum, is_integer, diff
import chiphifunc
def eval_full_Xi_n_p(n, X_coef_cp, Y_coef_cp, Z_coef_cp, \
    kap_p, dl_p, tau_p, iota_coef):
    def sum_arg_13(i290):
        # Child args for sum_arg_13
        return(X_coef_cp[i290]*diff(Y_coef_cp[n-i290+1],'chi',1))

    def sum_arg_12(i286):
        # Child args for sum_arg_12
        return(Y_coef_cp[i286]*diff(X_coef_cp[n-i286+1],'chi',1))

    def sum_arg_11(i292):
        # Child args for sum_arg_11
        return(X_coef_cp[i292]*diff(Z_coef_cp[n-i292+1],'chi',1))

    def sum_arg_10(i288):
        # Child args for sum_arg_10
        return(Z_coef_cp[i288]*diff(X_coef_cp[n-i288+1],'chi',1))

    def sum_arg_9(i283):
        # Child args for sum_arg_9
        def sum_arg_8(i284):
            # Child args for sum_arg_8
            return(diff(Z_coef_cp[i284],'chi',1)*diff(Z_coef_cp[(-n)-i284+2*i283-1],'chi',1)*is_seq(n-i283+1,i283-i284))

        return(is_seq(0,n-i283+1)*iota_coef[n-i283+1]*is_integer(n-i283+1)*py_sum(sum_arg_8,0,i283))

    def sum_arg_7(i282):
        # Child args for sum_arg_7
        return(diff(Z_coef_cp[i282],'chi',1)*diff(Z_coef_cp[n-i282+1],'phi',1))

    def sum_arg_6(i279):
        # Child args for sum_arg_6
        def sum_arg_5(i280):
            # Child args for sum_arg_5
            return(diff(Y_coef_cp[i280],'chi',1)*diff(Y_coef_cp[(-n)-i280+2*i279-1],'chi',1)*is_seq(n-i279+1,i279-i280))

        return(is_seq(0,n-i279+1)*iota_coef[n-i279+1]*is_integer(n-i279+1)*py_sum(sum_arg_5,0,i279))

    def sum_arg_4(i278):
        # Child args for sum_arg_4
        return(diff(Y_coef_cp[i278],'chi',1)*diff(Y_coef_cp[n-i278+1],'phi',1))

    def sum_arg_3(i275):
        # Child args for sum_arg_3
        def sum_arg_2(i276):
            # Child args for sum_arg_2
            return(diff(X_coef_cp[i276],'chi',1)*diff(X_coef_cp[(-n)-i276+2*i275-1],'chi',1)*is_seq(n-i275+1,i275-i276))

        return(is_seq(0,n-i275+1)*iota_coef[n-i275+1]*is_integer(n-i275+1)*py_sum(sum_arg_2,0,i275))

    def sum_arg_1(i274):
        # Child args for sum_arg_1
        return(diff(X_coef_cp[i274],'chi',1)*diff(X_coef_cp[n-i274+1],'phi',1))


    out = (-is_seq(0,n+1)*dl_p*is_integer(n+1)*py_sum(sum_arg_13,0,n+1)*tau_p)\
        +(is_seq(0,n+1)*dl_p*is_integer(n+1)*py_sum(sum_arg_12,0,n+1)*tau_p)\
        +(py_sum(sum_arg_9,ceil(0.5*n+0.5),floor(n)+1))\
        +(is_seq(0,n+1)*is_integer(n+1)*py_sum(sum_arg_7,0,n+1))\
        +(py_sum(sum_arg_6,ceil(0.5*n+0.5),floor(n)+1))\
        +(is_seq(0,n+1)*is_integer(n+1)*py_sum(sum_arg_4,0,n+1))\
        +(py_sum(sum_arg_3,ceil(0.5*n+0.5),floor(n)+1))\
        +(-is_seq(0,n+1)*dl_p*kap_p*is_integer(n+1)*py_sum(sum_arg_11,0,n+1))\
        +(is_seq(0,n+1)*dl_p*kap_p*is_integer(n+1)*py_sum(sum_arg_10,0,n+1))\
        +(is_seq(0,n+1)*is_integer(n+1)*py_sum(sum_arg_1,0,n+1))\
        +(is_seq(0,n+1)*dl_p*is_integer(n+1)*diff(Z_coef_cp[n+1],'chi',1))
    return(out)

# RHS - LHS
# Used in (conv(a) + conv(b)@dchi)@Yn+1 = RHS - LHS(Yn+1 = 0)
# Must run with Yn+1=0.# Depends on Xn+1, Yn, Zn, B_theta n, B_psi n-2
# iota (n-2)/2 or (n-3)/2, B_alpha n/2 or (n-1)/2.
from math import floor, ceil
from math_utilities import *
import chiphifunc
from jax import jit
from functools import partial
@partial(jit, static_argnums=(0,))
def rhs_minus_lhs(n,
    X_coef_cp,
    Y_coef_cp,
    Z_coef_cp,
    B_psi_coef_cp,
    B_theta_coef_cp,
    B_alpha_coef,
    kap_p, dl_p, tau_p,
    iota_coef):
    def sum_arg_31(i1744):
        # Child args for sum_arg_31
        def sum_arg_30(i1742):
            # Child args for sum_arg_30
            return(B_theta_coef_cp[i1742]*X_coef_cp[n-i1744-i1742+2])

        return(i1744*X_coef_cp[i1744]*py_sum(sum_arg_30,0,n-i1744+2))

    def sum_arg_29(i1740):
        # Child args for sum_arg_29
        def sum_arg_28(i1738):
            # Child args for sum_arg_28
            return(B_theta_coef_cp[i1738]*Y_coef_cp[n-i1740-i1738+2])

        return(i1740*Y_coef_cp[i1740]*py_sum(sum_arg_28,0,n-i1740+2))

    def sum_arg_27(i1736):
        # Child args for sum_arg_27
        def sum_arg_26(i1734):
            # Child args for sum_arg_26
            return(B_psi_coef_cp[i1734]*X_coef_cp[n-i1736-i1734])

        return(diff(X_coef_cp[i1736],True,1)*py_sum(sum_arg_26,0,n-i1736))

    def sum_arg_25(i1732):
        # Child args for sum_arg_25
        def sum_arg_24(i1730):
            # Child args for sum_arg_24
            return(B_psi_coef_cp[i1730]*Y_coef_cp[n-i1732-i1730])

        return(diff(Y_coef_cp[i1732],True,1)*py_sum(sum_arg_24,0,n-i1732))

    def sum_arg_23(i1920):
        # Child args for sum_arg_23
        def sum_arg_22(i1762):
            # Child args for sum_arg_22
            return(B_psi_coef_cp[i1762]*diff(X_coef_cp[n-i1920-i1762],False,1))

        return(diff(Y_coef_cp[i1920],True,1)*py_sum(sum_arg_22,0,n-i1920))

    def sum_arg_21(i1918):
        # Child args for sum_arg_21
        def sum_arg_20(i1758):
            # Child args for sum_arg_20
            return(B_psi_coef_cp[i1758]*diff(X_coef_cp[n-i1918-i1758],True,1))

        return(diff(Y_coef_cp[i1918],False,1)*py_sum(sum_arg_20,0,n-i1918))

    def sum_arg_19(i1921):
        # Child args for sum_arg_19
        def sum_arg_18(i1922):
            # Child args for sum_arg_18
            return((((-i1922)+2*i1921-2)*diff(Y_coef_cp[i1922],True,1)*X_coef_cp[(-n)-i1922+2*i1921-2]-diff(Y_coef_cp[i1922],True,1)*X_coef_cp[(-n)-i1922+2*i1921-2]*n)*is_seq(n-i1921+2,i1921-i1922))

        return(is_seq(0,n-i1921+2)*B_alpha_coef[n-i1921+2]*is_integer(n-i1921+2)*py_sum(sum_arg_18,0,i1921))

    def sum_arg_17(i1914):
        # Child args for sum_arg_17
        def sum_arg_16(i1760):
            # Child args for sum_arg_16
            return((B_theta_coef_cp[i1760]*n-B_theta_coef_cp[i1760]*i1914+(2-i1760)*B_theta_coef_cp[i1760])*X_coef_cp[n-i1914-i1760+2])

        return(diff(Y_coef_cp[i1914],False,1)*py_sum(sum_arg_16,0,n-i1914+2))

    def sum_arg_15(i1771):
        # Child args for sum_arg_15
        def sum_arg_14(i1772):
            # Child args for sum_arg_14
            return((((-i1772)+2*i1771-2)*diff(X_coef_cp[i1772],True,1)*Y_coef_cp[(-n)-i1772+2*i1771-2]-diff(X_coef_cp[i1772],True,1)*Y_coef_cp[(-n)-i1772+2*i1771-2]*n)*is_seq(n-i1771+2,i1771-i1772))

        return(is_seq(0,n-i1771+2)*B_alpha_coef[n-i1771+2]*is_integer(n-i1771+2)*py_sum(sum_arg_14,0,i1771))

    def sum_arg_13(i1768):
        # Child args for sum_arg_13
        def sum_arg_12(i1766):
            # Child args for sum_arg_12
            return((B_theta_coef_cp[i1766]*n-B_theta_coef_cp[i1766]*i1768+(2-i1766)*B_theta_coef_cp[i1766])*Y_coef_cp[n-i1768-i1766+2])

        return(diff(X_coef_cp[i1768],False,1)*py_sum(sum_arg_12,0,n-i1768+2))

    def sum_arg_11(i1752):
        # Child args for sum_arg_11
        def sum_arg_10(i1750):
            # Child args for sum_arg_10
            return((B_theta_coef_cp[i1750]*n-B_theta_coef_cp[i1750]*i1752+(2-i1750)*B_theta_coef_cp[i1750])*Y_coef_cp[n-i1752-i1750+2])

        return(Z_coef_cp[i1752]*py_sum(sum_arg_10,0,n-i1752+2))

    def sum_arg_9(i1748):
        # Child args for sum_arg_9
        def sum_arg_8(i1746):
            # Child args for sum_arg_8
            return(B_psi_coef_cp[i1746]*Z_coef_cp[n-i1748-i1746])

        return(diff(Y_coef_cp[i1748],True,1)*py_sum(sum_arg_8,0,n-i1748))

    def sum_arg_7(i1923):
        # Child args for sum_arg_7
        def sum_arg_6(i1924):
            # Child args for sum_arg_6
            def sum_arg_5(i2380):
                # Child args for sum_arg_5
                return(diff(X_coef_cp[(-i2380)-i1924+i1923],True,1)*i2380*Y_coef_cp[i2380])

            return(is_seq(0,(-n)+i1924+i1923-2)*B_theta_coef_cp[(-n)+i1924+i1923-2]*is_integer((-n)+i1924+i1923-2)*is_seq((-n)+i1924+i1923-2,i1924)*py_sum(sum_arg_5,0,i1923-i1924))

        return(iota_coef[n-i1923+2]*py_sum(sum_arg_6,0,i1923))

    def sum_arg_4(i1915):
        # Child args for sum_arg_4
        def sum_arg_3(i1916):
            # Child args for sum_arg_3
            def sum_arg_2(i2364):
                # Child args for sum_arg_2
                return(diff(Y_coef_cp[(-i2364)-i1916+i1915],True,1)*i2364*X_coef_cp[i2364])

            return(is_seq(0,(-n)+i1916+i1915-2)*B_theta_coef_cp[(-n)+i1916+i1915-2]*is_integer((-n)+i1916+i1915-2)*is_seq((-n)+i1916+i1915-2,i1916)*py_sum(sum_arg_2,0,i1915-i1916))

        return(iota_coef[n-i1915+2]*py_sum(sum_arg_3,0,i1915))

    def sum_arg_1(i1773):
        # Child args for sum_arg_1
        return(is_seq(0,n-i1773)*diff(Z_coef_cp[2*i1773-n],True,1)*iota_coef[n-i1773]*is_integer(n-i1773)*is_seq(n-i1773,i1773))


    out = -((is_seq(0,n+2)*dl_p*is_integer(n+2)*py_sum(sum_arg_31,0,n+2)+is_seq(0,n+2)*dl_p*is_integer(n+2)*py_sum(sum_arg_29,0,n+2)-2*is_seq(0,n)*dl_p*is_integer(n)*py_sum(sum_arg_27,0,n)-2*is_seq(0,n)*dl_p*is_integer(n)*py_sum(sum_arg_25,0,n))*tau_p-2*is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum(sum_arg_9,0,n)+py_sum(sum_arg_7,ceil(n/2)+1,floor(n)+2)-py_sum(sum_arg_4,ceil(n/2)+1,floor(n)+2)-2*is_seq(0,n)*is_integer(n)*py_sum(sum_arg_23,0,n)+2*is_seq(0,n)*is_integer(n)*py_sum(sum_arg_21,0,n)+py_sum(sum_arg_19,ceil(n/2)+1,floor(n)+2)-is_seq(0,n+2)*is_integer(n+2)*py_sum(sum_arg_17,0,n+2)-py_sum(sum_arg_15,ceil(n/2)+1,floor(n)+2)+is_seq(0,n+2)*is_integer(n+2)*py_sum(sum_arg_13,0,n+2)+is_seq(0,n+2)*dl_p*kap_p*is_integer(n+2)*py_sum(sum_arg_11,0,n+2)-2*py_sum(sum_arg_1,ceil(n/2),floor(n))-2*is_seq(0,n)*is_integer(n)*diff(Z_coef_cp[n],False,1)+2*is_seq(0,n)*dl_p*kap_p*X_coef_cp[n]*is_integer(n))/2
    return(out)


# Coefficient for Yn+1 - only X1 and Balpha0 needed
@partial(jit, static_argnums=(0,))
def coef_a(n, B_alpha_coef, X_coef_cp):

    out = (B_alpha_coef[0]*diff(X_coef_cp[1],True,1)*((-n)-1))/2
    return(out)


# Coefficient for dchi Yn+1  - only X1 and Balpha0 needed
def coef_b(B_alpha_coef, X_coef_cp):

    out = (B_alpha_coef[0]*X_coef_cp[1])/2
    return(out)

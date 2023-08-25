# Evaluating the m=0 component of II, which serevs as the m=0 component
# of the looped equation at odd orders.
from math import floor, ceil
from aqsc.math_utilities import *
# from jax import jit
# from functools import partial
# @partial(jit, static_argnums=(0,))
def eval_II_center(n,
    B_theta_coef_cp, B_alpha_coef, B_denom_coef_c,
    p_perp_coef_cp, Delta_coef_cp, iota_coef):
    def sum_arg_23(i264):
        # Child args for sum_arg_23
        def sum_arg_22(i228):
            # Child args for sum_arg_22
            def sum_arg_21(i226):
                # Child args for sum_arg_21
                return(B_denom_coef_c[i226]*B_denom_coef_c[n-i264-i228-i226])

            return(diff(p_perp_coef_cp[i228],False,1)*py_sum(sum_arg_21,0,n-i264-i228))

        return(B_theta_coef_cp[i264]*py_sum(sum_arg_22,0,n-i264))

    def sum_arg_20(i261):
        # Child args for sum_arg_20
        def sum_arg_19(i262):
            # Child args for sum_arg_19
            def sum_arg_18(i260):
                # Child args for sum_arg_18
                return(Delta_coef_cp[i260]*diff(B_theta_coef_cp[(-n)-i262+2*i261-i260],True,1)*is_seq(n-i261,(-i262)+i261-i260))

            return(B_denom_coef_c[i262]*py_sum(sum_arg_18,0,i261-i262))

        return(is_seq(0,n-i261)*iota_coef[n-i261]*is_integer(n-i261)*py_sum(sum_arg_19,0,i261))

    def sum_arg_17(i258):
        # Child args for sum_arg_17
        def sum_arg_16(i256):
            # Child args for sum_arg_16
            return(Delta_coef_cp[i256]*diff(B_theta_coef_cp[n-i258-i256],False,1))

        return(B_denom_coef_c[i258]*py_sum(sum_arg_16,0,n-i258))

    def sum_arg_15(i253):
        # Child args for sum_arg_15
        def sum_arg_14(i254):
            # Child args for sum_arg_14
            return(B_denom_coef_c[i254]*diff(B_theta_coef_cp[(-n)-i254+2*i253],True,1)*is_seq(n-i253,i253-i254))

        return(is_seq(0,n-i253)*iota_coef[n-i253]*is_integer(n-i253)*py_sum(sum_arg_14,0,i253))

    def sum_arg_13(i252):
        # Child args for sum_arg_13
        return(B_denom_coef_c[i252]*diff(B_theta_coef_cp[n-i252],False,1))

    def sum_arg_12(i271):
        # Child args for sum_arg_12
        def sum_arg_11(i272):
            # Child args for sum_arg_11
            def sum_arg_10(i720):
                # Child args for sum_arg_10
                return(diff(B_denom_coef_c[(-i720)-i272+i271],True,1)*Delta_coef_cp[i720])

            return(is_seq(0,(-n)+i272+i271)*B_theta_coef_cp[(-n)+i272+i271]*is_integer((-n)+i272+i271)*is_seq((-n)+i272+i271,i272)*py_sum(sum_arg_10,0,i271-i272))

        return(iota_coef[n-i271]*py_sum(sum_arg_11,0,i271))

    def sum_arg_9(i269):
        # Child args for sum_arg_9
        def sum_arg_8(i238):
            # Child args for sum_arg_8
            return(Delta_coef_cp[i238]*diff(B_denom_coef_c[(-n)+2*i269-i238],True,1))

        return(is_seq(0,n-i269)*B_alpha_coef[n-i269]*is_integer(n-i269)*is_seq(n-i269,i269)*py_sum(sum_arg_8,0,2*i269-n))

    def sum_arg_7(i267):
        # Child args for sum_arg_7
        def sum_arg_6(i268):
            # Child args for sum_arg_6
            def sum_arg_5(i681):
                # Child args for sum_arg_5
                def sum_arg_4(i704):
                    # Child args for sum_arg_4
                    return(B_denom_coef_c[(-i704)-i681-i268+i267]*B_denom_coef_c[i704])

                return(diff(p_perp_coef_cp[i681],True,1)*py_sum(sum_arg_4,0,(-i681)-i268+i267))

            return(is_seq(0,(-n)+i268+i267)*B_theta_coef_cp[(-n)+i268+i267]*is_integer((-n)+i268+i267)*is_seq((-n)+i268+i267,i268)*py_sum(sum_arg_5,0,i267-i268))

        return(iota_coef[n-i267]*py_sum(sum_arg_6,0,i267))

    def sum_arg_3(i265):
        # Child args for sum_arg_3
        def sum_arg_2(i234):
            # Child args for sum_arg_2
            def sum_arg_1(i230):
                # Child args for sum_arg_1
                return(B_denom_coef_c[i230]*B_denom_coef_c[(-n)+2*i265-i234-i230])

            return(diff(p_perp_coef_cp[i234],True,1)*py_sum(sum_arg_1,0,(-n)+2*i265-i234))

        return(is_seq(0,n-i265)*B_alpha_coef[n-i265]*is_integer(n-i265)*is_seq(n-i265,i265)*py_sum(sum_arg_2,0,2*i265-n))


    out = -(py_sum_parallel(sum_arg_9,ceil(n/2),floor(n))+2*py_sum_parallel(sum_arg_7,ceil(n/2),floor(n))-2*py_sum_parallel(sum_arg_3,ceil(n/2),floor(n))+2*is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_23,0,n)-2*py_sum_parallel(sum_arg_20,ceil(n/2),floor(n))-2*is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_17,0,n)+2*py_sum_parallel(sum_arg_15,ceil(n/2),floor(n))+2*is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_13,0,n)-py_sum_parallel(sum_arg_12,ceil(n/2),floor(n)))/2
    return(out)

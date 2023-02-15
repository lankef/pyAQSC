# Evaluating the m=0 component of II, which serevs as the m=0 component
# of the looped equation at odd orders.
from math import floor, ceil
from math_utilities import *
import chiphifunc
def eval_II_center(n, \
    B_theta_coef_cp, B_alpha_coef, B_denom_coef_c, \
    p_perp_coef_cp, Delta_coef_cp, iota_coef):
    def sum_arg_23(i530):
        # Child args for sum_arg_23
        def sum_arg_22(i492):
            # Child args for sum_arg_22
            def sum_arg_21(i490):
                # Child args for sum_arg_21
                return(B_denom_coef_c[i490]*B_denom_coef_c[n-i530-i492-i490])

            return(diff(p_perp_coef_cp[i492],'phi',1)*py_sum(sum_arg_21,0,n-i530-i492))

        return(B_theta_coef_cp[i530]*py_sum(sum_arg_22,0,n-i530))

    def sum_arg_20(i527):
        # Child args for sum_arg_20
        def sum_arg_19(i528):
            # Child args for sum_arg_19
            def sum_arg_18(i526):
                # Child args for sum_arg_18
                return(Delta_coef_cp[i526]*diff(B_theta_coef_cp[(-n)-i528+2*i527-i526],'chi',1)*is_seq(n-i527,(-i528)+i527-i526))

            return(B_denom_coef_c[i528]*py_sum(sum_arg_18,0,i527-i528))

        return(is_seq(0,n-i527)*iota_coef[n-i527]*is_integer(n-i527)*py_sum(sum_arg_19,0,i527))

    def sum_arg_17(i524):
        # Child args for sum_arg_17
        def sum_arg_16(i522):
            # Child args for sum_arg_16
            return(Delta_coef_cp[i522]*diff(B_theta_coef_cp[n-i524-i522],'phi',1))

        return(B_denom_coef_c[i524]*py_sum(sum_arg_16,0,n-i524))

    def sum_arg_15(i519):
        # Child args for sum_arg_15
        def sum_arg_14(i520):
            # Child args for sum_arg_14
            return(B_denom_coef_c[i520]*diff(B_theta_coef_cp[(-n)-i520+2*i519],'chi',1)*is_seq(n-i519,i519-i520))

        return(is_seq(0,n-i519)*iota_coef[n-i519]*is_integer(n-i519)*py_sum(sum_arg_14,0,i519))

    def sum_arg_13(i518):
        # Child args for sum_arg_13
        return(B_denom_coef_c[i518]*diff(B_theta_coef_cp[n-i518],'phi',1))

    def sum_arg_12(i537):
        # Child args for sum_arg_12
        def sum_arg_11(i538):
            # Child args for sum_arg_11
            def sum_arg_10(i986):
                # Child args for sum_arg_10
                return(diff(B_denom_coef_c[(-i986)-i538+i537],'chi',1)*Delta_coef_cp[i986])

            return(is_seq(0,(-n)+i538+i537)*B_theta_coef_cp[(-n)+i538+i537]*is_integer((-n)+i538+i537)*is_seq((-n)+i538+i537,i538)*py_sum(sum_arg_10,0,i537-i538))

        return(iota_coef[n-i537]*py_sum(sum_arg_11,0,i537))

    def sum_arg_9(i535):
        # Child args for sum_arg_9
        def sum_arg_8(i502):
            # Child args for sum_arg_8
            return(Delta_coef_cp[i502]*diff(B_denom_coef_c[(-n)+2*i535-i502],'chi',1))

        return(is_seq(0,n-i535)*B_alpha_coef[n-i535]*is_integer(n-i535)*is_seq(n-i535,i535)*py_sum(sum_arg_8,0,2*i535-n))

    def sum_arg_7(i533):
        # Child args for sum_arg_7
        def sum_arg_6(i534):
            # Child args for sum_arg_6
            def sum_arg_5(i947):
                # Child args for sum_arg_5
                def sum_arg_4(i970):
                    # Child args for sum_arg_4
                    return(B_denom_coef_c[(-i970)-i947-i534+i533]*B_denom_coef_c[i970])

                return(diff(p_perp_coef_cp[i947],'chi',1)*py_sum(sum_arg_4,0,(-i947)-i534+i533))

            return(is_seq(0,(-n)+i534+i533)*B_theta_coef_cp[(-n)+i534+i533]*is_integer((-n)+i534+i533)*is_seq((-n)+i534+i533,i534)*py_sum(sum_arg_5,0,i533-i534))

        return(iota_coef[n-i533]*py_sum(sum_arg_6,0,i533))

    def sum_arg_3(i531):
        # Child args for sum_arg_3
        def sum_arg_2(i498):
            # Child args for sum_arg_2
            def sum_arg_1(i494):
                # Child args for sum_arg_1
                return(B_denom_coef_c[i494]*B_denom_coef_c[(-n)+2*i531-i498-i494])

            return(diff(p_perp_coef_cp[i498],'chi',1)*py_sum(sum_arg_1,0,(-n)+2*i531-i498))

        return(is_seq(0,n-i531)*B_alpha_coef[n-i531]*is_integer(n-i531)*is_seq(n-i531,i531)*py_sum(sum_arg_2,0,2*i531-n))


    out = -(py_sum_parallel(sum_arg_9,ceil(n/2),floor(n))+2*py_sum_parallel(sum_arg_7,ceil(n/2),floor(n))-2*py_sum_parallel(sum_arg_3,ceil(n/2),floor(n))+2*is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_23,0,n)-2*py_sum_parallel(sum_arg_20,ceil(n/2),floor(n))-2*is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_17,0,n)+2*py_sum_parallel(sum_arg_15,ceil(n/2),floor(n))+2*is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_13,0,n)-py_sum_parallel(sum_arg_12,ceil(n/2),floor(n)))/2
    return(out)

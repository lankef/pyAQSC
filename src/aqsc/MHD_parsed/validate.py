# This script evaluates all governing equations to order n.
from math import floor, ceil
from aqsc.math_utilities import *
# from jax import jit
# from functools import partial
# @partial(jit, static_argnums=(0,))
def validate_J(n, X_coef_cp, Y_coef_cp, Z_coef_cp,
    B_denom_coef_c, B_alpha_coef,
    kap_p, dl_p, tau_p, iota_coef):
    def sum_arg_38(i20):
        # Child args for sum_arg_38
        return(X_coef_cp[i20]*X_coef_cp[n-i20])

    def sum_arg_37(i18):
        # Child args for sum_arg_37
        return(Y_coef_cp[i18]*Y_coef_cp[n-i18])

    def sum_arg_36(i30):
        # Child args for sum_arg_36
        return(Y_coef_cp[i30]*diff(X_coef_cp[n-i30],False,1))

    def sum_arg_35(i28):
        # Child args for sum_arg_35
        return(X_coef_cp[i28]*diff(Y_coef_cp[n-i28],False,1))

    def sum_arg_34(i25):
        # Child args for sum_arg_34
        def sum_arg_33(i26):
            # Child args for sum_arg_33
            return(Y_coef_cp[i26]*diff(X_coef_cp[(-n)-i26+2*i25],True,1)*is_seq(n-i25,i25-i26))

        return(is_seq(0,n-i25)*iota_coef[n-i25]*is_integer(n-i25)*py_sum(sum_arg_33,0,i25))

    def sum_arg_32(i23):
        # Child args for sum_arg_32
        def sum_arg_31(i24):
            # Child args for sum_arg_31
            return(X_coef_cp[i24]*diff(Y_coef_cp[(-n)-i24+2*i23],True,1)*is_seq(n-i23,i23-i24))

        return(is_seq(0,n-i23)*iota_coef[n-i23]*is_integer(n-i23)*py_sum(sum_arg_31,0,i23))

    def sum_arg_30(i22):
        # Child args for sum_arg_30
        return(Y_coef_cp[i22]*Z_coef_cp[n-i22])

    def sum_arg_29(i53):
        # Child args for sum_arg_29
        def sum_arg_28(i54):
            # Child args for sum_arg_28
            def sum_arg_27(i55):
                # Child args for sum_arg_27
                return((is_seq(0,n-i55-i53))\
                    *(diff(X_coef_cp[(-i55)-i54+i53],True,1))\
                    *(iota_coef[i55])\
                    *(diff(X_coef_cp[(-n)+i55+i54+i53],True,1))\
                    *(iota_coef[n-i55-i53])\
                    *(is_integer(n-i55-i53))\
                    *(is_seq(n-i55-i53,i54)))

            return(py_sum(sum_arg_27,0,i53-i54))

        return(py_sum(sum_arg_28,0,i53))

    def sum_arg_26(i47):
        # Child args for sum_arg_26
        def sum_arg_25(i48):
            # Child args for sum_arg_25
            def sum_arg_24(i49):
                # Child args for sum_arg_24
                return((is_seq(0,n-i49-i47))\
                    *(diff(Y_coef_cp[(-i49)-i48+i47],True,1))\
                    *(iota_coef[i49])\
                    *(diff(Y_coef_cp[(-n)+i49+i48+i47],True,1))\
                    *(iota_coef[n-i49-i47])\
                    *(is_integer(n-i49-i47))\
                    *(is_seq(n-i49-i47,i48)))

            return(py_sum(sum_arg_24,0,i47-i48))

        return(py_sum(sum_arg_25,0,i47))

    def sum_arg_23(i43):
        # Child args for sum_arg_23
        def sum_arg_22(i44):
            # Child args for sum_arg_22
            def sum_arg_21(i45):
                # Child args for sum_arg_21
                return((is_seq(0,n-i45-i43))\
                    *(diff(Z_coef_cp[(-i45)-i44+i43],True,1))\
                    *(iota_coef[i45])\
                    *(diff(Z_coef_cp[(-n)+i45+i44+i43],True,1))\
                    *(iota_coef[n-i45-i43])\
                    *(is_integer(n-i45-i43))\
                    *(is_seq(n-i45-i43,i44)))

            return(py_sum(sum_arg_21,0,i43-i44))

        return(py_sum(sum_arg_22,0,i43))

    def sum_arg_20(i9):
        # Child args for sum_arg_20
        def sum_arg_19(i8):
            # Child args for sum_arg_19
            return(B_alpha_coef[i8]*B_alpha_coef[n-i9-i8])

        return(is_seq(0,2*i9-n)*B_denom_coef_c[2*i9-n]*is_integer(2*i9-n)*is_seq(2*i9-n,i9)*py_sum(sum_arg_19,0,n-i9))

    def sum_arg_18(i64):
        # Child args for sum_arg_18
        return(diff(X_coef_cp[i64],False,1)*diff(X_coef_cp[n-i64],False,1))

    def sum_arg_17(i62):
        # Child args for sum_arg_17
        return(diff(Y_coef_cp[i62],False,1)*diff(Y_coef_cp[n-i62],False,1))

    def sum_arg_16(i60):
        # Child args for sum_arg_16
        return(diff(Z_coef_cp[i60],False,1)*diff(Z_coef_cp[n-i60],False,1))

    def sum_arg_15(i57):
        # Child args for sum_arg_15
        def sum_arg_14(i58):
            # Child args for sum_arg_14
            return(diff(X_coef_cp[i58],False,1)*diff(X_coef_cp[(-n)-i58+2*i57],True,1)*is_seq(n-i57,i57-i58))

        return(is_seq(0,n-i57)*iota_coef[n-i57]*is_integer(n-i57)*py_sum(sum_arg_14,0,i57))

    def sum_arg_13(i51):
        # Child args for sum_arg_13
        def sum_arg_12(i52):
            # Child args for sum_arg_12
            return(diff(Y_coef_cp[i52],False,1)*diff(Y_coef_cp[(-n)-i52+2*i51],True,1)*is_seq(n-i51,i51-i52))

        return(is_seq(0,n-i51)*iota_coef[n-i51]*is_integer(n-i51)*py_sum(sum_arg_12,0,i51))

    def sum_arg_11(i42):
        # Child args for sum_arg_11
        return(Z_coef_cp[i42]*diff(X_coef_cp[n-i42],False,1))

    def sum_arg_10(i40):
        # Child args for sum_arg_10
        return(X_coef_cp[i40]*diff(Z_coef_cp[n-i40],False,1))

    def sum_arg_9(i37):
        # Child args for sum_arg_9
        def sum_arg_8(i38):
            # Child args for sum_arg_8
            return(Z_coef_cp[i38]*diff(X_coef_cp[(-n)-i38+2*i37],True,1)*is_seq(n-i37,i37-i38))

        return(is_seq(0,n-i37)*iota_coef[n-i37]*is_integer(n-i37)*py_sum(sum_arg_8,0,i37))

    def sum_arg_7(i35):
        # Child args for sum_arg_7
        def sum_arg_6(i36):
            # Child args for sum_arg_6
            return(X_coef_cp[i36]*diff(Z_coef_cp[(-n)-i36+2*i35],True,1)*is_seq(n-i35,i35-i36))

        return(is_seq(0,n-i35)*iota_coef[n-i35]*is_integer(n-i35)*py_sum(sum_arg_6,0,i35))

    def sum_arg_5(i251):
        # Child args for sum_arg_5
        def sum_arg_4(i252):
            # Child args for sum_arg_4
            return(diff(Z_coef_cp[i252],False,1)*diff(Z_coef_cp[(-n)-i252+2*i251],True,1)*is_seq(n-i251,i251-i252))

        return(is_seq(0,n-i251)*iota_coef[n-i251]*is_integer(n-i251)*py_sum(sum_arg_4,0,i251))

    def sum_arg_3(i34):
        # Child args for sum_arg_3
        return(X_coef_cp[i34]*X_coef_cp[n-i34])

    def sum_arg_2(i32):
        # Child args for sum_arg_2
        return(Z_coef_cp[i32]*Z_coef_cp[n-i32])

    def sum_arg_1(i15):
        # Child args for sum_arg_1
        return(is_seq(0,n-i15)*diff(Z_coef_cp[2*i15-n],True,1)*iota_coef[n-i15]*is_integer(n-i15)*is_seq(n-i15,i15))


    out = (((-is_seq(0,n)*dl_p**2*is_integer(n)*py_sum_parallel(sum_arg_38,0,n))-is_seq(0,n)*dl_p**2*is_integer(n)*py_sum_parallel(sum_arg_37,0,n))*tau_p**2)\
        +(((-2*is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_36,0,n))+2*is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_35,0,n)-2*dl_p*py_sum_parallel(sum_arg_34,ceil(n/2),floor(n))+2*dl_p*py_sum_parallel(sum_arg_32,ceil(n/2),floor(n))-2*is_seq(0,n)*dl_p**2*kap_p*is_integer(n)*py_sum_parallel(sum_arg_30,0,n))*tau_p)\
        +(-2*dl_p*kap_p*py_sum_parallel(sum_arg_9,ceil(n/2),floor(n)))\
        +(2*dl_p*kap_p*py_sum_parallel(sum_arg_7,ceil(n/2),floor(n)))\
        +(-2*py_sum_parallel(sum_arg_5,ceil(n/2),floor(n)))\
        +(-is_seq(0,n)*dl_p**2*kap_p**2*is_integer(n)*py_sum_parallel(sum_arg_3,0,n))\
        +(-py_sum_parallel(sum_arg_29,ceil(n/2),floor(n)))\
        +(-py_sum_parallel(sum_arg_26,ceil(n/2),floor(n)))\
        +(-py_sum_parallel(sum_arg_23,ceil(n/2),floor(n)))\
        +(py_sum_parallel(sum_arg_20,ceil(n/2),floor(n)))\
        +(-is_seq(0,n)*dl_p**2*kap_p**2*is_integer(n)*py_sum_parallel(sum_arg_2,0,n))\
        +(-is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_18,0,n))\
        +(-is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_17,0,n))\
        +(-is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_16,0,n))\
        +(-2*py_sum_parallel(sum_arg_15,ceil(n/2),floor(n)))\
        +(-2*py_sum_parallel(sum_arg_13,ceil(n/2),floor(n)))\
        +(-2*is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_11,0,n))\
        +(2*is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_10,0,n))\
        +(-2*dl_p*py_sum_parallel(sum_arg_1,ceil(n/2),floor(n)))\
        +(-2*is_seq(0,n)*dl_p*is_integer(n)*diff(Z_coef_cp[n],False,1))\
        +(2*is_seq(0,n)*dl_p**2*kap_p*X_coef_cp[n]*is_integer(n))
    return(out)
# @partial(jit, static_argnums=(0,))
def validate_Cb(n, X_coef_cp, Y_coef_cp, Z_coef_cp,
    B_denom_coef_c, B_alpha_coef,
    B_psi_coef_cp, B_theta_coef_cp,
    kap_p, dl_p, tau_p, iota_coef):
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

        return(diff(X_coef_cp[i74],True,1)*py_sum(sum_arg_26,0,n-i74))

    def sum_arg_25(i70):
        # Child args for sum_arg_25
        def sum_arg_24(i68):
            # Child args for sum_arg_24
            return(B_psi_coef_cp[i68]*Y_coef_cp[n-i70-i68])

        return(diff(Y_coef_cp[i70],True,1)*py_sum(sum_arg_24,0,n-i70))

    def sum_arg_23(i258):
        # Child args for sum_arg_23
        def sum_arg_22(i96):
            # Child args for sum_arg_22
            return(B_psi_coef_cp[i96]*diff(X_coef_cp[n-i96-i258],True,1))

        return(diff(Y_coef_cp[i258],False,1)*py_sum(sum_arg_22,0,n-i258))

    def sum_arg_21(i262):
        # Child args for sum_arg_21
        def sum_arg_20(i100):
            # Child args for sum_arg_20
            return(B_psi_coef_cp[i100]*diff(X_coef_cp[n-i262-i100],False,1))

        return(diff(Y_coef_cp[i262],True,1)*py_sum(sum_arg_20,0,n-i262))

    def sum_arg_19(i254):
        # Child args for sum_arg_19
        def sum_arg_18(i98):
            # Child args for sum_arg_18
            return((B_theta_coef_cp[i98]*n+((-i98)-i254+2)*B_theta_coef_cp[i98])*X_coef_cp[n-i98-i254+2])

        return(diff(Y_coef_cp[i254],False,1)*py_sum(sum_arg_18,0,n-i254+2))

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

        return(diff(Y_coef_cp[i86],True,1)*py_sum(sum_arg_14,0,n-i86))

    def sum_arg_13(i259):
        # Child args for sum_arg_13
        def sum_arg_12(i260):
            # Child args for sum_arg_12
            return((((-i260)+2*i259-2)*diff(Y_coef_cp[i260],True,1)*X_coef_cp[(-n)-i260+2*i259-2]-diff(Y_coef_cp[i260],True,1)*X_coef_cp[(-n)-i260+2*i259-2]*n)*is_seq(n-i259+2,i259-i260))

        return(is_seq(0,n-i259+2)*B_alpha_coef[n-i259+2]*is_integer(n-i259+2)*py_sum(sum_arg_12,0,i259))

    def sum_arg_11(i109):
        # Child args for sum_arg_11
        def sum_arg_10(i110):
            # Child args for sum_arg_10
            return((((-i110)+2*i109-2)*diff(X_coef_cp[i110],True,1)*Y_coef_cp[(-n)-i110+2*i109-2]-diff(X_coef_cp[i110],True,1)*Y_coef_cp[(-n)-i110+2*i109-2]*n)*is_seq(n-i109+2,i109-i110))

        return(is_seq(0,n-i109+2)*B_alpha_coef[n-i109+2]*is_integer(n-i109+2)*py_sum(sum_arg_10,0,i109))

    def sum_arg_9(i106):
        # Child args for sum_arg_9
        def sum_arg_8(i104):
            # Child args for sum_arg_8
            return((B_theta_coef_cp[i104]*n-B_theta_coef_cp[i104]*i106+(2-i104)*B_theta_coef_cp[i104])*Y_coef_cp[n-i106-i104+2])

        return(diff(X_coef_cp[i106],False,1)*py_sum(sum_arg_8,0,n-i106+2))

    def sum_arg_7(i263):
        # Child args for sum_arg_7
        def sum_arg_6(i264):
            # Child args for sum_arg_6
            def sum_arg_5(i720):
                # Child args for sum_arg_5
                return(diff(X_coef_cp[(-i720)-i264+i263],True,1)*i720*Y_coef_cp[i720])

            return(is_seq(0,(-n)+i264+i263-2)*B_theta_coef_cp[(-n)+i264+i263-2]*is_integer((-n)+i264+i263-2)*is_seq((-n)+i264+i263-2,i264)*py_sum(sum_arg_5,0,i263-i264))

        return(iota_coef[n-i263+2]*py_sum(sum_arg_6,0,i263))

    def sum_arg_4(i255):
        # Child args for sum_arg_4
        def sum_arg_3(i256):
            # Child args for sum_arg_3
            def sum_arg_2(i704):
                # Child args for sum_arg_2
                return(diff(Y_coef_cp[(-i704)-i256+i255],True,1)*i704*X_coef_cp[i704])

            return(is_seq(0,(-n)+i256+i255-2)*B_theta_coef_cp[(-n)+i256+i255-2]*is_integer((-n)+i256+i255-2)*is_seq((-n)+i256+i255-2,i256)*py_sum(sum_arg_2,0,i255-i256))

        return(iota_coef[n-i255+2]*py_sum(sum_arg_3,0,i255))

    def sum_arg_1(i111):
        # Child args for sum_arg_1
        return(is_seq(0,n-i111)*diff(Z_coef_cp[2*i111-n],True,1)*iota_coef[n-i111]*is_integer(n-i111)*is_seq(n-i111,i111))


    out = ((is_seq(0,n+2)*dl_p*is_integer(n+2)*py_sum_parallel(sum_arg_31,0,n+2)+is_seq(0,n+2)*dl_p*is_integer(n+2)*py_sum_parallel(sum_arg_29,0,n+2)-2*is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_27,0,n)-2*is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_25,0,n))*tau_p+is_seq(0,n+2)*is_integer(n+2)*py_sum_parallel(sum_arg_9,0,n+2)+py_sum_parallel(sum_arg_7,ceil(n/2)+1,floor(n)+2)-py_sum_parallel(sum_arg_4,ceil(n/2)+1,floor(n)+2)+2*is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_23,0,n)-2*is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_21,0,n)-is_seq(0,n+2)*is_integer(n+2)*py_sum_parallel(sum_arg_19,0,n+2)+is_seq(0,n+2)*dl_p*kap_p*is_integer(n+2)*py_sum_parallel(sum_arg_17,0,n+2)-2*is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_15,0,n)+py_sum_parallel(sum_arg_13,ceil(n/2)+1,floor(n)+2)-py_sum_parallel(sum_arg_11,ceil(n/2)+1,floor(n)+2)-2*py_sum_parallel(sum_arg_1,ceil(n/2),floor(n))-2*is_seq(0,n)*is_integer(n)*diff(Z_coef_cp[n],False,1)+2*is_seq(0,n)*dl_p*kap_p*X_coef_cp[n]*is_integer(n))/2
    return(out)
# @partial(jit, static_argnums=(0,))
def validate_Ck(n, X_coef_cp, Y_coef_cp, Z_coef_cp,
    B_denom_coef_c, B_alpha_coef,
    B_psi_coef_cp, B_theta_coef_cp,
    kap_p, dl_p, tau_p, iota_coef):
    def sum_arg_29(i122):
        # Child args for sum_arg_29
        def sum_arg_28(i120):
            # Child args for sum_arg_28
            return(B_theta_coef_cp[i120]*X_coef_cp[n-i122-i120+2])

        return(i122*Z_coef_cp[i122]*py_sum(sum_arg_28,0,n-i122+2))

    def sum_arg_27(i118):
        # Child args for sum_arg_27
        def sum_arg_26(i116):
            # Child args for sum_arg_26
            return(B_psi_coef_cp[i116]*X_coef_cp[n-i118-i116])

        return(diff(Z_coef_cp[i118],True,1)*py_sum(sum_arg_26,0,n-i118))

    def sum_arg_25(i732):
        # Child args for sum_arg_25
        def sum_arg_24(i140):
            # Child args for sum_arg_24
            return(B_psi_coef_cp[i140]*diff(Y_coef_cp[n-i732-i140],False,1))

        return(diff(Z_coef_cp[i732],True,1)*py_sum(sum_arg_24,0,n-i732))

    def sum_arg_23(i726):
        # Child args for sum_arg_23
        def sum_arg_22(i136):
            # Child args for sum_arg_22
            return(B_psi_coef_cp[i136]*diff(Y_coef_cp[n-i726-i136],True,1))

        return(diff(Z_coef_cp[i726],False,1)*py_sum(sum_arg_22,0,n-i726))

    def sum_arg_21(i734):
        # Child args for sum_arg_21
        def sum_arg_20(i138):
            # Child args for sum_arg_20
            return(B_theta_coef_cp[i138]*(n-i734-i138+2)*Y_coef_cp[n-i734-i138+2])

        return(diff(Z_coef_cp[i734],False,1)*py_sum(sum_arg_20,0,n-i734+2))

    def sum_arg_19(i729):
        # Child args for sum_arg_19
        def sum_arg_18(i730):
            # Child args for sum_arg_18
            return(diff(Y_coef_cp[i730],True,1)*Z_coef_cp[(-n)-i730+2*i729-2]*((-n)-i730+2*i729-2)*is_seq(n-i729+2,i729-i730))

        return(is_seq(0,n-i729+2)*B_alpha_coef[n-i729+2]*is_integer(n-i729+2)*py_sum(sum_arg_18,0,i729))

    def sum_arg_17(i728):
        # Child args for sum_arg_17
        return(B_psi_coef_cp[i728]*diff(Y_coef_cp[n-i728],True,1))

    def sum_arg_16(i721):
        # Child args for sum_arg_16
        def sum_arg_15(i722):
            # Child args for sum_arg_15
            return(diff(Z_coef_cp[i722],True,1)*Y_coef_cp[(-n)-i722+2*i721-2]*((-n)-i722+2*i721-2)*is_seq(n-i721+2,i721-i722))

        return(is_seq(0,n-i721+2)*B_alpha_coef[n-i721+2]*is_integer(n-i721+2)*py_sum(sum_arg_15,0,i721))

    def sum_arg_14(i146):
        # Child args for sum_arg_14
        def sum_arg_13(i144):
            # Child args for sum_arg_13
            return(B_theta_coef_cp[i144]*(n-i146-i144+2)*Z_coef_cp[n-i146-i144+2])

        return(diff(Y_coef_cp[i146],False,1)*py_sum(sum_arg_13,0,n-i146+2))

    def sum_arg_12(i130):
        # Child args for sum_arg_12
        def sum_arg_11(i128):
            # Child args for sum_arg_11
            return(B_theta_coef_cp[i128]*X_coef_cp[n-i130-i128+2])

        return(i130*Y_coef_cp[i130]*py_sum(sum_arg_11,0,n-i130+2))

    def sum_arg_10(i126):
        # Child args for sum_arg_10
        def sum_arg_9(i124):
            # Child args for sum_arg_9
            return(B_psi_coef_cp[i124]*X_coef_cp[n-i126-i124])

        return(diff(Y_coef_cp[i126],True,1)*py_sum(sum_arg_9,0,n-i126))

    def sum_arg_8(i735):
        # Child args for sum_arg_8
        def sum_arg_7(i736):
            # Child args for sum_arg_7
            def sum_arg_6(i1191):
                # Child args for sum_arg_6
                return(i1191*Y_coef_cp[i1191]*diff(Z_coef_cp[(-i736)+i735-i1191],True,1))

            return(is_seq(0,(-n)+i736+i735-2)*B_theta_coef_cp[(-n)+i736+i735-2]*is_integer((-n)+i736+i735-2)*is_seq((-n)+i736+i735-2,i736)*py_sum(sum_arg_6,0,i735-i736))

        return(iota_coef[n-i735+2]*py_sum(sum_arg_7,0,i735))

    def sum_arg_5(i723):
        # Child args for sum_arg_5
        def sum_arg_4(i724):
            # Child args for sum_arg_4
            def sum_arg_3(i1175):
                # Child args for sum_arg_3
                return(i1175*Z_coef_cp[i1175]*diff(Y_coef_cp[(-i724)+i723-i1175],True,1))

            return(is_seq(0,(-n)+i724+i723-2)*B_theta_coef_cp[(-n)+i724+i723-2]*is_integer((-n)+i724+i723-2)*is_seq((-n)+i724+i723-2,i724)*py_sum(sum_arg_3,0,i723-i724))

        return(iota_coef[n-i723+2]*py_sum(sum_arg_4,0,i723))

    def sum_arg_2(i151):
        # Child args for sum_arg_2
        return(is_seq(0,n-i151)*diff(X_coef_cp[2*i151-n],True,1)*iota_coef[n-i151]*is_integer(n-i151)*is_seq(n-i151,i151))

    def sum_arg_1(i150):
        # Child args for sum_arg_1
        return(B_theta_coef_cp[i150]*(n-i150+2)*Y_coef_cp[n-i150+2])


    out = (-(is_seq(0,n+2)*dl_p*is_integer(n+2)*py_sum_parallel(sum_arg_29,0,n+2)*tau_p)/2)\
        +(is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_27,0,n)*tau_p)\
        +(-is_seq(0,n)*dl_p*Y_coef_cp[n]*is_integer(n)*tau_p)\
        +(-py_sum_parallel(sum_arg_8,ceil(0.5*n)+1,floor(n)+2)/2)\
        +(py_sum_parallel(sum_arg_5,ceil(0.5*n)+1,floor(n)+2)/2)\
        +(-is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_25,0,n))\
        +(is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_23,0,n))\
        +(-(is_seq(0,n+2)*is_integer(n+2)*py_sum_parallel(sum_arg_21,0,n+2))/2)\
        +(-py_sum_parallel(sum_arg_2,ceil(0.5*n),floor(n)))\
        +(-py_sum_parallel(sum_arg_19,ceil(0.5*n)+1,floor(n)+2)/2)\
        +(is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_17,0,n))\
        +(py_sum_parallel(sum_arg_16,ceil(0.5*n)+1,floor(n)+2)/2)\
        +((is_seq(0,n+2)*is_integer(n+2)*py_sum_parallel(sum_arg_14,0,n+2))/2)\
        +((is_seq(0,n+2)*dl_p*kap_p*is_integer(n+2)*py_sum_parallel(sum_arg_12,0,n+2))/2)\
        +(-is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_10,0,n))\
        +(-(is_seq(0,n+2)*dl_p*is_integer(n+2)*py_sum_parallel(sum_arg_1,0,n+2))/2)\
        +(-is_seq(0,n)*is_integer(n)*diff(X_coef_cp[n],False,1))\
        +(-is_seq(0,n)*dl_p*kap_p*Z_coef_cp[n]*is_integer(n))
    return(out)
# @partial(jit, static_argnums=(0,))
def validate_Ct(n, X_coef_cp, Y_coef_cp, Z_coef_cp,
    B_denom_coef_c, B_alpha_coef,
    B_psi_coef_cp, B_theta_coef_cp,
    kap_p, dl_p, tau_p, iota_coef):
    def sum_arg_33(i162):
        # Child args for sum_arg_33
        def sum_arg_32(i160):
            # Child args for sum_arg_32
            return(B_theta_coef_cp[i160]*Y_coef_cp[n-i162-i160+2])

        return(i162*Z_coef_cp[i162]*py_sum(sum_arg_32,0,n-i162+2))

    def sum_arg_31(i158):
        # Child args for sum_arg_31
        def sum_arg_30(i156):
            # Child args for sum_arg_30
            return(B_psi_coef_cp[i156]*Y_coef_cp[n-i158-i156])

        return(diff(Z_coef_cp[i158],True,1)*py_sum(sum_arg_30,0,n-i158))

    def sum_arg_29(i1200):
        # Child args for sum_arg_29
        def sum_arg_28(i188):
            # Child args for sum_arg_28
            return(B_psi_coef_cp[i188]*diff(X_coef_cp[n-i188-i1200],False,1))

        return(diff(Z_coef_cp[i1200],True,1)*py_sum(sum_arg_28,0,n-i1200))

    def sum_arg_27(i1206):
        # Child args for sum_arg_27
        def sum_arg_26(i184):
            # Child args for sum_arg_26
            return(B_psi_coef_cp[i184]*diff(X_coef_cp[n-i184-i1206],True,1))

        return(diff(Z_coef_cp[i1206],False,1)*py_sum(sum_arg_26,0,n-i1206))

    def sum_arg_25(i194):
        # Child args for sum_arg_25
        def sum_arg_24(i192):
            # Child args for sum_arg_24
            return(B_theta_coef_cp[i192]*(n-i194-i192+2)*Z_coef_cp[n-i194-i192+2])

        return(diff(X_coef_cp[i194],False,1)*py_sum(sum_arg_24,0,n-i194+2))

    def sum_arg_23(i1196):
        # Child args for sum_arg_23
        def sum_arg_22(i186):
            # Child args for sum_arg_22
            return(B_theta_coef_cp[i186]*(n-i186-i1196+2)*X_coef_cp[n-i186-i1196+2])

        return(diff(Z_coef_cp[i1196],False,1)*py_sum(sum_arg_22,0,n-i1196+2))

    def sum_arg_21(i178):
        # Child args for sum_arg_21
        def sum_arg_20(i176):
            # Child args for sum_arg_20
            return(B_theta_coef_cp[i176]*X_coef_cp[n-i178-i176+2])

        return(i178*X_coef_cp[i178]*py_sum(sum_arg_20,0,n-i178+2))

    def sum_arg_19(i174):
        # Child args for sum_arg_19
        def sum_arg_18(i172):
            # Child args for sum_arg_18
            return(B_theta_coef_cp[i172]*Z_coef_cp[n-i174-i172+2])

        return(i174*Z_coef_cp[i174]*py_sum(sum_arg_18,0,n-i174+2))

    def sum_arg_17(i170):
        # Child args for sum_arg_17
        def sum_arg_16(i168):
            # Child args for sum_arg_16
            return(B_psi_coef_cp[i168]*X_coef_cp[n-i170-i168])

        return(diff(X_coef_cp[i170],True,1)*py_sum(sum_arg_16,0,n-i170))

    def sum_arg_15(i166):
        # Child args for sum_arg_15
        def sum_arg_14(i164):
            # Child args for sum_arg_14
            return(B_psi_coef_cp[i164]*Z_coef_cp[n-i166-i164])

        return(diff(Z_coef_cp[i166],True,1)*py_sum(sum_arg_14,0,n-i166))

    def sum_arg_13(i1203):
        # Child args for sum_arg_13
        def sum_arg_12(i1204):
            # Child args for sum_arg_12
            return(diff(Z_coef_cp[i1204],True,1)*X_coef_cp[(-n)-i1204+2*i1203-2]*((-n)-i1204+2*i1203-2)*is_seq(n-i1203+2,i1203-i1204))

        return(is_seq(0,n-i1203+2)*B_alpha_coef[n-i1203+2]*is_integer(n-i1203+2)*py_sum(sum_arg_12,0,i1203))

    def sum_arg_11(i1202):
        # Child args for sum_arg_11
        return(B_psi_coef_cp[i1202]*diff(X_coef_cp[n-i1202],True,1))

    def sum_arg_10(i1193):
        # Child args for sum_arg_10
        def sum_arg_9(i1194):
            # Child args for sum_arg_9
            return(diff(X_coef_cp[i1194],True,1)*Z_coef_cp[(-n)-i1194+2*i1193-2]*((-n)-i1194+2*i1193-2)*is_seq(n-i1193+2,i1193-i1194))

        return(is_seq(0,n-i1193+2)*B_alpha_coef[n-i1193+2]*is_integer(n-i1193+2)*py_sum(sum_arg_9,0,i1193))

    def sum_arg_8(i199):
        # Child args for sum_arg_8
        return(is_seq(0,n-i199)*diff(Y_coef_cp[2*i199-n],True,1)*iota_coef[n-i199]*is_integer(n-i199)*is_seq(n-i199,i199))

    def sum_arg_7(i198):
        # Child args for sum_arg_7
        return(B_theta_coef_cp[i198]*(n-i198+2)*X_coef_cp[n-i198+2])

    def sum_arg_6(i1207):
        # Child args for sum_arg_6
        def sum_arg_5(i1208):
            # Child args for sum_arg_5
            def sum_arg_4(i1664):
                # Child args for sum_arg_4
                return(diff(X_coef_cp[(-i1664)-i1208+i1207],True,1)*i1664*Z_coef_cp[i1664])

            return(is_seq(0,(-n)+i1208+i1207-2)*B_theta_coef_cp[(-n)+i1208+i1207-2]*is_integer((-n)+i1208+i1207-2)*is_seq((-n)+i1208+i1207-2,i1208)*py_sum(sum_arg_4,0,i1207-i1208))

        return(iota_coef[n-i1207+2]*py_sum(sum_arg_5,0,i1207))

    def sum_arg_3(i1197):
        # Child args for sum_arg_3
        def sum_arg_2(i1198):
            # Child args for sum_arg_2
            def sum_arg_1(i1648):
                # Child args for sum_arg_1
                return(diff(Z_coef_cp[(-i1648)-i1198+i1197],True,1)*i1648*X_coef_cp[i1648])

            return(is_seq(0,(-n)+i1198+i1197-2)*B_theta_coef_cp[(-n)+i1198+i1197-2]*is_integer((-n)+i1198+i1197-2)*is_seq((-n)+i1198+i1197-2,i1198)*py_sum(sum_arg_1,0,i1197-i1198))

        return(iota_coef[n-i1197+2]*py_sum(sum_arg_2,0,i1197))


    out = (-(is_seq(0,n+2)*dl_p*is_integer(n+2)*py_sum_parallel(sum_arg_33,0,n+2)*tau_p)/2)\
        +(is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_31,0,n)*tau_p)\
        +(is_seq(0,n)*dl_p*X_coef_cp[n]*is_integer(n)*tau_p)\
        +(-py_sum_parallel(sum_arg_8,ceil(0.5*n),floor(n)))\
        +((is_seq(0,n+2)*dl_p*is_integer(n+2)*py_sum_parallel(sum_arg_7,0,n+2))/2)\
        +(-py_sum_parallel(sum_arg_6,ceil(0.5*n)+1,floor(n)+2)/2)\
        +(py_sum_parallel(sum_arg_3,ceil(0.5*n)+1,floor(n)+2)/2)\
        +(is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_29,0,n))\
        +(-is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_27,0,n))\
        +(-(is_seq(0,n+2)*is_integer(n+2)*py_sum_parallel(sum_arg_25,0,n+2))/2)\
        +((is_seq(0,n+2)*is_integer(n+2)*py_sum_parallel(sum_arg_23,0,n+2))/2)\
        +(-(is_seq(0,n+2)*dl_p*kap_p*is_integer(n+2)*py_sum_parallel(sum_arg_21,0,n+2))/2)\
        +(-(is_seq(0,n+2)*dl_p*kap_p*is_integer(n+2)*py_sum_parallel(sum_arg_19,0,n+2))/2)\
        +(is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_17,0,n))\
        +(is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_15,0,n))\
        +(-py_sum_parallel(sum_arg_13,ceil(0.5*n)+1,floor(n)+2)/2)\
        +(-is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_11,0,n))\
        +(py_sum_parallel(sum_arg_10,ceil(0.5*n)+1,floor(n)+2)/2)\
        +(-is_seq(0,n)*is_integer(n)*diff(Y_coef_cp[n],False,1))
    return(out)
# @partial(jit, static_argnums=(0,))
def validate_I(n, B_denom_coef_c,
    p_perp_coef_cp, Delta_coef_cp,
    iota_coef):
    def sum_arg_10(i1670):
        # Child args for sum_arg_10
        def sum_arg_9(i220):
            # Child args for sum_arg_9
            return(B_denom_coef_c[i220]*B_denom_coef_c[n-i220-i1670])

        return(diff(p_perp_coef_cp[i1670],False,1)*py_sum(sum_arg_9,0,n-i1670))

    def sum_arg_8(i1671):
        # Child args for sum_arg_8
        def sum_arg_7(i1672):
            # Child args for sum_arg_7
            def sum_arg_6(i1873):
                # Child args for sum_arg_6
                return(B_denom_coef_c[i1672-i1873]*B_denom_coef_c[i1873])

            return(diff(p_perp_coef_cp[(-n)-i1672+2*i1671],True,1)*is_seq(n-i1671,i1671-i1672)*py_sum(sum_arg_6,0,i1672))

        return(is_seq(0,n-i1671)*iota_coef[n-i1671]*is_integer(n-i1671)*py_sum(sum_arg_7,0,i1671))

    def sum_arg_5(i1667):
        # Child args for sum_arg_5
        def sum_arg_4(i1668):
            # Child args for sum_arg_4
            return(B_denom_coef_c[i1668]*diff(Delta_coef_cp[(-n)-i1668+2*i1667],True,1)*is_seq(n-i1667,i1667-i1668))

        return(is_seq(0,n-i1667)*iota_coef[n-i1667]*is_integer(n-i1667)*py_sum(sum_arg_4,0,i1667))

    def sum_arg_3(i1666):
        # Child args for sum_arg_3
        return(B_denom_coef_c[i1666]*diff(Delta_coef_cp[n-i1666],False,1))

    def sum_arg_2(i1875):
        # Child args for sum_arg_2
        def sum_arg_1(i224):
            # Child args for sum_arg_1
            return(Delta_coef_cp[i224]*diff(B_denom_coef_c[(-n)-i224+2*i1875],True,1))

        return(is_seq(0,n-i1875)*iota_coef[n-i1875]*is_integer(n-i1875)*is_seq(n-i1875,i1875)*py_sum(sum_arg_1,0,2*i1875-n))


    out = (2*py_sum_parallel(sum_arg_8,ceil(n/2),floor(n))+2*py_sum_parallel(sum_arg_5,ceil(n/2),floor(n))+2*is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_3,0,n)-py_sum_parallel(sum_arg_2,ceil(n/2),floor(n))+2*is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_10,0,n))/2
    return(out)
# @partial(jit, static_argnums=(0,))
def validate_II(n,
    B_theta_coef_cp, B_alpha_coef, B_denom_coef_c,
    p_perp_coef_cp, Delta_coef_cp, iota_coef):
    def sum_arg_23(i1890):
        # Child args for sum_arg_23
        def sum_arg_22(i228):
            # Child args for sum_arg_22
            def sum_arg_21(i226):
                # Child args for sum_arg_21
                return(B_denom_coef_c[i226]*B_denom_coef_c[n-i228-i226-i1890])

            return(diff(p_perp_coef_cp[i228],False,1)*py_sum(sum_arg_21,0,n-i228-i1890))

        return(B_theta_coef_cp[i1890]*py_sum(sum_arg_22,0,n-i1890))

    def sum_arg_20(i1887):
        # Child args for sum_arg_20
        def sum_arg_19(i1888):
            # Child args for sum_arg_19
            def sum_arg_18(i1886):
                # Child args for sum_arg_18
                return(Delta_coef_cp[i1886]*diff(B_theta_coef_cp[(-n)-i1888+2*i1887-i1886],True,1)*is_seq(n-i1887,(-i1888)+i1887-i1886))

            return(B_denom_coef_c[i1888]*py_sum(sum_arg_18,0,i1887-i1888))

        return(is_seq(0,n-i1887)*iota_coef[n-i1887]*is_integer(n-i1887)*py_sum(sum_arg_19,0,i1887))

    def sum_arg_17(i1884):
        # Child args for sum_arg_17
        def sum_arg_16(i1882):
            # Child args for sum_arg_16
            return(Delta_coef_cp[i1882]*diff(B_theta_coef_cp[n-i1884-i1882],False,1))

        return(B_denom_coef_c[i1884]*py_sum(sum_arg_16,0,n-i1884))

    def sum_arg_15(i1879):
        # Child args for sum_arg_15
        def sum_arg_14(i1880):
            # Child args for sum_arg_14
            return(B_denom_coef_c[i1880]*diff(B_theta_coef_cp[(-n)-i1880+2*i1879],True,1)*is_seq(n-i1879,i1879-i1880))

        return(is_seq(0,n-i1879)*iota_coef[n-i1879]*is_integer(n-i1879)*py_sum(sum_arg_14,0,i1879))

    def sum_arg_13(i1878):
        # Child args for sum_arg_13
        return(B_denom_coef_c[i1878]*diff(B_theta_coef_cp[n-i1878],False,1))

    def sum_arg_12(i1897):
        # Child args for sum_arg_12
        def sum_arg_11(i1898):
            # Child args for sum_arg_11
            def sum_arg_10(i2338):
                # Child args for sum_arg_10
                return(diff(B_denom_coef_c[(-i2338)-i1898+i1897],True,1)*Delta_coef_cp[i2338])

            return(is_seq(0,(-n)+i1898+i1897)*B_theta_coef_cp[(-n)+i1898+i1897]*is_integer((-n)+i1898+i1897)*is_seq((-n)+i1898+i1897,i1898)*py_sum(sum_arg_10,0,i1897-i1898))

        return(iota_coef[n-i1897]*py_sum(sum_arg_11,0,i1897))

    def sum_arg_9(i1895):
        # Child args for sum_arg_9
        def sum_arg_8(i238):
            # Child args for sum_arg_8
            return(Delta_coef_cp[i238]*diff(B_denom_coef_c[(-n)-i238+2*i1895],True,1))

        return(is_seq(0,n-i1895)*B_alpha_coef[n-i1895]*is_integer(n-i1895)*is_seq(n-i1895,i1895)*py_sum(sum_arg_8,0,2*i1895-n))

    def sum_arg_7(i1893):
        # Child args for sum_arg_7
        def sum_arg_6(i1894):
            # Child args for sum_arg_6
            def sum_arg_5(i2322):
                # Child args for sum_arg_5
                def sum_arg_4(i230):
                    # Child args for sum_arg_4
                    return(B_denom_coef_c[i230]*B_denom_coef_c[(-i2322)-i230-i1894+i1893])

                return(diff(p_perp_coef_cp[i2322],True,1)*py_sum(sum_arg_4,0,(-i2322)-i1894+i1893))

            return(is_seq(0,(-n)+i1894+i1893)*B_theta_coef_cp[(-n)+i1894+i1893]*is_integer((-n)+i1894+i1893)*is_seq((-n)+i1894+i1893,i1894)*py_sum(sum_arg_5,0,i1893-i1894))

        return(iota_coef[n-i1893]*py_sum(sum_arg_6,0,i1893))

    def sum_arg_3(i1891):
        # Child args for sum_arg_3
        def sum_arg_2(i234):
            # Child args for sum_arg_2
            def sum_arg_1(i230):
                # Child args for sum_arg_1
                return(B_denom_coef_c[i230]*B_denom_coef_c[(-n)-i234-i230+2*i1891])

            return(diff(p_perp_coef_cp[i234],True,1)*py_sum(sum_arg_1,0,(-n)-i234+2*i1891))

        return(is_seq(0,n-i1891)*B_alpha_coef[n-i1891]*is_integer(n-i1891)*is_seq(n-i1891,i1891)*py_sum(sum_arg_2,0,2*i1891-n))


    out = (py_sum_parallel(sum_arg_9,ceil(n/2),floor(n))+2*py_sum_parallel(sum_arg_7,ceil(n/2),floor(n))-2*py_sum_parallel(sum_arg_3,ceil(n/2),floor(n))+2*is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_23,0,n)-2*py_sum_parallel(sum_arg_20,ceil(n/2),floor(n))-2*is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_17,0,n)+2*py_sum_parallel(sum_arg_15,ceil(n/2),floor(n))+2*is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_13,0,n)-py_sum_parallel(sum_arg_12,ceil(n/2),floor(n)))/2
    return(out)
# @partial(jit, static_argnums=(0,))
def validate_III(n,
    B_theta_coef_cp, B_psi_coef_cp,
    B_alpha_coef, B_denom_coef_c,
    p_perp_coef_cp, Delta_coef_cp,
    iota_coef):
    def sum_arg_29(i2368):
        # Child args for sum_arg_29
        def sum_arg_28(i2366):
            # Child args for sum_arg_28
            return(Delta_coef_cp[i2366]*diff(B_psi_coef_cp[n-i2368-i2366],False,1))

        return(B_denom_coef_c[i2368]*py_sum(sum_arg_28,0,n-i2368))

    def sum_arg_27(i2355):
        # Child args for sum_arg_27
        def sum_arg_26(i2356):
            # Child args for sum_arg_26
            def sum_arg_25(i2350):
                # Child args for sum_arg_25
                return(Delta_coef_cp[i2350]*diff(B_psi_coef_cp[(-n)-i2356+2*i2355-i2350],True,1)*is_seq(n-i2355,(-i2356)+i2355-i2350))

            return(B_denom_coef_c[i2356]*py_sum(sum_arg_25,0,i2355-i2356))

        return(is_seq(0,n-i2355)*iota_coef[n-i2355]*is_integer(n-i2355)*py_sum(sum_arg_26,0,i2355))

    def sum_arg_24(i2353):
        # Child args for sum_arg_24
        def sum_arg_23(i2354):
            # Child args for sum_arg_23
            def sum_arg_22(i2348):
                # Child args for sum_arg_22
                return(B_psi_coef_cp[i2348]*diff(Delta_coef_cp[(-n)-i2354+2*i2353-i2348],True,1)*is_seq(n-i2353,(-i2354)+i2353-i2348))

            return(B_denom_coef_c[i2354]*py_sum(sum_arg_22,0,i2353-i2354))

        return(is_seq(0,n-i2353)*iota_coef[n-i2353]*is_integer(n-i2353)*py_sum(sum_arg_23,0,i2353))

    def sum_arg_21(i2352):
        # Child args for sum_arg_21
        def sum_arg_20(i2346):
            # Child args for sum_arg_20
            return(B_psi_coef_cp[i2346]*diff(Delta_coef_cp[n-i2352-i2346],False,1))

        return(B_denom_coef_c[i2352]*py_sum(sum_arg_20,0,n-i2352))

    def sum_arg_19(i2343):
        # Child args for sum_arg_19
        def sum_arg_18(i2344):
            # Child args for sum_arg_18
            def sum_arg_17(i244):
                # Child args for sum_arg_17
                return(B_denom_coef_c[i244]*B_denom_coef_c[(-n)-i244-i2344+2*i2343-2]*is_seq(n-i2343+2,(-i244)-i2344+i2343))

            return(i2344*p_perp_coef_cp[i2344]*py_sum(sum_arg_17,0,i2343-i2344))

        return(is_seq(0,n-i2343+2)*B_alpha_coef[n-i2343+2]*is_integer(n-i2343+2)*py_sum(sum_arg_18,0,i2343))

    def sum_arg_16(i2375):
        # Child args for sum_arg_16
        def sum_arg_15(i2376):
            # Child args for sum_arg_15
            return(B_denom_coef_c[i2376]*diff(B_psi_coef_cp[(-n)-i2376+2*i2375],True,1)*is_seq(n-i2375,i2375-i2376))

        return(is_seq(0,n-i2375)*iota_coef[n-i2375]*is_integer(n-i2375)*py_sum(sum_arg_15,0,i2375))

    def sum_arg_14(i2364):
        # Child args for sum_arg_14
        return(B_denom_coef_c[i2364]*diff(B_psi_coef_cp[n-i2364],False,1))

    def sum_arg_13(i2361):
        # Child args for sum_arg_13
        def sum_arg_12(i2362):
            # Child args for sum_arg_12
            return(B_denom_coef_c[i2362]*Delta_coef_cp[(-n)-i2362+2*i2361-2]*is_seq(n-i2361+2,i2361-i2362))

        return((is_seq(0,n-i2361+2)*n-is_seq(0,n-i2361+2)*i2361+2*is_seq(0,n-i2361+2))*B_alpha_coef[n-i2361+2]*is_integer(n-i2361+2)*py_sum(sum_arg_12,0,i2361))

    def sum_arg_11(i2341):
        # Child args for sum_arg_11
        def sum_arg_10(i2342):
            # Child args for sum_arg_10
            return(i2342*B_denom_coef_c[i2342]*Delta_coef_cp[(-n)-i2342+2*i2341-2]*is_seq(n-i2341+2,i2341-i2342))

        return(is_seq(0,n-i2341+2)*B_alpha_coef[n-i2341+2]*is_integer(n-i2341+2)*py_sum(sum_arg_10,0,i2341))

    def sum_arg_9(i2373):
        # Child args for sum_arg_9
        def sum_arg_6(i2374):
            # Child args for sum_arg_6
            def sum_arg_5(i2372):
                # Child args for sum_arg_5
                return(is_seq(0,(-n)-i2374+2*i2373-i2372-2)*Delta_coef_cp[i2372]*B_theta_coef_cp[(-n)-i2374+2*i2373-i2372-2]*is_integer((-n)-i2374+2*i2373-i2372-2)*is_seq((-n)-i2374+2*i2373-i2372-2,(-i2374)+i2373-i2372))

            return(B_denom_coef_c[i2374]*py_sum(sum_arg_5,0,i2373-i2374))

        def sum_arg_8(i2374):
            # Child args for sum_arg_8
            def sum_arg_7(i2372):
                # Child args for sum_arg_7
                return(is_seq(0,(-n)-i2374+2*i2373-i2372-2)*Delta_coef_cp[i2372]*B_theta_coef_cp[(-n)-i2374+2*i2373-i2372-2]*is_integer((-n)-i2374+2*i2373-i2372-2)*is_seq((-n)-i2374+2*i2373-i2372-2,(-i2374)+i2373-i2372))

            return(B_denom_coef_c[i2374]*py_sum(sum_arg_7,0,i2373-i2374))

        return(iota_coef[n-i2373+2]*(n*py_sum(sum_arg_8,0,i2373)+(2-i2373)*py_sum(sum_arg_6,0,i2373)))

    def sum_arg_4(i2369):
        # Child args for sum_arg_4
        def sum_arg_2(i2370):
            # Child args for sum_arg_2
            return(is_seq(0,(-n)-i2370+2*i2369-2)*B_denom_coef_c[i2370]*B_theta_coef_cp[(-n)-i2370+2*i2369-2]*is_integer((-n)-i2370+2*i2369-2)*is_seq((-n)-i2370+2*i2369-2,i2369-i2370))

        def sum_arg_3(i2370):
            # Child args for sum_arg_3
            return(is_seq(0,(-n)-i2370+2*i2369-2)*B_denom_coef_c[i2370]*B_theta_coef_cp[(-n)-i2370+2*i2369-2]*is_integer((-n)-i2370+2*i2369-2)*is_seq((-n)-i2370+2*i2369-2,i2369-i2370))

        return(iota_coef[n-i2369+2]*(n*py_sum(sum_arg_3,0,i2369)+(2-i2369)*py_sum(sum_arg_2,0,i2369)))

    def sum_arg_1(i2357):
        # Child args for sum_arg_1
        return((is_seq(0,n-i2357+2)*B_denom_coef_c[(-n)+2*i2357-2]*n+(2*is_seq(0,n-i2357+2)-is_seq(0,n-i2357+2)*i2357)*B_denom_coef_c[(-n)+2*i2357-2])*B_alpha_coef[n-i2357+2]*is_integer(n-i2357+2)*is_seq(n-i2357+2,i2357))


    out = (4*py_sum_parallel(sum_arg_9,ceil(n/2)+1,floor(n)+2)-4*py_sum_parallel(sum_arg_4,ceil(n/2)+1,floor(n)+2)+4*is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_29,0,n)+4*py_sum_parallel(sum_arg_27,ceil(n/2),floor(n))+4*py_sum_parallel(sum_arg_24,ceil(n/2),floor(n))+4*is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_21,0,n)+2*py_sum_parallel(sum_arg_19,ceil(n/2)+1,floor(n)+2)-4*py_sum_parallel(sum_arg_16,ceil(n/2),floor(n))-4*is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_14,0,n)-4*py_sum_parallel(sum_arg_13,ceil(n/2)+1,floor(n)+2)-py_sum_parallel(sum_arg_11,ceil(n/2)+1,floor(n)+2)+4*py_sum_parallel(sum_arg_1,ceil(n/2)+1,floor(n)+2))/4
    return(out)
# @partial(jit, static_argnums=(0,))
def validate_E6(n,
    B_theta_coef_cp, B_psi_coef_cp,
    B_alpha_coef, B_denom_coef_c,
    p_perp_coef_cp, Delta_coef_cp,
    iota_coef):
    def sum_arg_29(i2420):
        # Child args for sum_arg_29
        def sum_arg_28(i2382):
            # Child args for sum_arg_28
            return(B_psi_coef_cp[i2382]*diff(Delta_coef_cp[n-i2420-i2382],False,1))

        return(B_denom_coef_c[i2420]*py_sum(sum_arg_28,0,n-i2420))

    def sum_arg_27(i2416):
        # Child args for sum_arg_27
        def sum_arg_26(i2414):
            # Child args for sum_arg_26
            return(Delta_coef_cp[i2414]*diff(B_psi_coef_cp[n-i2416-i2414],False,1))

        return(B_denom_coef_c[i2416]*py_sum(sum_arg_26,0,n-i2416))

    def sum_arg_25(i2411):
        # Child args for sum_arg_25
        def sum_arg_24(i2412):
            # Child args for sum_arg_24
            def sum_arg_23(i2384):
                # Child args for sum_arg_23
                return(B_psi_coef_cp[i2384]*diff(Delta_coef_cp[(-n)-i2412+2*i2411-i2384],True,1))

            return(B_denom_coef_c[i2412]*is_seq(n-i2411,i2411-i2412)*py_sum(sum_arg_23,0,(-n)-i2412+2*i2411))

        return(is_seq(0,n-i2411)*iota_coef[n-i2411]*is_integer(n-i2411)*py_sum(sum_arg_24,0,i2411))

    def sum_arg_22(i2407):
        # Child args for sum_arg_22
        def sum_arg_21(i2408):
            # Child args for sum_arg_21
            def sum_arg_20(i2404):
                # Child args for sum_arg_20
                return(Delta_coef_cp[i2404]*diff(B_psi_coef_cp[(-n)-i2408+2*i2407-i2404],True,1))

            return(B_denom_coef_c[i2408]*is_seq(n-i2407,i2407-i2408)*py_sum(sum_arg_20,0,(-n)-i2408+2*i2407))

        return(is_seq(0,n-i2407)*iota_coef[n-i2407]*is_integer(n-i2407)*py_sum(sum_arg_21,0,i2407))

    def sum_arg_19(i2397):
        # Child args for sum_arg_19
        def sum_arg_18(i2398):
            # Child args for sum_arg_18
            return(B_denom_coef_c[i2398]*Delta_coef_cp[(-n)-i2398+2*i2397-2]*is_seq(n-i2397+2,i2397-i2398))

        return((is_seq(0,n-i2397+2)*n-is_seq(0,n-i2397+2)*i2397+2*is_seq(0,n-i2397+2))*B_alpha_coef[n-i2397+2]*is_integer(n-i2397+2)*py_sum(sum_arg_18,0,i2397))

    def sum_arg_17(i2391):
        # Child args for sum_arg_17
        def sum_arg_16(i2392):
            # Child args for sum_arg_16
            return(B_denom_coef_c[i2392]*diff(B_psi_coef_cp[(-n)-i2392+2*i2391],True,1)*is_seq(n-i2391,i2391-i2392))

        return(is_seq(0,n-i2391)*iota_coef[n-i2391]*is_integer(n-i2391)*py_sum(sum_arg_16,0,i2391))

    def sum_arg_15(i2390):
        # Child args for sum_arg_15
        return(B_denom_coef_c[i2390]*diff(B_psi_coef_cp[n-i2390],False,1))

    def sum_arg_14(i2385):
        # Child args for sum_arg_14
        def sum_arg_13(i2386):
            # Child args for sum_arg_13
            return(i2386*B_denom_coef_c[i2386]*Delta_coef_cp[(-n)-i2386+2*i2385-2]*is_seq(n-i2385+2,i2385-i2386))

        return(is_seq(0,n-i2385+2)*B_alpha_coef[n-i2385+2]*is_integer(n-i2385+2)*py_sum(sum_arg_13,0,i2385))

    def sum_arg_12(i2377):
        # Child args for sum_arg_12
        def sum_arg_11(i2378):
            # Child args for sum_arg_11
            def sum_arg_10(i246):
                # Child args for sum_arg_10
                return(B_denom_coef_c[i246]*B_denom_coef_c[(-n)-i246-i2378+2*i2377-2])

            return(i2378*p_perp_coef_cp[i2378]*is_seq(n-i2377+2,i2377-i2378)*py_sum(sum_arg_10,0,(-n)-i2378+2*i2377-2))

        return(is_seq(0,n-i2377+2)*B_alpha_coef[n-i2377+2]*is_integer(n-i2377+2)*py_sum(sum_arg_11,0,i2377))

    def sum_arg_9(i2417):
        # Child args for sum_arg_9
        def sum_arg_7(i2418):
            # Child args for sum_arg_7
            return(is_seq(0,(-n)-i2418+2*i2417-2)*B_denom_coef_c[i2418]*B_theta_coef_cp[(-n)-i2418+2*i2417-2]*is_integer((-n)-i2418+2*i2417-2)*is_seq((-n)-i2418+2*i2417-2,i2417-i2418))

        def sum_arg_8(i2418):
            # Child args for sum_arg_8
            return(is_seq(0,(-n)-i2418+2*i2417-2)*B_denom_coef_c[i2418]*B_theta_coef_cp[(-n)-i2418+2*i2417-2]*is_integer((-n)-i2418+2*i2417-2)*is_seq((-n)-i2418+2*i2417-2,i2417-i2418))

        return(iota_coef[n-i2417+2]*(n*py_sum(sum_arg_8,0,i2417)+(2-i2417)*py_sum(sum_arg_7,0,i2417)))

    def sum_arg_6(i2401):
        # Child args for sum_arg_6
        def sum_arg_3(i2402):
            # Child args for sum_arg_3
            def sum_arg_2(i2400):
                # Child args for sum_arg_2
                return(is_seq(0,(-n)-i2402+2*i2401-i2400-2)*Delta_coef_cp[i2400]*B_theta_coef_cp[(-n)-i2402+2*i2401-i2400-2]*is_integer((-n)-i2402+2*i2401-i2400-2)*is_seq((-n)-i2402+2*i2401-i2400-2,(-i2402)+i2401-i2400))

            return(B_denom_coef_c[i2402]*py_sum(sum_arg_2,0,i2401-i2402))

        def sum_arg_5(i2402):
            # Child args for sum_arg_5
            def sum_arg_4(i2400):
                # Child args for sum_arg_4
                return(is_seq(0,(-n)-i2402+2*i2401-i2400-2)*Delta_coef_cp[i2400]*B_theta_coef_cp[(-n)-i2402+2*i2401-i2400-2]*is_integer((-n)-i2402+2*i2401-i2400-2)*is_seq((-n)-i2402+2*i2401-i2400-2,(-i2402)+i2401-i2400))

            return(B_denom_coef_c[i2402]*py_sum(sum_arg_4,0,i2401-i2402))

        return(iota_coef[n-i2401+2]*(n*py_sum(sum_arg_5,0,i2401)+(2-i2401)*py_sum(sum_arg_3,0,i2401)))

    def sum_arg_1(i2393):
        # Child args for sum_arg_1
        return((is_seq(0,n-i2393+2)*B_denom_coef_c[(-n)+2*i2393-2]*n+(2*is_seq(0,n-i2393+2)-is_seq(0,n-i2393+2)*i2393)*B_denom_coef_c[(-n)+2*i2393-2])*B_alpha_coef[n-i2393+2]*is_integer(n-i2393+2)*is_seq(n-i2393+2,i2393))


    out = ((-4*py_sum_parallel(sum_arg_9,ceil(n/2)+1,floor(n)+2))+4*py_sum_parallel(sum_arg_6,ceil(n/2)+1,floor(n)+2)+4*is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_29,0,n)+4*is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_27,0,n)+4*py_sum_parallel(sum_arg_25,ceil(n/2),floor(n))+4*py_sum_parallel(sum_arg_22,ceil(n/2),floor(n))-4*py_sum_parallel(sum_arg_19,ceil(n/2)+1,floor(n)+2)-4*py_sum_parallel(sum_arg_17,ceil(n/2),floor(n))-4*is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_15,0,n)-py_sum_parallel(sum_arg_14,ceil(n/2)+1,floor(n)+2)+2*py_sum_parallel(sum_arg_12,ceil(n/2)+1,floor(n)+2)+4*py_sum_parallel(sum_arg_1,ceil(n/2)+1,floor(n)+2))/4
    return(out)
# @partial(jit, static_argnums=(0,))
def validate_D2(n, X_coef_cp, Y_coef_cp, Z_coef_cp,
    B_denom_coef_c, B_alpha_coef,
    B_psi_coef_cp, B_theta_coef_cp,
    kap_p, dl_p, tau_p, iota_coef):
    def sum_arg_15(i2446):
        # Child args for sum_arg_15
        return(i2446*X_coef_cp[i2446]*Y_coef_cp[n-i2446+2])

    def sum_arg_14(i2444):
        # Child args for sum_arg_14
        return((X_coef_cp[i2444]*n+(2-i2444)*X_coef_cp[i2444])*Y_coef_cp[n-i2444+2])

    def sum_arg_13(i2459):
        # Child args for sum_arg_13
        def sum_arg_12(i2460):
            # Child args for sum_arg_12
            return(i2460*Z_coef_cp[i2460]*diff(Z_coef_cp[(-n)-i2460+2*i2459-2],True,1)*is_seq(n-i2459+2,i2459-i2460))

        return(is_seq(0,n-i2459+2)*iota_coef[n-i2459+2]*is_integer(n-i2459+2)*py_sum(sum_arg_12,0,i2459))

    def sum_arg_11(i2457):
        # Child args for sum_arg_11
        def sum_arg_10(i2458):
            # Child args for sum_arg_10
            return(i2458*Y_coef_cp[i2458]*diff(Y_coef_cp[(-n)-i2458+2*i2457-2],True,1)*is_seq(n-i2457+2,i2457-i2458))

        return(is_seq(0,n-i2457+2)*iota_coef[n-i2457+2]*is_integer(n-i2457+2)*py_sum(sum_arg_10,0,i2457))

    def sum_arg_9(i2455):
        # Child args for sum_arg_9
        def sum_arg_8(i2456):
            # Child args for sum_arg_8
            return(i2456*X_coef_cp[i2456]*diff(X_coef_cp[(-n)-i2456+2*i2455-2],True,1)*is_seq(n-i2455+2,i2455-i2456))

        return(is_seq(0,n-i2455+2)*iota_coef[n-i2455+2]*is_integer(n-i2455+2)*py_sum(sum_arg_8,0,i2455))

    def sum_arg_7(i2454):
        # Child args for sum_arg_7
        return(i2454*Z_coef_cp[i2454]*diff(Z_coef_cp[n-i2454+2],False,1))

    def sum_arg_6(i2452):
        # Child args for sum_arg_6
        return(i2452*Y_coef_cp[i2452]*diff(Y_coef_cp[n-i2452+2],False,1))

    def sum_arg_5(i2450):
        # Child args for sum_arg_5
        return(i2450*X_coef_cp[i2450]*diff(X_coef_cp[n-i2450+2],False,1))

    def sum_arg_4(i201):
        # Child args for sum_arg_4
        def sum_arg_3(i202):
            # Child args for sum_arg_3
            return(B_psi_coef_cp[i202]*B_denom_coef_c[(-n)-i202+2*i201]*is_seq(n-i201,i201-i202))

        return(is_seq(0,n-i201)*B_alpha_coef[n-i201]*is_integer(n-i201)*py_sum(sum_arg_3,0,i201))

    def sum_arg_2(i2448):
        # Child args for sum_arg_2
        return(i2448*X_coef_cp[i2448]*Z_coef_cp[n-i2448+2])

    def sum_arg_1(i2442):
        # Child args for sum_arg_1
        return((X_coef_cp[i2442]*n+(2-i2442)*X_coef_cp[i2442])*Z_coef_cp[n-i2442+2])


    out = -((is_seq(0,n+2)*dl_p*is_integer(n+2)*py_sum_parallel(sum_arg_15,0,n+2)-is_seq(0,n+2)*dl_p*is_integer(n+2)*py_sum_parallel(sum_arg_14,0,n+2))*tau_p+py_sum_parallel(sum_arg_9,ceil(n/2)+1,floor(n)+2)+is_seq(0,n+2)*is_integer(n+2)*py_sum_parallel(sum_arg_7,0,n+2)+is_seq(0,n+2)*is_integer(n+2)*py_sum_parallel(sum_arg_6,0,n+2)+is_seq(0,n+2)*is_integer(n+2)*py_sum_parallel(sum_arg_5,0,n+2)-2*py_sum_parallel(sum_arg_4,ceil(n/2),floor(n))+is_seq(0,n+2)*dl_p*kap_p*is_integer(n+2)*py_sum_parallel(sum_arg_2,0,n+2)+py_sum_parallel(sum_arg_13,ceil(n/2)+1,floor(n)+2)+py_sum_parallel(sum_arg_11,ceil(n/2)+1,floor(n)+2)-is_seq(0,n+2)*dl_p*kap_p*is_integer(n+2)*py_sum_parallel(sum_arg_1,0,n+2)+(is_seq(0,n+2)*dl_p*n+2*is_seq(0,n+2)*dl_p)*Z_coef_cp[n+2]*is_integer(n+2))/2
    return(out)
# @partial(jit, static_argnums=(0,))
def validate_D3(n, X_coef_cp, Y_coef_cp, Z_coef_cp,
    B_denom_coef_c, B_alpha_coef,
    B_psi_coef_cp, B_theta_coef_cp,
    kap_p, dl_p, tau_p, iota_coef):
    def sum_arg_15(i2438):
        # Child args for sum_arg_15
        return(X_coef_cp[i2438]*diff(Y_coef_cp[n-i2438],True,1))

    def sum_arg_14(i2434):
        # Child args for sum_arg_14
        return(Y_coef_cp[i2434]*diff(X_coef_cp[n-i2434],True,1))

    def sum_arg_13(i2440):
        # Child args for sum_arg_13
        return(X_coef_cp[i2440]*diff(Z_coef_cp[n-i2440],True,1))

    def sum_arg_12(i2436):
        # Child args for sum_arg_12
        return(Z_coef_cp[i2436]*diff(X_coef_cp[n-i2436],True,1))

    def sum_arg_11(i2431):
        # Child args for sum_arg_11
        def sum_arg_10(i2432):
            # Child args for sum_arg_10
            return(diff(Z_coef_cp[i2432],True,1)*diff(Z_coef_cp[(-n)-i2432+2*i2431],True,1)*is_seq(n-i2431,i2431-i2432))

        return(is_seq(0,n-i2431)*iota_coef[n-i2431]*is_integer(n-i2431)*py_sum(sum_arg_10,0,i2431))

    def sum_arg_9(i2430):
        # Child args for sum_arg_9
        return(diff(Z_coef_cp[i2430],True,1)*diff(Z_coef_cp[n-i2430],False,1))

    def sum_arg_8(i2427):
        # Child args for sum_arg_8
        def sum_arg_7(i2428):
            # Child args for sum_arg_7
            return(diff(Y_coef_cp[i2428],True,1)*diff(Y_coef_cp[(-n)-i2428+2*i2427],True,1)*is_seq(n-i2427,i2427-i2428))

        return(is_seq(0,n-i2427)*iota_coef[n-i2427]*is_integer(n-i2427)*py_sum(sum_arg_7,0,i2427))

    def sum_arg_6(i2426):
        # Child args for sum_arg_6
        return(diff(Y_coef_cp[i2426],True,1)*diff(Y_coef_cp[n-i2426],False,1))

    def sum_arg_5(i2423):
        # Child args for sum_arg_5
        def sum_arg_4(i2424):
            # Child args for sum_arg_4
            return(diff(X_coef_cp[i2424],True,1)*diff(X_coef_cp[(-n)-i2424+2*i2423],True,1)*is_seq(n-i2423,i2423-i2424))

        return(is_seq(0,n-i2423)*iota_coef[n-i2423]*is_integer(n-i2423)*py_sum(sum_arg_4,0,i2423))

    def sum_arg_3(i2422):
        # Child args for sum_arg_3
        return(diff(X_coef_cp[i2422],True,1)*diff(X_coef_cp[n-i2422],False,1))

    def sum_arg_2(i209):
        # Child args for sum_arg_2
        def sum_arg_1(i210):
            # Child args for sum_arg_1
            return(B_theta_coef_cp[i210]*B_denom_coef_c[(-n)-i210+2*i209]*is_seq(n-i209,i209-i210))

        return(is_seq(0,n-i209)*B_alpha_coef[n-i209]*is_integer(n-i209)*py_sum(sum_arg_1,0,i209))


    out = ((is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_15,0,n)-is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_14,0,n))*tau_p)\
        +(-is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_9,0,n))\
        +(-py_sum_parallel(sum_arg_8,ceil(n/2),floor(n)))\
        +(-is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_6,0,n))\
        +(-py_sum_parallel(sum_arg_5,ceil(n/2),floor(n)))\
        +(-is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_3,0,n))\
        +(py_sum_parallel(sum_arg_2,ceil(n/2),floor(n)))\
        +(is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_13,0,n))\
        +(-is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_12,0,n))\
        +(-py_sum_parallel(sum_arg_11,ceil(n/2),floor(n)))\
        +(-is_seq(0,n)*dl_p*is_integer(n)*diff(Z_coef_cp[n],True,1))
    return(out)
# @partial(jit, static_argnums=(0,))
def validate_kt(n, X_coef_cp, Y_coef_cp, Z_coef_cp,
    B_denom_coef_c, B_alpha_coef,
    B_psi_coef_cp, B_theta_coef_cp,
    kap_p, dl_p, tau_p, iota_coef):
    def sum_arg_62(i162):
        # Child args for sum_arg_62
        def sum_arg_61(i160):
            # Child args for sum_arg_61
            return(B_theta_coef_cp[i160]*Y_coef_cp[n-i162-i160+2])

        return(i162*Z_coef_cp[i162]*py_sum(sum_arg_61,0,n-i162+2))

    def sum_arg_60(i158):
        # Child args for sum_arg_60
        def sum_arg_59(i156):
            # Child args for sum_arg_59
            return(B_psi_coef_cp[i156]*Y_coef_cp[n-i158-i156])

        return(diff(Z_coef_cp[i158],True,1)*py_sum(sum_arg_59,0,n-i158))

    def sum_arg_58(i1200):
        # Child args for sum_arg_58
        def sum_arg_57(i188):
            # Child args for sum_arg_57
            return(B_psi_coef_cp[i188]*diff(X_coef_cp[n-i188-i1200],False,1))

        return(diff(Z_coef_cp[i1200],True,1)*py_sum(sum_arg_57,0,n-i1200))

    def sum_arg_56(i1206):
        # Child args for sum_arg_56
        def sum_arg_55(i184):
            # Child args for sum_arg_55
            return(B_psi_coef_cp[i184]*diff(X_coef_cp[n-i184-i1206],True,1))

        return(diff(Z_coef_cp[i1206],False,1)*py_sum(sum_arg_55,0,n-i1206))

    def sum_arg_54(i194):
        # Child args for sum_arg_54
        def sum_arg_53(i192):
            # Child args for sum_arg_53
            return(B_theta_coef_cp[i192]*(n-i194-i192+2)*Z_coef_cp[n-i194-i192+2])

        return(diff(X_coef_cp[i194],False,1)*py_sum(sum_arg_53,0,n-i194+2))

    def sum_arg_52(i1196):
        # Child args for sum_arg_52
        def sum_arg_51(i186):
            # Child args for sum_arg_51
            return(B_theta_coef_cp[i186]*(n-i186-i1196+2)*X_coef_cp[n-i186-i1196+2])

        return(diff(Z_coef_cp[i1196],False,1)*py_sum(sum_arg_51,0,n-i1196+2))

    def sum_arg_50(i178):
        # Child args for sum_arg_50
        def sum_arg_49(i176):
            # Child args for sum_arg_49
            return(B_theta_coef_cp[i176]*X_coef_cp[n-i178-i176+2])

        return(i178*X_coef_cp[i178]*py_sum(sum_arg_49,0,n-i178+2))

    def sum_arg_48(i174):
        # Child args for sum_arg_48
        def sum_arg_47(i172):
            # Child args for sum_arg_47
            return(B_theta_coef_cp[i172]*Z_coef_cp[n-i174-i172+2])

        return(i174*Z_coef_cp[i174]*py_sum(sum_arg_47,0,n-i174+2))

    def sum_arg_46(i170):
        # Child args for sum_arg_46
        def sum_arg_45(i168):
            # Child args for sum_arg_45
            return(B_psi_coef_cp[i168]*X_coef_cp[n-i170-i168])

        return(diff(X_coef_cp[i170],True,1)*py_sum(sum_arg_45,0,n-i170))

    def sum_arg_44(i166):
        # Child args for sum_arg_44
        def sum_arg_43(i164):
            # Child args for sum_arg_43
            return(B_psi_coef_cp[i164]*Z_coef_cp[n-i166-i164])

        return(diff(Z_coef_cp[i166],True,1)*py_sum(sum_arg_43,0,n-i166))

    def sum_arg_42(i1203):
        # Child args for sum_arg_42
        def sum_arg_41(i1204):
            # Child args for sum_arg_41
            return(diff(Z_coef_cp[i1204],True,1)*X_coef_cp[(-n)-i1204+2*i1203-2]*((-n)-i1204+2*i1203-2)*is_seq(n-i1203+2,i1203-i1204))

        return(is_seq(0,n-i1203+2)*B_alpha_coef[n-i1203+2]*is_integer(n-i1203+2)*py_sum(sum_arg_41,0,i1203))

    def sum_arg_40(i1202):
        # Child args for sum_arg_40
        return(B_psi_coef_cp[i1202]*diff(X_coef_cp[n-i1202],True,1))

    def sum_arg_39(i1193):
        # Child args for sum_arg_39
        def sum_arg_38(i1194):
            # Child args for sum_arg_38
            return(diff(X_coef_cp[i1194],True,1)*Z_coef_cp[(-n)-i1194+2*i1193-2]*((-n)-i1194+2*i1193-2)*is_seq(n-i1193+2,i1193-i1194))

        return(is_seq(0,n-i1193+2)*B_alpha_coef[n-i1193+2]*is_integer(n-i1193+2)*py_sum(sum_arg_38,0,i1193))

    def sum_arg_37(i198):
        # Child args for sum_arg_37
        return(B_theta_coef_cp[i198]*(n-i198+2)*X_coef_cp[n-i198+2])

    def sum_arg_36(i1207):
        # Child args for sum_arg_36
        def sum_arg_35(i1208):
            # Child args for sum_arg_35
            def sum_arg_34(i1664):
                # Child args for sum_arg_34
                return(diff(X_coef_cp[(-i1664)-i1208+i1207],True,1)*i1664*Z_coef_cp[i1664])

            return(is_seq(0,(-n)+i1208+i1207-2)*B_theta_coef_cp[(-n)+i1208+i1207-2]*is_integer((-n)+i1208+i1207-2)*is_seq((-n)+i1208+i1207-2,i1208)*py_sum(sum_arg_34,0,i1207-i1208))

        return(iota_coef[n-i1207+2]*py_sum(sum_arg_35,0,i1207))

    def sum_arg_33(i1197):
        # Child args for sum_arg_33
        def sum_arg_32(i1198):
            # Child args for sum_arg_32
            def sum_arg_31(i1648):
                # Child args for sum_arg_31
                return(diff(Z_coef_cp[(-i1648)-i1198+i1197],True,1)*i1648*X_coef_cp[i1648])

            return(is_seq(0,(-n)+i1198+i1197-2)*B_theta_coef_cp[(-n)+i1198+i1197-2]*is_integer((-n)+i1198+i1197-2)*is_seq((-n)+i1198+i1197-2,i1198)*py_sum(sum_arg_31,0,i1197-i1198))

        return(iota_coef[n-i1197+2]*py_sum(sum_arg_32,0,i1197))

    def sum_arg_30(i122):
        # Child args for sum_arg_30
        def sum_arg_29(i120):
            # Child args for sum_arg_29
            return(B_theta_coef_cp[i120]*X_coef_cp[n-i122-i120+2])

        return(i122*Z_coef_cp[i122]*py_sum(sum_arg_29,0,n-i122+2))

    def sum_arg_28(i118):
        # Child args for sum_arg_28
        def sum_arg_27(i116):
            # Child args for sum_arg_27
            return(B_psi_coef_cp[i116]*X_coef_cp[n-i118-i116])

        return(diff(Z_coef_cp[i118],True,1)*py_sum(sum_arg_27,0,n-i118))

    def sum_arg_26(i732):
        # Child args for sum_arg_26
        def sum_arg_25(i140):
            # Child args for sum_arg_25
            return(B_psi_coef_cp[i140]*diff(Y_coef_cp[n-i732-i140],False,1))

        return(diff(Z_coef_cp[i732],True,1)*py_sum(sum_arg_25,0,n-i732))

    def sum_arg_24(i726):
        # Child args for sum_arg_24
        def sum_arg_23(i136):
            # Child args for sum_arg_23
            return(B_psi_coef_cp[i136]*diff(Y_coef_cp[n-i726-i136],True,1))

        return(diff(Z_coef_cp[i726],False,1)*py_sum(sum_arg_23,0,n-i726))

    def sum_arg_22(i734):
        # Child args for sum_arg_22
        def sum_arg_21(i138):
            # Child args for sum_arg_21
            return(B_theta_coef_cp[i138]*(n-i734-i138+2)*Y_coef_cp[n-i734-i138+2])

        return(diff(Z_coef_cp[i734],False,1)*py_sum(sum_arg_21,0,n-i734+2))

    def sum_arg_20(i729):
        # Child args for sum_arg_20
        def sum_arg_19(i730):
            # Child args for sum_arg_19
            return(diff(Y_coef_cp[i730],True,1)*Z_coef_cp[(-n)-i730+2*i729-2]*((-n)-i730+2*i729-2)*is_seq(n-i729+2,i729-i730))

        return(is_seq(0,n-i729+2)*B_alpha_coef[n-i729+2]*is_integer(n-i729+2)*py_sum(sum_arg_19,0,i729))

    def sum_arg_18(i728):
        # Child args for sum_arg_18
        return(B_psi_coef_cp[i728]*diff(Y_coef_cp[n-i728],True,1))

    def sum_arg_17(i721):
        # Child args for sum_arg_17
        def sum_arg_16(i722):
            # Child args for sum_arg_16
            return(diff(Z_coef_cp[i722],True,1)*Y_coef_cp[(-n)-i722+2*i721-2]*((-n)-i722+2*i721-2)*is_seq(n-i721+2,i721-i722))

        return(is_seq(0,n-i721+2)*B_alpha_coef[n-i721+2]*is_integer(n-i721+2)*py_sum(sum_arg_16,0,i721))

    def sum_arg_15(i146):
        # Child args for sum_arg_15
        def sum_arg_14(i144):
            # Child args for sum_arg_14
            return(B_theta_coef_cp[i144]*(n-i146-i144+2)*Z_coef_cp[n-i146-i144+2])

        return(diff(Y_coef_cp[i146],False,1)*py_sum(sum_arg_14,0,n-i146+2))

    def sum_arg_13(i130):
        # Child args for sum_arg_13
        def sum_arg_12(i128):
            # Child args for sum_arg_12
            return(B_theta_coef_cp[i128]*X_coef_cp[n-i130-i128+2])

        return(i130*Y_coef_cp[i130]*py_sum(sum_arg_12,0,n-i130+2))

    def sum_arg_11(i126):
        # Child args for sum_arg_11
        def sum_arg_10(i124):
            # Child args for sum_arg_10
            return(B_psi_coef_cp[i124]*X_coef_cp[n-i126-i124])

        return(diff(Y_coef_cp[i126],True,1)*py_sum(sum_arg_10,0,n-i126))

    def sum_arg_9(i735):
        # Child args for sum_arg_9
        def sum_arg_8(i736):
            # Child args for sum_arg_8
            def sum_arg_7(i1191):
                # Child args for sum_arg_7
                return(i1191*Y_coef_cp[i1191]*diff(Z_coef_cp[(-i736)+i735-i1191],True,1))

            return(is_seq(0,(-n)+i736+i735-2)*B_theta_coef_cp[(-n)+i736+i735-2]*is_integer((-n)+i736+i735-2)*is_seq((-n)+i736+i735-2,i736)*py_sum(sum_arg_7,0,i735-i736))

        return(iota_coef[n-i735+2]*py_sum(sum_arg_8,0,i735))

    def sum_arg_6(i723):
        # Child args for sum_arg_6
        def sum_arg_5(i724):
            # Child args for sum_arg_5
            def sum_arg_4(i1175):
                # Child args for sum_arg_4
                return(i1175*Z_coef_cp[i1175]*diff(Y_coef_cp[(-i724)+i723-i1175],True,1))

            return(is_seq(0,(-n)+i724+i723-2)*B_theta_coef_cp[(-n)+i724+i723-2]*is_integer((-n)+i724+i723-2)*is_seq((-n)+i724+i723-2,i724)*py_sum(sum_arg_4,0,i723-i724))

        return(iota_coef[n-i723+2]*py_sum(sum_arg_5,0,i723))

    def sum_arg_3(i150):
        # Child args for sum_arg_3
        return(B_theta_coef_cp[i150]*(n-i150+2)*Y_coef_cp[n-i150+2])

    def sum_arg_2(i151):
        # Child args for sum_arg_2
        return(is_seq(0,n-i151)*diff(X_coef_cp[2*i151-n],True,1)*iota_coef[n-i151]*is_integer(n-i151)*is_seq(n-i151,i151))

    def sum_arg_1(i199):
        # Child args for sum_arg_1
        return(is_seq(0,n-i199)*diff(Y_coef_cp[2*i199-n],True,1)*iota_coef[n-i199]*is_integer(n-i199)*is_seq(n-i199,i199))


    out = (-Y_coef_cp[1]*((-(is_seq(0,n+2)*dl_p*is_integer(n+2)*py_sum_parallel(sum_arg_62,0,n+2)*tau_p)/2)+is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_60,0,n)*tau_p+is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_58,0,n)-is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_56,0,n)-(is_seq(0,n+2)*is_integer(n+2)*py_sum_parallel(sum_arg_54,0,n+2))/2+(is_seq(0,n+2)*is_integer(n+2)*py_sum_parallel(sum_arg_52,0,n+2))/2-(is_seq(0,n+2)*dl_p*kap_p*is_integer(n+2)*py_sum_parallel(sum_arg_50,0,n+2))/2-(is_seq(0,n+2)*dl_p*kap_p*is_integer(n+2)*py_sum_parallel(sum_arg_48,0,n+2))/2+is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_46,0,n)+is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_44,0,n)-py_sum_parallel(sum_arg_42,ceil(0.5*n)+1,floor(n)+2)/2-is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_40,0,n)+py_sum_parallel(sum_arg_39,ceil(0.5*n)+1,floor(n)+2)/2+(is_seq(0,n+2)*dl_p*is_integer(n+2)*py_sum_parallel(sum_arg_37,0,n+2))/2-py_sum_parallel(sum_arg_36,ceil(0.5*n)+1,floor(n)+2)/2+py_sum_parallel(sum_arg_33,ceil(0.5*n)+1,floor(n)+2)/2))-X_coef_cp[1]*((-(is_seq(0,n+2)*dl_p*is_integer(n+2)*py_sum_parallel(sum_arg_30,0,n+2)*tau_p)/2)+is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_28,0,n)*tau_p-py_sum_parallel(sum_arg_9,ceil(0.5*n)+1,floor(n)+2)/2+py_sum_parallel(sum_arg_6,ceil(0.5*n)+1,floor(n)+2)/2-(is_seq(0,n+2)*dl_p*is_integer(n+2)*py_sum_parallel(sum_arg_3,0,n+2))/2-is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_26,0,n)+is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_24,0,n)-(is_seq(0,n+2)*is_integer(n+2)*py_sum_parallel(sum_arg_22,0,n+2))/2-py_sum_parallel(sum_arg_20,ceil(0.5*n)+1,floor(n)+2)/2+is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_18,0,n)+py_sum_parallel(sum_arg_17,ceil(0.5*n)+1,floor(n)+2)/2+(is_seq(0,n+2)*is_integer(n+2)*py_sum_parallel(sum_arg_15,0,n+2))/2+(is_seq(0,n+2)*dl_p*kap_p*is_integer(n+2)*py_sum_parallel(sum_arg_13,0,n+2))/2-is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_11,0,n))+X_coef_cp[1]*(is_seq(0,n)*dl_p*Y_coef_cp[n]*is_integer(n)*tau_p+py_sum_parallel(sum_arg_2,ceil(0.5*n),floor(n))+is_seq(0,n)*is_integer(n)*diff(X_coef_cp[n],False,1)+is_seq(0,n)*dl_p*kap_p*Z_coef_cp[n]*is_integer(n))+Y_coef_cp[1]*((-is_seq(0,n)*dl_p*X_coef_cp[n]*is_integer(n)*tau_p)+py_sum_parallel(sum_arg_1,ceil(0.5*n),floor(n))+is_seq(0,n)*is_integer(n)*diff(Y_coef_cp[n],False,1))
    return(out)

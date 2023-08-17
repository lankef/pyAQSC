# Evaluates Zn+1. Requires X[..., n], Y[..., n], Z[..., n],
# B_theta_coef_cp[..., n], B_psi_coef_cp[..., n-1],
# B_alpha_coef [..., (n-1)/2 or (n-2)/2]
# iota_coef [..., (n-1)/2 or (n-2)/2]
# kap_p, dl_p, tau_p
from math import floor, ceil
from aqsc.math_utilities import *
def eval_Znp1_cp(n, X_coef_cp, Y_coef_cp, Z_coef_cp, \
    B_theta_coef_cp, B_psi_coef_cp, \
    B_alpha_coef, \
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

    def sum_arg_58(i122):
        # Child args for sum_arg_58
        def sum_arg_57(i120):
            # Child args for sum_arg_57
            return(B_theta_coef_cp[i120]*X_coef_cp[n-i122-i120+2])

        return(i122*Z_coef_cp[i122]*py_sum(sum_arg_57,0,n-i122+2))

    def sum_arg_56(i118):
        # Child args for sum_arg_56
        def sum_arg_55(i116):
            # Child args for sum_arg_55
            return(B_psi_coef_cp[i116]*X_coef_cp[n-i118-i116])

        return(diff(Z_coef_cp[i118],True,1)*py_sum(sum_arg_55,0,n-i118))

    def sum_arg_54(i736):
        # Child args for sum_arg_54
        def sum_arg_53(i184):
            # Child args for sum_arg_53
            return(B_psi_coef_cp[i184]*diff(X_coef_cp[n-i736-i184],True,1))

        return(diff(Z_coef_cp[i736],False,1)*py_sum(sum_arg_53,0,n-i736))

    def sum_arg_52(i730):
        # Child args for sum_arg_52
        def sum_arg_51(i188):
            # Child args for sum_arg_51
            return(B_psi_coef_cp[i188]*diff(X_coef_cp[n-i730-i188],False,1))

        return(diff(Z_coef_cp[i730],True,1)*py_sum(sum_arg_51,0,n-i730))

    def sum_arg_50(i262):
        # Child args for sum_arg_50
        def sum_arg_49(i140):
            # Child args for sum_arg_49
            return(B_psi_coef_cp[i140]*diff(Y_coef_cp[n-i262-i140],False,1))

        return(diff(Z_coef_cp[i262],True,1)*py_sum(sum_arg_49,0,n-i262))

    def sum_arg_48(i256):
        # Child args for sum_arg_48
        def sum_arg_47(i136):
            # Child args for sum_arg_47
            return(B_psi_coef_cp[i136]*diff(Y_coef_cp[n-i256-i136],True,1))

        return(diff(Z_coef_cp[i256],False,1)*py_sum(sum_arg_47,0,n-i256))

    def sum_arg_46(i733):
        # Child args for sum_arg_46
        def sum_arg_45(i734):
            # Child args for sum_arg_45
            return((((-i734)+2*i733-2)*diff(Z_coef_cp[i734],True,1)*X_coef_cp[(-n)-i734+2*i733-2]-diff(Z_coef_cp[i734],True,1)*X_coef_cp[(-n)-i734+2*i733-2]*n)*is_seq(n-i733+2,i733-i734))

        return(is_seq(0,n-i733+2)*B_alpha_coef[n-i733+2]*is_integer(n-i733+2)*py_sum(sum_arg_45,0,i733))

    def sum_arg_44(i732):
        # Child args for sum_arg_44
        return(B_psi_coef_cp[i732]*diff(X_coef_cp[n-i732],True,1))

    def sum_arg_43(i726):
        # Child args for sum_arg_43
        def sum_arg_42(i186):
            # Child args for sum_arg_42
            return((B_theta_coef_cp[i186]*n-B_theta_coef_cp[i186]*i726+(2-i186)*B_theta_coef_cp[i186])*X_coef_cp[n-i726-i186+2])

        return(diff(Z_coef_cp[i726],False,1)*py_sum(sum_arg_42,0,n-i726+2))

    def sum_arg_41(i723):
        # Child args for sum_arg_41
        def sum_arg_40(i724):
            # Child args for sum_arg_40
            return((((-i724)+2*i723-2)*diff(X_coef_cp[i724],True,1)*Z_coef_cp[(-n)-i724+2*i723-2]-diff(X_coef_cp[i724],True,1)*Z_coef_cp[(-n)-i724+2*i723-2]*n)*is_seq(n-i723+2,i723-i724))

        return(is_seq(0,n-i723+2)*B_alpha_coef[n-i723+2]*is_integer(n-i723+2)*py_sum(sum_arg_40,0,i723))

    def sum_arg_39(i264):
        # Child args for sum_arg_39
        def sum_arg_38(i138):
            # Child args for sum_arg_38
            return((B_theta_coef_cp[i138]*n-B_theta_coef_cp[i138]*i264+(2-i138)*B_theta_coef_cp[i138])*Y_coef_cp[n-i264-i138+2])

        return(diff(Z_coef_cp[i264],False,1)*py_sum(sum_arg_38,0,n-i264+2))

    def sum_arg_37(i259):
        # Child args for sum_arg_37
        def sum_arg_36(i260):
            # Child args for sum_arg_36
            return((((-i260)+2*i259-2)*diff(Y_coef_cp[i260],True,1)*Z_coef_cp[(-n)-i260+2*i259-2]-diff(Y_coef_cp[i260],True,1)*Z_coef_cp[(-n)-i260+2*i259-2]*n)*is_seq(n-i259+2,i259-i260))

        return(is_seq(0,n-i259+2)*B_alpha_coef[n-i259+2]*is_integer(n-i259+2)*py_sum(sum_arg_36,0,i259))

    def sum_arg_35(i258):
        # Child args for sum_arg_35
        return(B_psi_coef_cp[i258]*diff(Y_coef_cp[n-i258],True,1))

    def sum_arg_34(i251):
        # Child args for sum_arg_34
        def sum_arg_33(i252):
            # Child args for sum_arg_33
            return((((-i252)+2*i251-2)*diff(Z_coef_cp[i252],True,1)*Y_coef_cp[(-n)-i252+2*i251-2]-diff(Z_coef_cp[i252],True,1)*Y_coef_cp[(-n)-i252+2*i251-2]*n)*is_seq(n-i251+2,i251-i252))

        return(is_seq(0,n-i251+2)*B_alpha_coef[n-i251+2]*is_integer(n-i251+2)*py_sum(sum_arg_33,0,i251))

    def sum_arg_32(i194):
        # Child args for sum_arg_32
        def sum_arg_31(i192):
            # Child args for sum_arg_31
            return((B_theta_coef_cp[i192]*n-B_theta_coef_cp[i192]*i194+(2-i192)*B_theta_coef_cp[i192])*Z_coef_cp[n-i194-i192+2])

        return(diff(X_coef_cp[i194],False,1)*py_sum(sum_arg_31,0,n-i194+2))

    def sum_arg_30(i178):
        # Child args for sum_arg_30
        def sum_arg_29(i176):
            # Child args for sum_arg_29
            return(B_theta_coef_cp[i176]*X_coef_cp[n-i178-i176+2])

        return(i178*X_coef_cp[i178]*py_sum(sum_arg_29,0,n-i178+2))

    def sum_arg_28(i174):
        # Child args for sum_arg_28
        def sum_arg_27(i172):
            # Child args for sum_arg_27
            return(B_theta_coef_cp[i172]*Z_coef_cp[n-i174-i172+2])

        return(i174*Z_coef_cp[i174]*py_sum(sum_arg_27,0,n-i174+2))

    def sum_arg_26(i170):
        # Child args for sum_arg_26
        def sum_arg_25(i168):
            # Child args for sum_arg_25
            return(B_psi_coef_cp[i168]*X_coef_cp[n-i170-i168])

        return(diff(X_coef_cp[i170],True,1)*py_sum(sum_arg_25,0,n-i170))

    def sum_arg_24(i166):
        # Child args for sum_arg_24
        def sum_arg_23(i164):
            # Child args for sum_arg_23
            return(B_psi_coef_cp[i164]*Z_coef_cp[n-i166-i164])

        return(diff(Z_coef_cp[i166],True,1)*py_sum(sum_arg_23,0,n-i166))

    def sum_arg_22(i146):
        # Child args for sum_arg_22
        def sum_arg_21(i144):
            # Child args for sum_arg_21
            return((B_theta_coef_cp[i144]*n-B_theta_coef_cp[i144]*i146+(2-i144)*B_theta_coef_cp[i144])*Z_coef_cp[n-i146-i144+2])

        return(diff(Y_coef_cp[i146],False,1)*py_sum(sum_arg_21,0,n-i146+2))

    def sum_arg_20(i130):
        # Child args for sum_arg_20
        def sum_arg_19(i128):
            # Child args for sum_arg_19
            return(B_theta_coef_cp[i128]*X_coef_cp[n-i130-i128+2])

        return(i130*Y_coef_cp[i130]*py_sum(sum_arg_19,0,n-i130+2))

    def sum_arg_18(i126):
        # Child args for sum_arg_18
        def sum_arg_17(i124):
            # Child args for sum_arg_17
            return(B_psi_coef_cp[i124]*X_coef_cp[n-i126-i124])

        return(diff(Y_coef_cp[i126],True,1)*py_sum(sum_arg_17,0,n-i126))

    def sum_arg_16(i737):
        # Child args for sum_arg_16
        def sum_arg_15(i738):
            # Child args for sum_arg_15
            def sum_arg_14(i1193):
                # Child args for sum_arg_14
                return(i1193*Z_coef_cp[i1193]*diff(X_coef_cp[(-i738)+i737-i1193],True,1))

            return(is_seq(0,(-n)+i738+i737-2)*B_theta_coef_cp[(-n)+i738+i737-2]*is_integer((-n)+i738+i737-2)*is_seq((-n)+i738+i737-2,i738)*py_sum(sum_arg_14,0,i737-i738))

        return(iota_coef[n-i737+2]*py_sum(sum_arg_15,0,i737))

    def sum_arg_13(i727):
        # Child args for sum_arg_13
        def sum_arg_12(i728):
            # Child args for sum_arg_12
            def sum_arg_11(i1177):
                # Child args for sum_arg_11
                return(i1177*X_coef_cp[i1177]*diff(Z_coef_cp[(-i728)+i727-i1177],True,1))

            return(is_seq(0,(-n)+i728+i727-2)*B_theta_coef_cp[(-n)+i728+i727-2]*is_integer((-n)+i728+i727-2)*is_seq((-n)+i728+i727-2,i728)*py_sum(sum_arg_11,0,i727-i728))

        return(iota_coef[n-i727+2]*py_sum(sum_arg_12,0,i727))

    def sum_arg_10(i265):
        # Child args for sum_arg_10
        def sum_arg_9(i266):
            # Child args for sum_arg_9
            def sum_arg_8(i722):
                # Child args for sum_arg_8
                return(diff(Z_coef_cp[(-i722)-i266+i265],True,1)*i722*Y_coef_cp[i722])

            return(is_seq(0,(-n)+i266+i265-2)*B_theta_coef_cp[(-n)+i266+i265-2]*is_integer((-n)+i266+i265-2)*is_seq((-n)+i266+i265-2,i266)*py_sum(sum_arg_8,0,i265-i266))

        return(iota_coef[n-i265+2]*py_sum(sum_arg_9,0,i265))

    def sum_arg_7(i253):
        # Child args for sum_arg_7
        def sum_arg_6(i254):
            # Child args for sum_arg_6
            def sum_arg_5(i706):
                # Child args for sum_arg_5
                return(diff(Y_coef_cp[(-i706)-i254+i253],True,1)*i706*Z_coef_cp[i706])

            return(is_seq(0,(-n)+i254+i253-2)*B_theta_coef_cp[(-n)+i254+i253-2]*is_integer((-n)+i254+i253-2)*is_seq((-n)+i254+i253-2,i254)*py_sum(sum_arg_5,0,i253-i254))

        return(iota_coef[n-i253+2]*py_sum(sum_arg_6,0,i253))

    def sum_arg_4(i199):
        # Child args for sum_arg_4
        return(is_seq(0,n-i199)*diff(Y_coef_cp[2*i199-n],True,1)*iota_coef[n-i199]*is_integer(n-i199)*is_seq(n-i199,i199))

    def sum_arg_3(i198):
        # Child args for sum_arg_3
        return((B_theta_coef_cp[i198]*n+(2-i198)*B_theta_coef_cp[i198])*X_coef_cp[n-i198+2])

    def sum_arg_2(i151):
        # Child args for sum_arg_2
        return(is_seq(0,n-i151)*diff(X_coef_cp[2*i151-n],True,1)*iota_coef[n-i151]*is_integer(n-i151)*is_seq(n-i151,i151))

    def sum_arg_1(i150):
        # Child args for sum_arg_1
        return((B_theta_coef_cp[i150]*n+(2-i150)*B_theta_coef_cp[i150])*Y_coef_cp[n-i150+2])


    out = -((is_seq(0,n+2)*Y_coef_cp[1]*dl_p*is_integer(n+2)*py_sum_parallel(sum_arg_62,0,n+2)-2*is_seq(0,n)*Y_coef_cp[1]*dl_p*is_integer(n)*py_sum_parallel(sum_arg_60,0,n)+is_seq(0,n+2)*X_coef_cp[1]*dl_p*is_integer(n+2)*py_sum_parallel(sum_arg_58,0,n+2)-2*is_seq(0,n)*X_coef_cp[1]*dl_p*is_integer(n)*py_sum_parallel(sum_arg_56,0,n)+(2*is_seq(0,n)*X_coef_cp[1]*dl_p*Y_coef_cp[n]-2*is_seq(0,n)*Y_coef_cp[1]*dl_p*X_coef_cp[n])*is_integer(n))*tau_p-X_coef_cp[1]*py_sum_parallel(sum_arg_7,ceil(n/2)+1,floor(n)+2)+2*is_seq(0,n)*Y_coef_cp[1]*is_integer(n)*py_sum_parallel(sum_arg_54,0,n)-2*is_seq(0,n)*Y_coef_cp[1]*is_integer(n)*py_sum_parallel(sum_arg_52,0,n)+2*is_seq(0,n)*X_coef_cp[1]*is_integer(n)*py_sum_parallel(sum_arg_50,0,n)-2*is_seq(0,n)*X_coef_cp[1]*is_integer(n)*py_sum_parallel(sum_arg_48,0,n)+Y_coef_cp[1]*py_sum_parallel(sum_arg_46,ceil(n/2)+1,floor(n)+2)+2*is_seq(0,n)*Y_coef_cp[1]*dl_p*is_integer(n)*py_sum_parallel(sum_arg_44,0,n)-is_seq(0,n+2)*Y_coef_cp[1]*is_integer(n+2)*py_sum_parallel(sum_arg_43,0,n+2)-Y_coef_cp[1]*py_sum_parallel(sum_arg_41,ceil(n/2)+1,floor(n)+2)+2*Y_coef_cp[1]*py_sum_parallel(sum_arg_4,ceil(n/2),floor(n))+is_seq(0,n+2)*X_coef_cp[1]*is_integer(n+2)*py_sum_parallel(sum_arg_39,0,n+2)+X_coef_cp[1]*py_sum_parallel(sum_arg_37,ceil(n/2)+1,floor(n)+2)-2*is_seq(0,n)*X_coef_cp[1]*dl_p*is_integer(n)*py_sum_parallel(sum_arg_35,0,n)-X_coef_cp[1]*py_sum_parallel(sum_arg_34,ceil(n/2)+1,floor(n)+2)+is_seq(0,n+2)*Y_coef_cp[1]*is_integer(n+2)*py_sum_parallel(sum_arg_32,0,n+2)+is_seq(0,n+2)*Y_coef_cp[1]*dl_p*kap_p*is_integer(n+2)*py_sum_parallel(sum_arg_30,0,n+2)-is_seq(0,n+2)*Y_coef_cp[1]*dl_p*is_integer(n+2)*py_sum_parallel(sum_arg_3,0,n+2)+is_seq(0,n+2)*Y_coef_cp[1]*dl_p*kap_p*is_integer(n+2)*py_sum_parallel(sum_arg_28,0,n+2)-2*is_seq(0,n)*Y_coef_cp[1]*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_26,0,n)-2*is_seq(0,n)*Y_coef_cp[1]*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_24,0,n)-is_seq(0,n+2)*X_coef_cp[1]*is_integer(n+2)*py_sum_parallel(sum_arg_22,0,n+2)-is_seq(0,n+2)*X_coef_cp[1]*dl_p*kap_p*is_integer(n+2)*py_sum_parallel(sum_arg_20,0,n+2)+2*X_coef_cp[1]*py_sum_parallel(sum_arg_2,ceil(n/2),floor(n))+2*is_seq(0,n)*X_coef_cp[1]*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_18,0,n)+Y_coef_cp[1]*py_sum_parallel(sum_arg_16,ceil(n/2)+1,floor(n)+2)-Y_coef_cp[1]*py_sum_parallel(sum_arg_13,ceil(n/2)+1,floor(n)+2)+X_coef_cp[1]*py_sum_parallel(sum_arg_10,ceil(n/2)+1,floor(n)+2)+is_seq(0,n+2)*X_coef_cp[1]*dl_p*is_integer(n+2)*py_sum_parallel(sum_arg_1,0,n+2)+2*is_seq(0,n)*Y_coef_cp[1]*is_integer(n)*diff(Y_coef_cp[n],False,1)+2*is_seq(0,n)*X_coef_cp[1]*is_integer(n)*diff(X_coef_cp[n],False,1)+2*is_seq(0,n)*X_coef_cp[1]*dl_p*kap_p*Z_coef_cp[n]*is_integer(n))/(2*dl_p*n+2*dl_p)
    return(out)

# Evaluates Zn. See Zn+1 for requirements.
def eval_Zn_cp(n, X_coef_cp, Y_coef_cp, Z_coef_cp,
    B_theta_coef_cp, B_psi_coef_cp,
    B_alpha_coef,
    kap_p, dl_p, tau_p, iota_coef):

    return(eval_Znp1_cp(n-1, X_coef_cp, Y_coef_cp, Z_coef_cp,
        B_theta_coef_cp, B_psi_coef_cp,
        B_alpha_coef,
        kap_p, dl_p, tau_p, iota_coef))

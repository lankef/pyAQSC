# Evaluating the loop eq. Solves for different quantities
# when different masks are added.
# Uses Xn-1, Yn-1, Zn-1,  B_theta_n-2, Delta_n-1, B_psi_n-3, B_denom n-1,
# iota_coef (n-3)/2 or (n-2)/2, and B_alpha (n-1)/2 or (n-2)/2
# Must be evaluated with Z_coef_cp[n] = 0, p_perp_coef_cp[n] = 0
# B_psi_coef_cp[n-2] = 0, B_denom_coef_c[n] = 0 and B_theta_coef_cp[n] = 0
from math import floor, ceil
from aqsc.math_utilities import *
# from jax import jit
# from functools import partial
# @partial(jit, static_argnums=(0,))
def eval_loop(n, X_coef_cp, Y_coef_cp, Z_coef_cp, \
    B_theta_coef_cp, B_psi_coef_cp, B_alpha_coef, B_denom_coef_c, \
    p_perp_coef_cp, Delta_coef_cp, kap_p, dl_p, tau_p, iota_coef):
    def sum_arg_153(i288):
        # Child args for sum_arg_153
        return(X_coef_cp[i288]*diff(Y_coef_cp[n-i288],True,1))

    def sum_arg_152(i284):
        # Child args for sum_arg_152
        return(Y_coef_cp[i284]*diff(X_coef_cp[n-i284],True,1))

    def sum_arg_151(i256):
        # Child args for sum_arg_151
        return(i256*X_coef_cp[i256]*diff(Y_coef_cp[n-i256],True,1)+i256*diff(X_coef_cp[i256],True,1)*Y_coef_cp[n-i256])

    def sum_arg_150(i254):
        # Child args for sum_arg_150
        return((X_coef_cp[i254]*n-i254*X_coef_cp[i254])*diff(Y_coef_cp[n-i254],True,1)+(diff(X_coef_cp[i254],True,1)*n-i254*diff(X_coef_cp[i254],True,1))*Y_coef_cp[n-i254])

    def sum_arg_149(i288):
        # Child args for sum_arg_149
        return(X_coef_cp[i288]*diff(Y_coef_cp[n-i288],True,2)+diff(X_coef_cp[i288],True,1)*diff(Y_coef_cp[n-i288],True,1))

    def sum_arg_148(i288):
        # Child args for sum_arg_148
        return(X_coef_cp[i288]*diff(Y_coef_cp[n-i288],True,1,False,1)+diff(X_coef_cp[i288],False,1)*diff(Y_coef_cp[n-i288],True,1))

    def sum_arg_147(i288):
        # Child args for sum_arg_147
        return(X_coef_cp[i288]*diff(Y_coef_cp[n-i288],True,1))

    def sum_arg_146(i284):
        # Child args for sum_arg_146
        return(Y_coef_cp[i284]*diff(X_coef_cp[n-i284],True,2)+diff(Y_coef_cp[i284],True,1)*diff(X_coef_cp[n-i284],True,1))

    def sum_arg_145(i284):
        # Child args for sum_arg_145
        return(Y_coef_cp[i284]*diff(X_coef_cp[n-i284],True,1,False,1)+diff(Y_coef_cp[i284],False,1)*diff(X_coef_cp[n-i284],True,1))

    def sum_arg_144(i284):
        # Child args for sum_arg_144
        return(Y_coef_cp[i284]*diff(X_coef_cp[n-i284],True,1))

    def sum_arg_143(i256):
        # Child args for sum_arg_143
        return(i256*diff(X_coef_cp[i256],True,1)*diff(Y_coef_cp[n-i256],False,1)+i256*X_coef_cp[i256]*diff(Y_coef_cp[n-i256],True,1,False,1)+i256*diff(X_coef_cp[i256],False,1)*diff(Y_coef_cp[n-i256],True,1)+i256*diff(X_coef_cp[i256],True,1,False,1)*Y_coef_cp[n-i256])

    def sum_arg_142(i256):
        # Child args for sum_arg_142
        return(i256*X_coef_cp[i256]*diff(Y_coef_cp[n-i256],True,2)+2*i256*diff(X_coef_cp[i256],True,1)*diff(Y_coef_cp[n-i256],True,1)+i256*diff(X_coef_cp[i256],True,2)*Y_coef_cp[n-i256])

    def sum_arg_141(i256):
        # Child args for sum_arg_141
        return(i256*X_coef_cp[i256]*diff(Y_coef_cp[n-i256],True,1)+i256*diff(X_coef_cp[i256],True,1)*Y_coef_cp[n-i256])

    def sum_arg_140(i254):
        # Child args for sum_arg_140
        return((diff(X_coef_cp[i254],True,1)*n-i254*diff(X_coef_cp[i254],True,1))*diff(Y_coef_cp[n-i254],False,1)+(X_coef_cp[i254]*n-i254*X_coef_cp[i254])*diff(Y_coef_cp[n-i254],True,1,False,1)+(diff(X_coef_cp[i254],False,1)*n-i254*diff(X_coef_cp[i254],False,1))*diff(Y_coef_cp[n-i254],True,1)+(diff(X_coef_cp[i254],True,1,False,1)*n-i254*diff(X_coef_cp[i254],True,1,False,1))*Y_coef_cp[n-i254])

    def sum_arg_139(i254):
        # Child args for sum_arg_139
        return((X_coef_cp[i254]*n-i254*X_coef_cp[i254])*diff(Y_coef_cp[n-i254],True,2)+(2*diff(X_coef_cp[i254],True,1)*n-2*i254*diff(X_coef_cp[i254],True,1))*diff(Y_coef_cp[n-i254],True,1)+(diff(X_coef_cp[i254],True,2)*n-i254*diff(X_coef_cp[i254],True,2))*Y_coef_cp[n-i254])

    def sum_arg_138(i254):
        # Child args for sum_arg_138
        return((X_coef_cp[i254]*n-i254*X_coef_cp[i254])*diff(Y_coef_cp[n-i254],True,1)+(diff(X_coef_cp[i254],True,1)*n-i254*diff(X_coef_cp[i254],True,1))*Y_coef_cp[n-i254])

    def sum_arg_137(i348):
        # Child args for sum_arg_137
        def sum_arg_136(i228):
            # Child args for sum_arg_136
            def sum_arg_135(i226):
                # Child args for sum_arg_135
                return(B_denom_coef_c[i226]*B_denom_coef_c[n-i348-i228-i226])

            return(diff(p_perp_coef_cp[i228],False,1)*py_sum(sum_arg_135,0,n-i348-i228))

        return(B_theta_coef_cp[i348]*py_sum(sum_arg_136,0,n-i348))

    def sum_arg_134(i345):
        # Child args for sum_arg_134
        def sum_arg_133(i346):
            # Child args for sum_arg_133
            def sum_arg_132(i344):
                # Child args for sum_arg_132
                return(Delta_coef_cp[i344]*diff(B_theta_coef_cp[(-n)-i346+2*i345-i344],True,1)*is_seq(n-i345,(-i346)+i345-i344))

            return(B_denom_coef_c[i346]*py_sum(sum_arg_132,0,i345-i346))

        return(is_seq(0,n-i345)*iota_coef[n-i345]*is_integer(n-i345)*py_sum(sum_arg_133,0,i345))

    def sum_arg_131(i342):
        # Child args for sum_arg_131
        def sum_arg_130(i340):
            # Child args for sum_arg_130
            return(Delta_coef_cp[i340]*diff(B_theta_coef_cp[n-i342-i340],False,1))

        return(B_denom_coef_c[i342]*py_sum(sum_arg_130,0,n-i342))

    def sum_arg_129(i334):
        # Child args for sum_arg_129
        def sum_arg_127(i296):
            # Child args for sum_arg_127
            return(B_psi_coef_cp[i296]*diff(Delta_coef_cp[n-i334-i296-2],False,1))

        def sum_arg_128(i296):
            # Child args for sum_arg_128
            return(diff(B_psi_coef_cp[i296],True,1)*diff(Delta_coef_cp[n-i334-i296-2],False,1)+B_psi_coef_cp[i296]*diff(Delta_coef_cp[n-i334-i296-2],True,1,False,1))

        return(B_denom_coef_c[i334]*py_sum(sum_arg_128,0,n-i334-2)+diff(B_denom_coef_c[i334],True,1)*py_sum(sum_arg_127,0,n-i334-2))

    def sum_arg_126(i330):
        # Child args for sum_arg_126
        def sum_arg_124(i328):
            # Child args for sum_arg_124
            return(Delta_coef_cp[i328]*diff(B_psi_coef_cp[n-i330-i328-2],False,1))

        def sum_arg_125(i328):
            # Child args for sum_arg_125
            return(diff(Delta_coef_cp[i328],True,1)*diff(B_psi_coef_cp[n-i330-i328-2],False,1)+Delta_coef_cp[i328]*diff(B_psi_coef_cp[n-i330-i328-2],True,1,False,1))

        return(B_denom_coef_c[i330]*py_sum(sum_arg_125,0,n-i330-2)+diff(B_denom_coef_c[i330],True,1)*py_sum(sum_arg_124,0,n-i330-2))

    def sum_arg_123(i337):
        # Child args for sum_arg_123
        def sum_arg_122(i338):
            # Child args for sum_arg_122
            return(B_denom_coef_c[i338]*diff(B_theta_coef_cp[(-n)-i338+2*i337],True,1)*is_seq(n-i337,i337-i338))

        return(is_seq(0,n-i337)*iota_coef[n-i337]*is_integer(n-i337)*py_sum(sum_arg_122,0,i337))

    def sum_arg_121(i336):
        # Child args for sum_arg_121
        return(B_denom_coef_c[i336]*diff(B_theta_coef_cp[n-i336],False,1))

    def sum_arg_120(i325):
        # Child args for sum_arg_120
        def sum_arg_119(i326):
            # Child args for sum_arg_119
            def sum_arg_117(i298):
                # Child args for sum_arg_117
                return(B_psi_coef_cp[i298]*diff(Delta_coef_cp[(-n)-i326+2*i325-i298+2],True,1))

            def sum_arg_118(i298):
                # Child args for sum_arg_118
                return(B_psi_coef_cp[i298]*diff(Delta_coef_cp[(-n)-i326+2*i325-i298+2],True,2)+diff(B_psi_coef_cp[i298],True,1)*diff(Delta_coef_cp[(-n)-i326+2*i325-i298+2],True,1))

            return(is_seq(n-i325-2,i325-i326)*(B_denom_coef_c[i326]*py_sum(sum_arg_118,0,(-n)-i326+2*i325+2)+diff(B_denom_coef_c[i326],True,1)*py_sum(sum_arg_117,0,(-n)-i326+2*i325+2)))

        return(is_seq(0,n-i325-2)*iota_coef[n-i325-2]*is_integer(n-i325-2)*py_sum(sum_arg_119,0,i325))

    def sum_arg_116(i321):
        # Child args for sum_arg_116
        def sum_arg_115(i322):
            # Child args for sum_arg_115
            def sum_arg_113(i318):
                # Child args for sum_arg_113
                return(Delta_coef_cp[i318]*diff(B_psi_coef_cp[(-n)-i322+2*i321-i318+2],True,1))

            def sum_arg_114(i318):
                # Child args for sum_arg_114
                return(Delta_coef_cp[i318]*diff(B_psi_coef_cp[(-n)-i322+2*i321-i318+2],True,2)+diff(Delta_coef_cp[i318],True,1)*diff(B_psi_coef_cp[(-n)-i322+2*i321-i318+2],True,1))

            return(is_seq(n-i321-2,i321-i322)*(B_denom_coef_c[i322]*py_sum(sum_arg_114,0,(-n)-i322+2*i321+2)+diff(B_denom_coef_c[i322],True,1)*py_sum(sum_arg_113,0,(-n)-i322+2*i321+2)))

        return(is_seq(0,n-i321-2)*iota_coef[n-i321-2]*is_integer(n-i321-2)*py_sum(sum_arg_115,0,i321))

    def sum_arg_112(i311):
        # Child args for sum_arg_112
        def sum_arg_111(i312):
            # Child args for sum_arg_111
            return((diff(B_denom_coef_c[i312],True,1)*Delta_coef_cp[(-n)-i312+2*i311]+B_denom_coef_c[i312]*diff(Delta_coef_cp[(-n)-i312+2*i311],True,1))*is_seq(n-i311,i311-i312))

        return((is_seq(0,n-i311)*n-is_seq(0,n-i311)*i311)*B_alpha_coef[n-i311]*is_integer(n-i311)*py_sum(sum_arg_111,0,i311))

    def sum_arg_110(i305):
        # Child args for sum_arg_110
        def sum_arg_109(i306):
            # Child args for sum_arg_109
            return((B_denom_coef_c[i306]*diff(B_psi_coef_cp[(-n)-i306+2*i305+2],True,2)+diff(B_denom_coef_c[i306],True,1)*diff(B_psi_coef_cp[(-n)-i306+2*i305+2],True,1))*is_seq(n-i305-2,i305-i306))

        return(is_seq(0,n-i305-2)*iota_coef[n-i305-2]*is_integer(n-i305-2)*py_sum(sum_arg_109,0,i305))

    def sum_arg_108(i304):
        # Child args for sum_arg_108
        return(diff(B_denom_coef_c[i304],True,1)*diff(B_psi_coef_cp[n-i304-2],False,1)+B_denom_coef_c[i304]*diff(B_psi_coef_cp[n-i304-2],True,1,False,1))

    def sum_arg_107(i299):
        # Child args for sum_arg_107
        def sum_arg_106(i300):
            # Child args for sum_arg_106
            return((i300*diff(B_denom_coef_c[i300],True,1)*Delta_coef_cp[(-n)-i300+2*i299]+i300*B_denom_coef_c[i300]*diff(Delta_coef_cp[(-n)-i300+2*i299],True,1))*is_seq(n-i299,i299-i300))

        return(is_seq(0,n-i299)*B_alpha_coef[n-i299]*is_integer(n-i299)*py_sum(sum_arg_106,0,i299))

    def sum_arg_105(i291):
        # Child args for sum_arg_105
        def sum_arg_104(i292):
            # Child args for sum_arg_104
            def sum_arg_102(i246):
                # Child args for sum_arg_102
                return(B_denom_coef_c[i246]*B_denom_coef_c[(-n)-i292+2*i291-i246])

            def sum_arg_103(i246):
                # Child args for sum_arg_103
                return(diff(B_denom_coef_c[i246],True,1)*B_denom_coef_c[(-n)-i292+2*i291-i246]+B_denom_coef_c[i246]*diff(B_denom_coef_c[(-n)-i292+2*i291-i246],True,1))

            return(is_seq(n-i291,i291-i292)*(i292*p_perp_coef_cp[i292]*py_sum(sum_arg_103,0,(-n)-i292+2*i291)+i292*diff(p_perp_coef_cp[i292],True,1)*py_sum(sum_arg_102,0,(-n)-i292+2*i291)))

        return(is_seq(0,n-i291)*B_alpha_coef[n-i291]*is_integer(n-i291)*py_sum(sum_arg_104,0,i291))

    def sum_arg_101(i290):
        # Child args for sum_arg_101
        return(X_coef_cp[i290]*diff(Z_coef_cp[n-i290],True,2)+diff(X_coef_cp[i290],True,1)*diff(Z_coef_cp[n-i290],True,1))

    def sum_arg_100(i290):
        # Child args for sum_arg_100
        return(X_coef_cp[i290]*diff(Z_coef_cp[n-i290],True,1,False,1)+diff(X_coef_cp[i290],False,1)*diff(Z_coef_cp[n-i290],True,1))

    def sum_arg_99(i290):
        # Child args for sum_arg_99
        return(X_coef_cp[i290]*diff(Z_coef_cp[n-i290],True,1))

    def sum_arg_98(i286):
        # Child args for sum_arg_98
        return(Z_coef_cp[i286]*diff(X_coef_cp[n-i286],True,2)+diff(Z_coef_cp[i286],True,1)*diff(X_coef_cp[n-i286],True,1))

    def sum_arg_97(i286):
        # Child args for sum_arg_97
        return(Z_coef_cp[i286]*diff(X_coef_cp[n-i286],True,1,False,1)+diff(Z_coef_cp[i286],False,1)*diff(X_coef_cp[n-i286],True,1))

    def sum_arg_96(i286):
        # Child args for sum_arg_96
        return(Z_coef_cp[i286]*diff(X_coef_cp[n-i286],True,1))

    def sum_arg_95(i281):
        # Child args for sum_arg_95
        def sum_arg_94(i282):
            # Child args for sum_arg_94
            return((diff(Z_coef_cp[i282],True,1)*diff(Z_coef_cp[(-n)-i282+2*i281],True,2)+diff(Z_coef_cp[i282],True,2)*diff(Z_coef_cp[(-n)-i282+2*i281],True,1))*is_seq(n-i281,i281-i282))

        return(is_seq(0,n-i281)*iota_coef[n-i281]*is_integer(n-i281)*py_sum(sum_arg_94,0,i281))

    def sum_arg_93(i281):
        # Child args for sum_arg_93
        def sum_arg_92(i282):
            # Child args for sum_arg_92
            return((diff(Z_coef_cp[i282],True,1)*diff(Z_coef_cp[(-n)-i282+2*i281],True,1,False,1)+diff(Z_coef_cp[i282],True,1,False,1)*diff(Z_coef_cp[(-n)-i282+2*i281],True,1))*is_seq(n-i281,i281-i282))

        return(is_seq(0,n-i281)*iota_coef[n-i281]*is_integer(n-i281)*py_sum(sum_arg_92,0,i281))

    def sum_arg_91(i281):
        # Child args for sum_arg_91
        def sum_arg_90(i282):
            # Child args for sum_arg_90
            return(diff(Z_coef_cp[i282],True,1)*diff(Z_coef_cp[(-n)-i282+2*i281],True,1)*is_seq(n-i281,i281-i282))

        return(is_seq(0,n-i281)*iota_coef[n-i281]*is_integer(n-i281)*py_sum(sum_arg_90,0,i281))

    def sum_arg_89(i280):
        # Child args for sum_arg_89
        return(diff(Z_coef_cp[i280],True,1)*diff(Z_coef_cp[n-i280],False,2)+diff(Z_coef_cp[i280],True,1,False,1)*diff(Z_coef_cp[n-i280],False,1))

    def sum_arg_88(i280):
        # Child args for sum_arg_88
        return(diff(Z_coef_cp[i280],True,2)*diff(Z_coef_cp[n-i280],False,1)+diff(Z_coef_cp[i280],True,1)*diff(Z_coef_cp[n-i280],True,1,False,1))

    def sum_arg_87(i280):
        # Child args for sum_arg_87
        return(diff(Z_coef_cp[i280],True,1)*diff(Z_coef_cp[n-i280],False,1))

    def sum_arg_86(i277):
        # Child args for sum_arg_86
        def sum_arg_85(i278):
            # Child args for sum_arg_85
            return((diff(Y_coef_cp[i278],True,1)*diff(Y_coef_cp[(-n)-i278+2*i277],True,2)+diff(Y_coef_cp[i278],True,2)*diff(Y_coef_cp[(-n)-i278+2*i277],True,1))*is_seq(n-i277,i277-i278))

        return(is_seq(0,n-i277)*iota_coef[n-i277]*is_integer(n-i277)*py_sum(sum_arg_85,0,i277))

    def sum_arg_84(i277):
        # Child args for sum_arg_84
        def sum_arg_83(i278):
            # Child args for sum_arg_83
            return((diff(Y_coef_cp[i278],True,1)*diff(Y_coef_cp[(-n)-i278+2*i277],True,1,False,1)+diff(Y_coef_cp[i278],True,1,False,1)*diff(Y_coef_cp[(-n)-i278+2*i277],True,1))*is_seq(n-i277,i277-i278))

        return(is_seq(0,n-i277)*iota_coef[n-i277]*is_integer(n-i277)*py_sum(sum_arg_83,0,i277))

    def sum_arg_82(i277):
        # Child args for sum_arg_82
        def sum_arg_81(i278):
            # Child args for sum_arg_81
            return(diff(Y_coef_cp[i278],True,1)*diff(Y_coef_cp[(-n)-i278+2*i277],True,1)*is_seq(n-i277,i277-i278))

        return(is_seq(0,n-i277)*iota_coef[n-i277]*is_integer(n-i277)*py_sum(sum_arg_81,0,i277))

    def sum_arg_80(i276):
        # Child args for sum_arg_80
        return(diff(Y_coef_cp[i276],True,1)*diff(Y_coef_cp[n-i276],False,2)+diff(Y_coef_cp[i276],True,1,False,1)*diff(Y_coef_cp[n-i276],False,1))

    def sum_arg_79(i276):
        # Child args for sum_arg_79
        return(diff(Y_coef_cp[i276],True,2)*diff(Y_coef_cp[n-i276],False,1)+diff(Y_coef_cp[i276],True,1)*diff(Y_coef_cp[n-i276],True,1,False,1))

    def sum_arg_78(i276):
        # Child args for sum_arg_78
        return(diff(Y_coef_cp[i276],True,1)*diff(Y_coef_cp[n-i276],False,1))

    def sum_arg_77(i273):
        # Child args for sum_arg_77
        def sum_arg_76(i274):
            # Child args for sum_arg_76
            return((diff(X_coef_cp[i274],True,1)*diff(X_coef_cp[(-n)-i274+2*i273],True,2)+diff(X_coef_cp[i274],True,2)*diff(X_coef_cp[(-n)-i274+2*i273],True,1))*is_seq(n-i273,i273-i274))

        return(is_seq(0,n-i273)*iota_coef[n-i273]*is_integer(n-i273)*py_sum(sum_arg_76,0,i273))

    def sum_arg_75(i273):
        # Child args for sum_arg_75
        def sum_arg_74(i274):
            # Child args for sum_arg_74
            return((diff(X_coef_cp[i274],True,1)*diff(X_coef_cp[(-n)-i274+2*i273],True,1,False,1)+diff(X_coef_cp[i274],True,1,False,1)*diff(X_coef_cp[(-n)-i274+2*i273],True,1))*is_seq(n-i273,i273-i274))

        return(is_seq(0,n-i273)*iota_coef[n-i273]*is_integer(n-i273)*py_sum(sum_arg_74,0,i273))

    def sum_arg_73(i273):
        # Child args for sum_arg_73
        def sum_arg_72(i274):
            # Child args for sum_arg_72
            return(diff(X_coef_cp[i274],True,1)*diff(X_coef_cp[(-n)-i274+2*i273],True,1)*is_seq(n-i273,i273-i274))

        return(is_seq(0,n-i273)*iota_coef[n-i273]*is_integer(n-i273)*py_sum(sum_arg_72,0,i273))

    def sum_arg_71(i272):
        # Child args for sum_arg_71
        return(diff(X_coef_cp[i272],True,1)*diff(X_coef_cp[n-i272],False,2)+diff(X_coef_cp[i272],True,1,False,1)*diff(X_coef_cp[n-i272],False,1))

    def sum_arg_70(i272):
        # Child args for sum_arg_70
        return(diff(X_coef_cp[i272],True,2)*diff(X_coef_cp[n-i272],False,1)+diff(X_coef_cp[i272],True,1)*diff(X_coef_cp[n-i272],True,1,False,1))

    def sum_arg_69(i272):
        # Child args for sum_arg_69
        return(diff(X_coef_cp[i272],True,1)*diff(X_coef_cp[n-i272],False,1))

    def sum_arg_68(i269):
        # Child args for sum_arg_68
        def sum_arg_67(i270):
            # Child args for sum_arg_67
            return((i270*Z_coef_cp[i270]*diff(Z_coef_cp[(-n)-i270+2*i269],True,3)+2*i270*diff(Z_coef_cp[i270],True,1)*diff(Z_coef_cp[(-n)-i270+2*i269],True,2)+i270*diff(Z_coef_cp[i270],True,2)*diff(Z_coef_cp[(-n)-i270+2*i269],True,1))*is_seq(n-i269,i269-i270))

        return(is_seq(0,n-i269)*iota_coef[n-i269]*is_integer(n-i269)*py_sum(sum_arg_67,0,i269))

    def sum_arg_66(i269):
        # Child args for sum_arg_66
        def sum_arg_65(i270):
            # Child args for sum_arg_65
            return((i270*Z_coef_cp[i270]*diff(Z_coef_cp[(-n)-i270+2*i269],True,2,False,1)+i270*diff(Z_coef_cp[i270],False,1)*diff(Z_coef_cp[(-n)-i270+2*i269],True,2)+i270*diff(Z_coef_cp[i270],True,1)*diff(Z_coef_cp[(-n)-i270+2*i269],True,1,False,1)+i270*diff(Z_coef_cp[i270],True,1,False,1)*diff(Z_coef_cp[(-n)-i270+2*i269],True,1))*is_seq(n-i269,i269-i270))

        return(is_seq(0,n-i269)*iota_coef[n-i269]*is_integer(n-i269)*py_sum(sum_arg_65,0,i269))

    def sum_arg_64(i269):
        # Child args for sum_arg_64
        def sum_arg_63(i270):
            # Child args for sum_arg_63
            return((i270*Z_coef_cp[i270]*diff(Z_coef_cp[(-n)-i270+2*i269],True,2)+i270*diff(Z_coef_cp[i270],True,1)*diff(Z_coef_cp[(-n)-i270+2*i269],True,1))*is_seq(n-i269,i269-i270))

        return(is_seq(0,n-i269)*iota_coef[n-i269]*is_integer(n-i269)*py_sum(sum_arg_63,0,i269))

    def sum_arg_62(i267):
        # Child args for sum_arg_62
        def sum_arg_61(i268):
            # Child args for sum_arg_61
            return((i268*Y_coef_cp[i268]*diff(Y_coef_cp[(-n)-i268+2*i267],True,3)+2*i268*diff(Y_coef_cp[i268],True,1)*diff(Y_coef_cp[(-n)-i268+2*i267],True,2)+i268*diff(Y_coef_cp[i268],True,2)*diff(Y_coef_cp[(-n)-i268+2*i267],True,1))*is_seq(n-i267,i267-i268))

        return(is_seq(0,n-i267)*iota_coef[n-i267]*is_integer(n-i267)*py_sum(sum_arg_61,0,i267))

    def sum_arg_60(i267):
        # Child args for sum_arg_60
        def sum_arg_59(i268):
            # Child args for sum_arg_59
            return((i268*Y_coef_cp[i268]*diff(Y_coef_cp[(-n)-i268+2*i267],True,2,False,1)+i268*diff(Y_coef_cp[i268],False,1)*diff(Y_coef_cp[(-n)-i268+2*i267],True,2)+i268*diff(Y_coef_cp[i268],True,1)*diff(Y_coef_cp[(-n)-i268+2*i267],True,1,False,1)+i268*diff(Y_coef_cp[i268],True,1,False,1)*diff(Y_coef_cp[(-n)-i268+2*i267],True,1))*is_seq(n-i267,i267-i268))

        return(is_seq(0,n-i267)*iota_coef[n-i267]*is_integer(n-i267)*py_sum(sum_arg_59,0,i267))

    def sum_arg_58(i267):
        # Child args for sum_arg_58
        def sum_arg_57(i268):
            # Child args for sum_arg_57
            return((i268*Y_coef_cp[i268]*diff(Y_coef_cp[(-n)-i268+2*i267],True,2)+i268*diff(Y_coef_cp[i268],True,1)*diff(Y_coef_cp[(-n)-i268+2*i267],True,1))*is_seq(n-i267,i267-i268))

        return(is_seq(0,n-i267)*iota_coef[n-i267]*is_integer(n-i267)*py_sum(sum_arg_57,0,i267))

    def sum_arg_56(i265):
        # Child args for sum_arg_56
        def sum_arg_55(i266):
            # Child args for sum_arg_55
            return((i266*X_coef_cp[i266]*diff(X_coef_cp[(-n)-i266+2*i265],True,3)+2*i266*diff(X_coef_cp[i266],True,1)*diff(X_coef_cp[(-n)-i266+2*i265],True,2)+i266*diff(X_coef_cp[i266],True,2)*diff(X_coef_cp[(-n)-i266+2*i265],True,1))*is_seq(n-i265,i265-i266))

        return(is_seq(0,n-i265)*iota_coef[n-i265]*is_integer(n-i265)*py_sum(sum_arg_55,0,i265))

    def sum_arg_54(i265):
        # Child args for sum_arg_54
        def sum_arg_53(i266):
            # Child args for sum_arg_53
            return((i266*X_coef_cp[i266]*diff(X_coef_cp[(-n)-i266+2*i265],True,2,False,1)+i266*diff(X_coef_cp[i266],False,1)*diff(X_coef_cp[(-n)-i266+2*i265],True,2)+i266*diff(X_coef_cp[i266],True,1)*diff(X_coef_cp[(-n)-i266+2*i265],True,1,False,1)+i266*diff(X_coef_cp[i266],True,1,False,1)*diff(X_coef_cp[(-n)-i266+2*i265],True,1))*is_seq(n-i265,i265-i266))

        return(is_seq(0,n-i265)*iota_coef[n-i265]*is_integer(n-i265)*py_sum(sum_arg_53,0,i265))

    def sum_arg_52(i265):
        # Child args for sum_arg_52
        def sum_arg_51(i266):
            # Child args for sum_arg_51
            return((i266*X_coef_cp[i266]*diff(X_coef_cp[(-n)-i266+2*i265],True,2)+i266*diff(X_coef_cp[i266],True,1)*diff(X_coef_cp[(-n)-i266+2*i265],True,1))*is_seq(n-i265,i265-i266))

        return(is_seq(0,n-i265)*iota_coef[n-i265]*is_integer(n-i265)*py_sum(sum_arg_51,0,i265))

    def sum_arg_50(i264):
        # Child args for sum_arg_50
        return(i264*diff(Z_coef_cp[i264],True,1)*diff(Z_coef_cp[n-i264],False,2)+i264*diff(Z_coef_cp[i264],True,1,False,1)*diff(Z_coef_cp[n-i264],False,1)+i264*Z_coef_cp[i264]*diff(Z_coef_cp[n-i264],True,1,False,2)+i264*diff(Z_coef_cp[i264],False,1)*diff(Z_coef_cp[n-i264],True,1,False,1))

    def sum_arg_49(i264):
        # Child args for sum_arg_49
        return(i264*diff(Z_coef_cp[i264],True,2)*diff(Z_coef_cp[n-i264],False,1)+i264*Z_coef_cp[i264]*diff(Z_coef_cp[n-i264],True,2,False,1)+2*i264*diff(Z_coef_cp[i264],True,1)*diff(Z_coef_cp[n-i264],True,1,False,1))

    def sum_arg_48(i264):
        # Child args for sum_arg_48
        return(i264*diff(Z_coef_cp[i264],True,1)*diff(Z_coef_cp[n-i264],False,1)+i264*Z_coef_cp[i264]*diff(Z_coef_cp[n-i264],True,1,False,1))

    def sum_arg_47(i262):
        # Child args for sum_arg_47
        return(i262*diff(Y_coef_cp[i262],True,1)*diff(Y_coef_cp[n-i262],False,2)+i262*diff(Y_coef_cp[i262],True,1,False,1)*diff(Y_coef_cp[n-i262],False,1)+i262*Y_coef_cp[i262]*diff(Y_coef_cp[n-i262],True,1,False,2)+i262*diff(Y_coef_cp[i262],False,1)*diff(Y_coef_cp[n-i262],True,1,False,1))

    def sum_arg_46(i262):
        # Child args for sum_arg_46
        return(i262*diff(Y_coef_cp[i262],True,2)*diff(Y_coef_cp[n-i262],False,1)+i262*Y_coef_cp[i262]*diff(Y_coef_cp[n-i262],True,2,False,1)+2*i262*diff(Y_coef_cp[i262],True,1)*diff(Y_coef_cp[n-i262],True,1,False,1))

    def sum_arg_45(i262):
        # Child args for sum_arg_45
        return(i262*diff(Y_coef_cp[i262],True,1)*diff(Y_coef_cp[n-i262],False,1)+i262*Y_coef_cp[i262]*diff(Y_coef_cp[n-i262],True,1,False,1))

    def sum_arg_44(i260):
        # Child args for sum_arg_44
        return(i260*diff(X_coef_cp[i260],True,1)*diff(X_coef_cp[n-i260],False,2)+i260*diff(X_coef_cp[i260],True,1,False,1)*diff(X_coef_cp[n-i260],False,1)+i260*X_coef_cp[i260]*diff(X_coef_cp[n-i260],True,1,False,2)+i260*diff(X_coef_cp[i260],False,1)*diff(X_coef_cp[n-i260],True,1,False,1))

    def sum_arg_43(i260):
        # Child args for sum_arg_43
        return(i260*diff(X_coef_cp[i260],True,2)*diff(X_coef_cp[n-i260],False,1)+i260*X_coef_cp[i260]*diff(X_coef_cp[n-i260],True,2,False,1)+2*i260*diff(X_coef_cp[i260],True,1)*diff(X_coef_cp[n-i260],True,1,False,1))

    def sum_arg_42(i260):
        # Child args for sum_arg_42
        return(i260*diff(X_coef_cp[i260],True,1)*diff(X_coef_cp[n-i260],False,1)+i260*X_coef_cp[i260]*diff(X_coef_cp[n-i260],True,1,False,1))

    def sum_arg_41(i258):
        # Child args for sum_arg_41
        return(i258*diff(X_coef_cp[i258],True,1)*diff(Z_coef_cp[n-i258],False,1)+i258*X_coef_cp[i258]*diff(Z_coef_cp[n-i258],True,1,False,1)+i258*diff(X_coef_cp[i258],False,1)*diff(Z_coef_cp[n-i258],True,1)+i258*diff(X_coef_cp[i258],True,1,False,1)*Z_coef_cp[n-i258])

    def sum_arg_40(i258):
        # Child args for sum_arg_40
        return(i258*X_coef_cp[i258]*diff(Z_coef_cp[n-i258],True,2)+2*i258*diff(X_coef_cp[i258],True,1)*diff(Z_coef_cp[n-i258],True,1)+i258*diff(X_coef_cp[i258],True,2)*Z_coef_cp[n-i258])

    def sum_arg_39(i258):
        # Child args for sum_arg_39
        return(i258*X_coef_cp[i258]*diff(Z_coef_cp[n-i258],True,1)+i258*diff(X_coef_cp[i258],True,1)*Z_coef_cp[n-i258])

    def sum_arg_38(i252):
        # Child args for sum_arg_38
        return((diff(X_coef_cp[i252],True,1)*n-i252*diff(X_coef_cp[i252],True,1))*diff(Z_coef_cp[n-i252],False,1)+(X_coef_cp[i252]*n-i252*X_coef_cp[i252])*diff(Z_coef_cp[n-i252],True,1,False,1)+(diff(X_coef_cp[i252],False,1)*n-i252*diff(X_coef_cp[i252],False,1))*diff(Z_coef_cp[n-i252],True,1)+(diff(X_coef_cp[i252],True,1,False,1)*n-i252*diff(X_coef_cp[i252],True,1,False,1))*Z_coef_cp[n-i252])

    def sum_arg_37(i252):
        # Child args for sum_arg_37
        return((X_coef_cp[i252]*n-i252*X_coef_cp[i252])*diff(Z_coef_cp[n-i252],True,2)+(2*diff(X_coef_cp[i252],True,1)*n-2*i252*diff(X_coef_cp[i252],True,1))*diff(Z_coef_cp[n-i252],True,1)+(diff(X_coef_cp[i252],True,2)*n-i252*diff(X_coef_cp[i252],True,2))*Z_coef_cp[n-i252])

    def sum_arg_36(i252):
        # Child args for sum_arg_36
        return((X_coef_cp[i252]*n-i252*X_coef_cp[i252])*diff(Z_coef_cp[n-i252],True,1)+(diff(X_coef_cp[i252],True,1)*n-i252*diff(X_coef_cp[i252],True,1))*Z_coef_cp[n-i252])

    def sum_arg_35(i209):
        # Child args for sum_arg_35
        def sum_arg_34(i210):
            # Child args for sum_arg_34
            return((diff(B_theta_coef_cp[i210],True,1)*B_denom_coef_c[(-n)-i210+2*i209]+B_theta_coef_cp[i210]*diff(B_denom_coef_c[(-n)-i210+2*i209],True,1))*is_seq(n-i209,i209-i210))

        return(is_seq(0,n-i209)*B_alpha_coef[n-i209]*is_integer(n-i209)*py_sum(sum_arg_34,0,i209))

    def sum_arg_33(i209):
        # Child args for sum_arg_33
        def sum_arg_32(i210):
            # Child args for sum_arg_32
            return(diff(B_theta_coef_cp[i210],False,1)*B_denom_coef_c[(-n)-i210+2*i209]*is_seq(n-i209,i209-i210))

        return(is_seq(0,n-i209)*B_alpha_coef[n-i209]*is_integer(n-i209)*py_sum(sum_arg_32,0,i209))

    def sum_arg_31(i209):
        # Child args for sum_arg_31
        def sum_arg_30(i210):
            # Child args for sum_arg_30
            return(B_theta_coef_cp[i210]*B_denom_coef_c[(-n)-i210+2*i209]*is_seq(n-i209,i209-i210))

        return(is_seq(0,n-i209)*B_alpha_coef[n-i209]*is_integer(n-i209)*py_sum(sum_arg_30,0,i209))

    def sum_arg_29(i201):
        # Child args for sum_arg_29
        def sum_arg_28(i202):
            # Child args for sum_arg_28
            return((diff(B_psi_coef_cp[i202],True,2)*B_denom_coef_c[(-n)-i202+2*i201+2]+B_psi_coef_cp[i202]*diff(B_denom_coef_c[(-n)-i202+2*i201+2],True,2)+2*diff(B_psi_coef_cp[i202],True,1)*diff(B_denom_coef_c[(-n)-i202+2*i201+2],True,1))*is_seq(n-i201-2,i201-i202))

        return(is_seq(0,n-i201-2)*B_alpha_coef[n-i201-2]*is_integer(n-i201-2)*py_sum(sum_arg_28,0,i201))

    def sum_arg_27(i201):
        # Child args for sum_arg_27
        def sum_arg_26(i202):
            # Child args for sum_arg_26
            return((diff(B_psi_coef_cp[i202],True,1,False,1)*B_denom_coef_c[(-n)-i202+2*i201+2]+diff(B_psi_coef_cp[i202],False,1)*diff(B_denom_coef_c[(-n)-i202+2*i201+2],True,1))*is_seq(n-i201-2,i201-i202))

        return(is_seq(0,n-i201-2)*B_alpha_coef[n-i201-2]*is_integer(n-i201-2)*py_sum(sum_arg_26,0,i201))

    def sum_arg_25(i201):
        # Child args for sum_arg_25
        def sum_arg_24(i202):
            # Child args for sum_arg_24
            return((diff(B_psi_coef_cp[i202],True,1)*B_denom_coef_c[(-n)-i202+2*i201+2]+B_psi_coef_cp[i202]*diff(B_denom_coef_c[(-n)-i202+2*i201+2],True,1))*is_seq(n-i201-2,i201-i202))

        return(is_seq(0,n-i201-2)*B_alpha_coef[n-i201-2]*is_integer(n-i201-2)*py_sum(sum_arg_24,0,i201))

    def sum_arg_23(i355):
        # Child args for sum_arg_23
        def sum_arg_22(i356):
            # Child args for sum_arg_22
            def sum_arg_21(i804):
                # Child args for sum_arg_21
                return(diff(B_denom_coef_c[(-i804)-i356+i355],True,1)*Delta_coef_cp[i804])

            return(is_seq(0,(-n)+i356+i355)*B_theta_coef_cp[(-n)+i356+i355]*is_integer((-n)+i356+i355)*is_seq((-n)+i356+i355,i356)*py_sum(sum_arg_21,0,i355-i356))

        return(iota_coef[n-i355]*py_sum(sum_arg_22,0,i355))

    def sum_arg_20(i353):
        # Child args for sum_arg_20
        def sum_arg_19(i238):
            # Child args for sum_arg_19
            return(Delta_coef_cp[i238]*diff(B_denom_coef_c[(-n)+2*i353-i238],True,1))

        return(is_seq(0,n-i353)*B_alpha_coef[n-i353]*is_integer(n-i353)*is_seq(n-i353,i353)*py_sum(sum_arg_19,0,2*i353-n))

    def sum_arg_18(i351):
        # Child args for sum_arg_18
        def sum_arg_17(i352):
            # Child args for sum_arg_17
            def sum_arg_16(i765):
                # Child args for sum_arg_16
                def sum_arg_15(i788):
                    # Child args for sum_arg_15
                    return(B_denom_coef_c[(-i788)-i765-i352+i351]*B_denom_coef_c[i788])

                return(diff(p_perp_coef_cp[i765],True,1)*py_sum(sum_arg_15,0,(-i765)-i352+i351))

            return(is_seq(0,(-n)+i352+i351)*B_theta_coef_cp[(-n)+i352+i351]*is_integer((-n)+i352+i351)*is_seq((-n)+i352+i351,i352)*py_sum(sum_arg_16,0,i351-i352))

        return(iota_coef[n-i351]*py_sum(sum_arg_17,0,i351))

    def sum_arg_14(i349):
        # Child args for sum_arg_14
        def sum_arg_13(i234):
            # Child args for sum_arg_13
            def sum_arg_12(i230):
                # Child args for sum_arg_12
                return(B_denom_coef_c[i230]*B_denom_coef_c[(-n)+2*i349-i234-i230])

            return(diff(p_perp_coef_cp[i234],True,1)*py_sum(sum_arg_12,0,(-n)+2*i349-i234))

        return(is_seq(0,n-i349)*B_alpha_coef[n-i349]*is_integer(n-i349)*is_seq(n-i349,i349)*py_sum(sum_arg_13,0,2*i349-n))

    def sum_arg_11(i331):
        # Child args for sum_arg_11
        def sum_arg_9(i332):
            # Child args for sum_arg_9
            return((is_seq(0,(-n)-i332+2*i331)*diff(B_denom_coef_c[i332],True,1)*B_theta_coef_cp[(-n)-i332+2*i331]+is_seq(0,(-n)-i332+2*i331)*B_denom_coef_c[i332]*diff(B_theta_coef_cp[(-n)-i332+2*i331],True,1))*is_integer((-n)-i332+2*i331)*is_seq((-n)-i332+2*i331,i331-i332))

        def sum_arg_10(i332):
            # Child args for sum_arg_10
            return((is_seq(0,(-n)-i332+2*i331)*diff(B_denom_coef_c[i332],True,1)*B_theta_coef_cp[(-n)-i332+2*i331]+is_seq(0,(-n)-i332+2*i331)*B_denom_coef_c[i332]*diff(B_theta_coef_cp[(-n)-i332+2*i331],True,1))*is_integer((-n)-i332+2*i331)*is_seq((-n)-i332+2*i331,i331-i332))

        return(iota_coef[n-i331]*(n*py_sum(sum_arg_10,0,i331)-i331*py_sum(sum_arg_9,0,i331)))

    def sum_arg_8(i315):
        # Child args for sum_arg_8
        def sum_arg_4(i316):
            # Child args for sum_arg_4
            def sum_arg_2(i314):
                # Child args for sum_arg_2
                return(is_seq(0,(-n)-i316+2*i315-i314)*Delta_coef_cp[i314]*B_theta_coef_cp[(-n)-i316+2*i315-i314]*is_integer((-n)-i316+2*i315-i314)*is_seq((-n)-i316+2*i315-i314,(-i316)+i315-i314))

            def sum_arg_3(i314):
                # Child args for sum_arg_3
                return((is_seq(0,(-n)-i316+2*i315-i314)*diff(Delta_coef_cp[i314],True,1)*B_theta_coef_cp[(-n)-i316+2*i315-i314]+is_seq(0,(-n)-i316+2*i315-i314)*Delta_coef_cp[i314]*diff(B_theta_coef_cp[(-n)-i316+2*i315-i314],True,1))*is_integer((-n)-i316+2*i315-i314)*is_seq((-n)-i316+2*i315-i314,(-i316)+i315-i314))

            return(B_denom_coef_c[i316]*py_sum(sum_arg_3,0,i315-i316)+diff(B_denom_coef_c[i316],True,1)*py_sum(sum_arg_2,0,i315-i316))

        def sum_arg_7(i316):
            # Child args for sum_arg_7
            def sum_arg_5(i314):
                # Child args for sum_arg_5
                return(is_seq(0,(-n)-i316+2*i315-i314)*Delta_coef_cp[i314]*B_theta_coef_cp[(-n)-i316+2*i315-i314]*is_integer((-n)-i316+2*i315-i314)*is_seq((-n)-i316+2*i315-i314,(-i316)+i315-i314))

            def sum_arg_6(i314):
                # Child args for sum_arg_6
                return((is_seq(0,(-n)-i316+2*i315-i314)*diff(Delta_coef_cp[i314],True,1)*B_theta_coef_cp[(-n)-i316+2*i315-i314]+is_seq(0,(-n)-i316+2*i315-i314)*Delta_coef_cp[i314]*diff(B_theta_coef_cp[(-n)-i316+2*i315-i314],True,1))*is_integer((-n)-i316+2*i315-i314)*is_seq((-n)-i316+2*i315-i314,(-i316)+i315-i314))

            return(B_denom_coef_c[i316]*py_sum(sum_arg_6,0,i315-i316)+diff(B_denom_coef_c[i316],True,1)*py_sum(sum_arg_5,0,i315-i316))

        return(iota_coef[n-i315]*(n*py_sum(sum_arg_7,0,i315)-i315*py_sum(sum_arg_4,0,i315)))

    def sum_arg_1(i307):
        # Child args for sum_arg_1
        return((is_seq(0,n-i307)*diff(B_denom_coef_c[2*i307-n],True,1)*n-is_seq(0,n-i307)*i307*diff(B_denom_coef_c[2*i307-n],True,1))*B_alpha_coef[n-i307]*is_integer(n-i307)*is_seq(n-i307,i307))


    out = -(((2*Delta_coef_cp[0]-2)*is_seq(0,n)*dl_p*n*is_integer(n)*py_sum_parallel(sum_arg_153,0,n)+(2-2*Delta_coef_cp[0])*is_seq(0,n)*dl_p*n*is_integer(n)*py_sum_parallel(sum_arg_152,0,n)+(2*Delta_coef_cp[0]-2)*is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_151,0,n)+(2-2*Delta_coef_cp[0])*is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_150,0,n))*diff(tau_p,False,1)+((2*Delta_coef_cp[0]-2)*iota_coef[0]*is_seq(0,n)*dl_p*n*is_integer(n)*py_sum_parallel(sum_arg_149,0,n)+(2*Delta_coef_cp[0]-2)*is_seq(0,n)*dl_p*n*is_integer(n)*py_sum_parallel(sum_arg_148,0,n)+2*is_seq(0,n)*diff(Delta_coef_cp[0],False,1)*dl_p*n*is_integer(n)*py_sum_parallel(sum_arg_147,0,n)+(2-2*Delta_coef_cp[0])*iota_coef[0]*is_seq(0,n)*dl_p*n*is_integer(n)*py_sum_parallel(sum_arg_146,0,n)+(2-2*Delta_coef_cp[0])*is_seq(0,n)*dl_p*n*is_integer(n)*py_sum_parallel(sum_arg_145,0,n)-2*is_seq(0,n)*diff(Delta_coef_cp[0],False,1)*dl_p*n*is_integer(n)*py_sum_parallel(sum_arg_144,0,n)+(2*Delta_coef_cp[0]-2)*is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_143,0,n)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_142,0,n)+2*is_seq(0,n)*diff(Delta_coef_cp[0],False,1)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_141,0,n)+(2-2*Delta_coef_cp[0])*is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_140,0,n)+(2-2*Delta_coef_cp[0])*iota_coef[0]*is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_139,0,n)-2*is_seq(0,n)*diff(Delta_coef_cp[0],False,1)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_138,0,n))*tau_p+((2*Delta_coef_cp[0]-2)*is_seq(0,n)*dl_p*diff(kap_p,False,1)+2*is_seq(0,n)*diff(Delta_coef_cp[0],False,1)*dl_p*kap_p)*n*is_integer(n)*py_sum_parallel(sum_arg_99,0,n)+(2-2*Delta_coef_cp[0])*iota_coef[0]*is_seq(0,n)*dl_p*kap_p*n*is_integer(n)*py_sum_parallel(sum_arg_98,0,n)+(2-2*Delta_coef_cp[0])*is_seq(0,n)*dl_p*kap_p*n*is_integer(n)*py_sum_parallel(sum_arg_97,0,n)+((2-2*Delta_coef_cp[0])*is_seq(0,n)*dl_p*diff(kap_p,False,1)-2*is_seq(0,n)*diff(Delta_coef_cp[0],False,1)*dl_p*kap_p)*n*is_integer(n)*py_sum_parallel(sum_arg_96,0,n)+(2-2*Delta_coef_cp[0])*iota_coef[0]*n*py_sum_parallel(sum_arg_95,ceil(n/2),floor(n))+(2-2*Delta_coef_cp[0])*n*py_sum_parallel(sum_arg_93,ceil(n/2),floor(n))-2*diff(Delta_coef_cp[0],False,1)*n*py_sum_parallel(sum_arg_91,ceil(n/2),floor(n))+(2-2*Delta_coef_cp[0])*is_seq(0,n)*n*is_integer(n)*py_sum_parallel(sum_arg_89,0,n)+(2-2*Delta_coef_cp[0])*iota_coef[0]*is_seq(0,n)*n*is_integer(n)*py_sum_parallel(sum_arg_88,0,n)-2*is_seq(0,n)*diff(Delta_coef_cp[0],False,1)*n*is_integer(n)*py_sum_parallel(sum_arg_87,0,n)+(2-2*Delta_coef_cp[0])*iota_coef[0]*n*py_sum_parallel(sum_arg_86,ceil(n/2),floor(n))+(2-2*Delta_coef_cp[0])*n*py_sum_parallel(sum_arg_84,ceil(n/2),floor(n))-2*diff(Delta_coef_cp[0],False,1)*n*py_sum_parallel(sum_arg_82,ceil(n/2),floor(n))+(2-2*Delta_coef_cp[0])*is_seq(0,n)*n*is_integer(n)*py_sum_parallel(sum_arg_80,0,n)+4*B_alpha_coef[0]*py_sum_parallel(sum_arg_8,ceil(n/2),floor(n))+(2-2*Delta_coef_cp[0])*iota_coef[0]*is_seq(0,n)*n*is_integer(n)*py_sum_parallel(sum_arg_79,0,n)-2*is_seq(0,n)*diff(Delta_coef_cp[0],False,1)*n*is_integer(n)*py_sum_parallel(sum_arg_78,0,n)+(2-2*Delta_coef_cp[0])*iota_coef[0]*n*py_sum_parallel(sum_arg_77,ceil(n/2),floor(n))+(2-2*Delta_coef_cp[0])*n*py_sum_parallel(sum_arg_75,ceil(n/2),floor(n))-2*diff(Delta_coef_cp[0],False,1)*n*py_sum_parallel(sum_arg_73,ceil(n/2),floor(n))+(2-2*Delta_coef_cp[0])*is_seq(0,n)*n*is_integer(n)*py_sum_parallel(sum_arg_71,0,n)+(2-2*Delta_coef_cp[0])*iota_coef[0]*is_seq(0,n)*n*is_integer(n)*py_sum_parallel(sum_arg_70,0,n)-2*is_seq(0,n)*diff(Delta_coef_cp[0],False,1)*n*is_integer(n)*py_sum_parallel(sum_arg_69,0,n)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*py_sum_parallel(sum_arg_68,ceil(n/2),floor(n))+(2*Delta_coef_cp[0]-2)*py_sum_parallel(sum_arg_66,ceil(n/2),floor(n))+2*diff(Delta_coef_cp[0],False,1)*py_sum_parallel(sum_arg_64,ceil(n/2),floor(n))+(2*Delta_coef_cp[0]-2)*iota_coef[0]*py_sum_parallel(sum_arg_62,ceil(n/2),floor(n))+(2*Delta_coef_cp[0]-2)*py_sum_parallel(sum_arg_60,ceil(n/2),floor(n))+2*diff(Delta_coef_cp[0],False,1)*py_sum_parallel(sum_arg_58,ceil(n/2),floor(n))+(2*Delta_coef_cp[0]-2)*iota_coef[0]*py_sum_parallel(sum_arg_56,ceil(n/2),floor(n))+(2*Delta_coef_cp[0]-2)*py_sum_parallel(sum_arg_54,ceil(n/2),floor(n))+2*diff(Delta_coef_cp[0],False,1)*py_sum_parallel(sum_arg_52,ceil(n/2),floor(n))+(2*Delta_coef_cp[0]-2)*is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_50,0,n)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_49,0,n)+2*is_seq(0,n)*diff(Delta_coef_cp[0],False,1)*is_integer(n)*py_sum_parallel(sum_arg_48,0,n)+(2*Delta_coef_cp[0]-2)*is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_47,0,n)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_46,0,n)+2*is_seq(0,n)*diff(Delta_coef_cp[0],False,1)*is_integer(n)*py_sum_parallel(sum_arg_45,0,n)+(2*Delta_coef_cp[0]-2)*is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_44,0,n)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_43,0,n)+2*is_seq(0,n)*diff(Delta_coef_cp[0],False,1)*is_integer(n)*py_sum_parallel(sum_arg_42,0,n)+(2*Delta_coef_cp[0]-2)*is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_41,0,n)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_40,0,n)+((2*Delta_coef_cp[0]-2)*is_seq(0,n)*dl_p*diff(kap_p,False,1)+2*is_seq(0,n)*diff(Delta_coef_cp[0],False,1)*dl_p*kap_p)*is_integer(n)*py_sum_parallel(sum_arg_39,0,n)+(2-2*Delta_coef_cp[0])*is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_38,0,n)+(2-2*Delta_coef_cp[0])*iota_coef[0]*is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_37,0,n)+((2-2*Delta_coef_cp[0])*is_seq(0,n)*dl_p*diff(kap_p,False,1)-2*is_seq(0,n)*diff(Delta_coef_cp[0],False,1)*dl_p*kap_p)*is_integer(n)*py_sum_parallel(sum_arg_36,0,n)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*n*py_sum_parallel(sum_arg_35,ceil(n/2),floor(n))+(2*Delta_coef_cp[0]-2)*n*py_sum_parallel(sum_arg_33,ceil(n/2),floor(n))+2*diff(Delta_coef_cp[0],False,1)*n*py_sum_parallel(sum_arg_31,ceil(n/2),floor(n))+(4-4*Delta_coef_cp[0])*iota_coef[0]*py_sum_parallel(sum_arg_29,ceil(n/2)-1,floor(n)-2)+(4-4*Delta_coef_cp[0])*py_sum_parallel(sum_arg_27,ceil(n/2)-1,floor(n)-2)-4*diff(Delta_coef_cp[0],False,1)*py_sum_parallel(sum_arg_25,ceil(n/2)-1,floor(n)-2)+n*((-B_alpha_coef[0]*py_sum_parallel(sum_arg_23,ceil(n/2),floor(n)))+B_alpha_coef[0]*py_sum_parallel(sum_arg_20,ceil(n/2),floor(n))+2*B_alpha_coef[0]*py_sum_parallel(sum_arg_18,ceil(n/2),floor(n))-2*B_alpha_coef[0]*py_sum_parallel(sum_arg_14,ceil(n/2),floor(n)))+2*B_alpha_coef[0]*is_seq(0,n)*n*is_integer(n)*py_sum_parallel(sum_arg_137,0,n)-2*B_alpha_coef[0]*n*py_sum_parallel(sum_arg_134,ceil(n/2),floor(n))-2*B_alpha_coef[0]*is_seq(0,n)*n*is_integer(n)*py_sum_parallel(sum_arg_131,0,n)+4*B_alpha_coef[0]*is_seq(0,n-2)*is_integer(n-2)*py_sum_parallel(sum_arg_129,0,n-2)+4*B_alpha_coef[0]*is_seq(0,n-2)*is_integer(n-2)*py_sum_parallel(sum_arg_126,0,n-2)+2*B_alpha_coef[0]*n*py_sum_parallel(sum_arg_123,ceil(n/2),floor(n))+2*B_alpha_coef[0]*is_seq(0,n)*n*is_integer(n)*py_sum_parallel(sum_arg_121,0,n)+4*B_alpha_coef[0]*py_sum_parallel(sum_arg_120,ceil(n/2)-1,floor(n)-2)+4*B_alpha_coef[0]*py_sum_parallel(sum_arg_116,ceil(n/2)-1,floor(n)-2)-4*B_alpha_coef[0]*py_sum_parallel(sum_arg_112,ceil(n/2),floor(n))-4*B_alpha_coef[0]*py_sum_parallel(sum_arg_110,ceil(n/2)-1,floor(n)-2)-4*B_alpha_coef[0]*py_sum_parallel(sum_arg_11,ceil(n/2),floor(n))-4*B_alpha_coef[0]*is_seq(0,n-2)*is_integer(n-2)*py_sum_parallel(sum_arg_108,0,n-2)-B_alpha_coef[0]*py_sum_parallel(sum_arg_107,ceil(n/2),floor(n))+2*B_alpha_coef[0]*py_sum_parallel(sum_arg_105,ceil(n/2),floor(n))+(2*Delta_coef_cp[0]-2)*iota_coef[0]*is_seq(0,n)*dl_p*kap_p*n*is_integer(n)*py_sum_parallel(sum_arg_101,0,n)+(2*Delta_coef_cp[0]-2)*is_seq(0,n)*dl_p*kap_p*n*is_integer(n)*py_sum_parallel(sum_arg_100,0,n)+4*B_alpha_coef[0]*py_sum_parallel(sum_arg_1,ceil(n/2),floor(n)))/(2*B_alpha_coef[0]*n)
    return(out)

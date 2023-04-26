# Evaluating the loop eq. Solves for different quantities
# when different masks are added.
# Uses Xn-1, Yn-1, Zn-1,  B_theta_n-2, Delta_n-1, B_psi_n-3, B_denom n-1,
# iota_coef (n-3)/2 or (n-2)/2, and B_alpha (n-1)/2 or (n-2)/2
# Must be evaluated with Z_coef_cp[n] = 0, p_perp_coef_cp[n] = 0
# B_psi_coef_cp[n-2] = 0, B_denom_coef_c[n] = 0 and B_theta_coef_cp[n] = 0
from math import floor, ceil
from math_utilities import *
import chiphifunc
from jax import jit
from functools import partial
@partial(jit, static_argnums=(0,))
def eval_loop(n, X_coef_cp, Y_coef_cp, Z_coef_cp, \
    B_theta_coef_cp, B_psi_coef_cp, B_alpha_coef, B_denom_coef_c, \
    p_perp_coef_cp, Delta_coef_cp, kap_p, dl_p, tau_p, iota_coef):
    def sum_arg_153(i290):
        # Child args for sum_arg_153
        return(X_coef_cp[i290]*diff(Y_coef_cp[n-i290],True,1))

    def sum_arg_152(i286):
        # Child args for sum_arg_152
        return(Y_coef_cp[i286]*diff(X_coef_cp[n-i286],True,1))

    def sum_arg_151(i258):
        # Child args for sum_arg_151
        return(i258*X_coef_cp[i258]*diff(Y_coef_cp[n-i258],True,1)+i258*diff(X_coef_cp[i258],True,1)*Y_coef_cp[n-i258])

    def sum_arg_150(i256):
        # Child args for sum_arg_150
        return((X_coef_cp[i256]*n-i256*X_coef_cp[i256])*diff(Y_coef_cp[n-i256],True,1)+(diff(X_coef_cp[i256],True,1)*n-i256*diff(X_coef_cp[i256],True,1))*Y_coef_cp[n-i256])

    def sum_arg_149(i290):
        # Child args for sum_arg_149
        return(X_coef_cp[i290]*diff(Y_coef_cp[n-i290],True,2)+diff(X_coef_cp[i290],True,1)*diff(Y_coef_cp[n-i290],True,1))

    def sum_arg_148(i290):
        # Child args for sum_arg_148
        return(X_coef_cp[i290]*diff(Y_coef_cp[n-i290],True,1,False,1)+diff(X_coef_cp[i290],False,1)*diff(Y_coef_cp[n-i290],True,1))

    def sum_arg_147(i290):
        # Child args for sum_arg_147
        return(X_coef_cp[i290]*diff(Y_coef_cp[n-i290],True,1))

    def sum_arg_146(i286):
        # Child args for sum_arg_146
        return(Y_coef_cp[i286]*diff(X_coef_cp[n-i286],True,2)+diff(Y_coef_cp[i286],True,1)*diff(X_coef_cp[n-i286],True,1))

    def sum_arg_145(i286):
        # Child args for sum_arg_145
        return(Y_coef_cp[i286]*diff(X_coef_cp[n-i286],True,1,False,1)+diff(Y_coef_cp[i286],False,1)*diff(X_coef_cp[n-i286],True,1))

    def sum_arg_144(i286):
        # Child args for sum_arg_144
        return(Y_coef_cp[i286]*diff(X_coef_cp[n-i286],True,1))

    def sum_arg_143(i258):
        # Child args for sum_arg_143
        return(i258*diff(X_coef_cp[i258],True,1)*diff(Y_coef_cp[n-i258],False,1)+i258*X_coef_cp[i258]*diff(Y_coef_cp[n-i258],True,1,False,1)+i258*diff(X_coef_cp[i258],False,1)*diff(Y_coef_cp[n-i258],True,1)+i258*diff(X_coef_cp[i258],True,1,False,1)*Y_coef_cp[n-i258])

    def sum_arg_142(i258):
        # Child args for sum_arg_142
        return(i258*X_coef_cp[i258]*diff(Y_coef_cp[n-i258],True,2)+2*i258*diff(X_coef_cp[i258],True,1)*diff(Y_coef_cp[n-i258],True,1)+i258*diff(X_coef_cp[i258],True,2)*Y_coef_cp[n-i258])

    def sum_arg_141(i258):
        # Child args for sum_arg_141
        return(i258*X_coef_cp[i258]*diff(Y_coef_cp[n-i258],True,1)+i258*diff(X_coef_cp[i258],True,1)*Y_coef_cp[n-i258])

    def sum_arg_140(i256):
        # Child args for sum_arg_140
        return((diff(X_coef_cp[i256],True,1)*n-i256*diff(X_coef_cp[i256],True,1))*diff(Y_coef_cp[n-i256],False,1)+(X_coef_cp[i256]*n-i256*X_coef_cp[i256])*diff(Y_coef_cp[n-i256],True,1,False,1)+(diff(X_coef_cp[i256],False,1)*n-i256*diff(X_coef_cp[i256],False,1))*diff(Y_coef_cp[n-i256],True,1)+(diff(X_coef_cp[i256],True,1,False,1)*n-i256*diff(X_coef_cp[i256],True,1,False,1))*Y_coef_cp[n-i256])

    def sum_arg_139(i256):
        # Child args for sum_arg_139
        return((X_coef_cp[i256]*n-i256*X_coef_cp[i256])*diff(Y_coef_cp[n-i256],True,2)+(2*diff(X_coef_cp[i256],True,1)*n-2*i256*diff(X_coef_cp[i256],True,1))*diff(Y_coef_cp[n-i256],True,1)+(diff(X_coef_cp[i256],True,2)*n-i256*diff(X_coef_cp[i256],True,2))*Y_coef_cp[n-i256])

    def sum_arg_138(i256):
        # Child args for sum_arg_138
        return((X_coef_cp[i256]*n-i256*X_coef_cp[i256])*diff(Y_coef_cp[n-i256],True,1)+(diff(X_coef_cp[i256],True,1)*n-i256*diff(X_coef_cp[i256],True,1))*Y_coef_cp[n-i256])

    def sum_arg_137(i350):
        # Child args for sum_arg_137
        def sum_arg_136(i228):
            # Child args for sum_arg_136
            def sum_arg_135(i226):
                # Child args for sum_arg_135
                return(B_denom_coef_c[i226]*B_denom_coef_c[n-i350-i228-i226])

            return(diff(p_perp_coef_cp[i228],False,1)*py_sum(sum_arg_135,0,n-i350-i228))

        return(B_theta_coef_cp[i350]*py_sum(sum_arg_136,0,n-i350))

    def sum_arg_134(i347):
        # Child args for sum_arg_134
        def sum_arg_133(i348):
            # Child args for sum_arg_133
            def sum_arg_132(i346):
                # Child args for sum_arg_132
                return(Delta_coef_cp[i346]*diff(B_theta_coef_cp[(-n)-i348+2*i347-i346],True,1)*is_seq(n-i347,(-i348)+i347-i346))

            return(B_denom_coef_c[i348]*py_sum(sum_arg_132,0,i347-i348))

        return(is_seq(0,n-i347)*iota_coef[n-i347]*is_integer(n-i347)*py_sum(sum_arg_133,0,i347))

    def sum_arg_131(i344):
        # Child args for sum_arg_131
        def sum_arg_130(i342):
            # Child args for sum_arg_130
            return(Delta_coef_cp[i342]*diff(B_theta_coef_cp[n-i344-i342],False,1))

        return(B_denom_coef_c[i344]*py_sum(sum_arg_130,0,n-i344))

    def sum_arg_129(i336):
        # Child args for sum_arg_129
        def sum_arg_127(i298):
            # Child args for sum_arg_127
            return(B_psi_coef_cp[i298]*diff(Delta_coef_cp[n-i336-i298-2],False,1))

        def sum_arg_128(i298):
            # Child args for sum_arg_128
            return(diff(B_psi_coef_cp[i298],True,1)*diff(Delta_coef_cp[n-i336-i298-2],False,1)+B_psi_coef_cp[i298]*diff(Delta_coef_cp[n-i336-i298-2],True,1,False,1))

        return(B_denom_coef_c[i336]*py_sum(sum_arg_128,0,n-i336-2)+diff(B_denom_coef_c[i336],True,1)*py_sum(sum_arg_127,0,n-i336-2))

    def sum_arg_126(i332):
        # Child args for sum_arg_126
        def sum_arg_124(i330):
            # Child args for sum_arg_124
            return(Delta_coef_cp[i330]*diff(B_psi_coef_cp[n-i332-i330-2],False,1))

        def sum_arg_125(i330):
            # Child args for sum_arg_125
            return(diff(Delta_coef_cp[i330],True,1)*diff(B_psi_coef_cp[n-i332-i330-2],False,1)+Delta_coef_cp[i330]*diff(B_psi_coef_cp[n-i332-i330-2],True,1,False,1))

        return(B_denom_coef_c[i332]*py_sum(sum_arg_125,0,n-i332-2)+diff(B_denom_coef_c[i332],True,1)*py_sum(sum_arg_124,0,n-i332-2))

    def sum_arg_123(i339):
        # Child args for sum_arg_123
        def sum_arg_122(i340):
            # Child args for sum_arg_122
            return(B_denom_coef_c[i340]*diff(B_theta_coef_cp[(-n)-i340+2*i339],True,1)*is_seq(n-i339,i339-i340))

        return(is_seq(0,n-i339)*iota_coef[n-i339]*is_integer(n-i339)*py_sum(sum_arg_122,0,i339))

    def sum_arg_121(i338):
        # Child args for sum_arg_121
        return(B_denom_coef_c[i338]*diff(B_theta_coef_cp[n-i338],False,1))

    def sum_arg_120(i327):
        # Child args for sum_arg_120
        def sum_arg_119(i328):
            # Child args for sum_arg_119
            def sum_arg_117(i300):
                # Child args for sum_arg_117
                return(B_psi_coef_cp[i300]*diff(Delta_coef_cp[(-n)-i328+2*i327-i300+2],True,1))

            def sum_arg_118(i300):
                # Child args for sum_arg_118
                return(B_psi_coef_cp[i300]*diff(Delta_coef_cp[(-n)-i328+2*i327-i300+2],True,2)+diff(B_psi_coef_cp[i300],True,1)*diff(Delta_coef_cp[(-n)-i328+2*i327-i300+2],True,1))

            return(is_seq(n-i327-2,i327-i328)*(B_denom_coef_c[i328]*py_sum(sum_arg_118,0,(-n)-i328+2*i327+2)+diff(B_denom_coef_c[i328],True,1)*py_sum(sum_arg_117,0,(-n)-i328+2*i327+2)))

        return(is_seq(0,n-i327-2)*iota_coef[n-i327-2]*is_integer(n-i327-2)*py_sum(sum_arg_119,0,i327))

    def sum_arg_116(i323):
        # Child args for sum_arg_116
        def sum_arg_115(i324):
            # Child args for sum_arg_115
            def sum_arg_113(i320):
                # Child args for sum_arg_113
                return(Delta_coef_cp[i320]*diff(B_psi_coef_cp[(-n)-i324+2*i323-i320+2],True,1))

            def sum_arg_114(i320):
                # Child args for sum_arg_114
                return(Delta_coef_cp[i320]*diff(B_psi_coef_cp[(-n)-i324+2*i323-i320+2],True,2)+diff(Delta_coef_cp[i320],True,1)*diff(B_psi_coef_cp[(-n)-i324+2*i323-i320+2],True,1))

            return(is_seq(n-i323-2,i323-i324)*(B_denom_coef_c[i324]*py_sum(sum_arg_114,0,(-n)-i324+2*i323+2)+diff(B_denom_coef_c[i324],True,1)*py_sum(sum_arg_113,0,(-n)-i324+2*i323+2)))

        return(is_seq(0,n-i323-2)*iota_coef[n-i323-2]*is_integer(n-i323-2)*py_sum(sum_arg_115,0,i323))

    def sum_arg_112(i313):
        # Child args for sum_arg_112
        def sum_arg_111(i314):
            # Child args for sum_arg_111
            return((diff(B_denom_coef_c[i314],True,1)*Delta_coef_cp[(-n)-i314+2*i313]+B_denom_coef_c[i314]*diff(Delta_coef_cp[(-n)-i314+2*i313],True,1))*is_seq(n-i313,i313-i314))

        return((is_seq(0,n-i313)*n-is_seq(0,n-i313)*i313)*B_alpha_coef[n-i313]*is_integer(n-i313)*py_sum(sum_arg_111,0,i313))

    def sum_arg_110(i307):
        # Child args for sum_arg_110
        def sum_arg_109(i308):
            # Child args for sum_arg_109
            return((B_denom_coef_c[i308]*diff(B_psi_coef_cp[(-n)-i308+2*i307+2],True,2)+diff(B_denom_coef_c[i308],True,1)*diff(B_psi_coef_cp[(-n)-i308+2*i307+2],True,1))*is_seq(n-i307-2,i307-i308))

        return(is_seq(0,n-i307-2)*iota_coef[n-i307-2]*is_integer(n-i307-2)*py_sum(sum_arg_109,0,i307))

    def sum_arg_108(i306):
        # Child args for sum_arg_108
        return(diff(B_denom_coef_c[i306],True,1)*diff(B_psi_coef_cp[n-i306-2],False,1)+B_denom_coef_c[i306]*diff(B_psi_coef_cp[n-i306-2],True,1,False,1))

    def sum_arg_107(i301):
        # Child args for sum_arg_107
        def sum_arg_106(i302):
            # Child args for sum_arg_106
            return((i302*diff(B_denom_coef_c[i302],True,1)*Delta_coef_cp[(-n)-i302+2*i301]+i302*B_denom_coef_c[i302]*diff(Delta_coef_cp[(-n)-i302+2*i301],True,1))*is_seq(n-i301,i301-i302))

        return(is_seq(0,n-i301)*B_alpha_coef[n-i301]*is_integer(n-i301)*py_sum(sum_arg_106,0,i301))

    def sum_arg_105(i293):
        # Child args for sum_arg_105
        def sum_arg_104(i294):
            # Child args for sum_arg_104
            def sum_arg_102(i248):
                # Child args for sum_arg_102
                return(B_denom_coef_c[i248]*B_denom_coef_c[(-n)-i294+2*i293-i248])

            def sum_arg_103(i248):
                # Child args for sum_arg_103
                return(diff(B_denom_coef_c[i248],True,1)*B_denom_coef_c[(-n)-i294+2*i293-i248]+B_denom_coef_c[i248]*diff(B_denom_coef_c[(-n)-i294+2*i293-i248],True,1))

            return(is_seq(n-i293,i293-i294)*(i294*p_perp_coef_cp[i294]*py_sum(sum_arg_103,0,(-n)-i294+2*i293)+i294*diff(p_perp_coef_cp[i294],True,1)*py_sum(sum_arg_102,0,(-n)-i294+2*i293)))

        return(is_seq(0,n-i293)*B_alpha_coef[n-i293]*is_integer(n-i293)*py_sum(sum_arg_104,0,i293))

    def sum_arg_101(i292):
        # Child args for sum_arg_101
        return(X_coef_cp[i292]*diff(Z_coef_cp[n-i292],True,2)+diff(X_coef_cp[i292],True,1)*diff(Z_coef_cp[n-i292],True,1))

    def sum_arg_100(i292):
        # Child args for sum_arg_100
        return(X_coef_cp[i292]*diff(Z_coef_cp[n-i292],True,1,False,1)+diff(X_coef_cp[i292],False,1)*diff(Z_coef_cp[n-i292],True,1))

    def sum_arg_99(i292):
        # Child args for sum_arg_99
        return(X_coef_cp[i292]*diff(Z_coef_cp[n-i292],True,1))

    def sum_arg_98(i288):
        # Child args for sum_arg_98
        return(Z_coef_cp[i288]*diff(X_coef_cp[n-i288],True,2)+diff(Z_coef_cp[i288],True,1)*diff(X_coef_cp[n-i288],True,1))

    def sum_arg_97(i288):
        # Child args for sum_arg_97
        return(Z_coef_cp[i288]*diff(X_coef_cp[n-i288],True,1,False,1)+diff(Z_coef_cp[i288],False,1)*diff(X_coef_cp[n-i288],True,1))

    def sum_arg_96(i288):
        # Child args for sum_arg_96
        return(Z_coef_cp[i288]*diff(X_coef_cp[n-i288],True,1))

    def sum_arg_95(i283):
        # Child args for sum_arg_95
        def sum_arg_94(i284):
            # Child args for sum_arg_94
            return((diff(Z_coef_cp[i284],True,1)*diff(Z_coef_cp[(-n)-i284+2*i283],True,2)+diff(Z_coef_cp[i284],True,2)*diff(Z_coef_cp[(-n)-i284+2*i283],True,1))*is_seq(n-i283,i283-i284))

        return(is_seq(0,n-i283)*iota_coef[n-i283]*is_integer(n-i283)*py_sum(sum_arg_94,0,i283))

    def sum_arg_93(i283):
        # Child args for sum_arg_93
        def sum_arg_92(i284):
            # Child args for sum_arg_92
            return((diff(Z_coef_cp[i284],True,1)*diff(Z_coef_cp[(-n)-i284+2*i283],True,1,False,1)+diff(Z_coef_cp[i284],True,1,False,1)*diff(Z_coef_cp[(-n)-i284+2*i283],True,1))*is_seq(n-i283,i283-i284))

        return(is_seq(0,n-i283)*iota_coef[n-i283]*is_integer(n-i283)*py_sum(sum_arg_92,0,i283))

    def sum_arg_91(i283):
        # Child args for sum_arg_91
        def sum_arg_90(i284):
            # Child args for sum_arg_90
            return(diff(Z_coef_cp[i284],True,1)*diff(Z_coef_cp[(-n)-i284+2*i283],True,1)*is_seq(n-i283,i283-i284))

        return(is_seq(0,n-i283)*iota_coef[n-i283]*is_integer(n-i283)*py_sum(sum_arg_90,0,i283))

    def sum_arg_89(i282):
        # Child args for sum_arg_89
        return(diff(Z_coef_cp[i282],True,1)*diff(Z_coef_cp[n-i282],False,2)+diff(Z_coef_cp[i282],True,1,False,1)*diff(Z_coef_cp[n-i282],False,1))

    def sum_arg_88(i282):
        # Child args for sum_arg_88
        return(diff(Z_coef_cp[i282],True,2)*diff(Z_coef_cp[n-i282],False,1)+diff(Z_coef_cp[i282],True,1)*diff(Z_coef_cp[n-i282],True,1,False,1))

    def sum_arg_87(i282):
        # Child args for sum_arg_87
        return(diff(Z_coef_cp[i282],True,1)*diff(Z_coef_cp[n-i282],False,1))

    def sum_arg_86(i279):
        # Child args for sum_arg_86
        def sum_arg_85(i280):
            # Child args for sum_arg_85
            return((diff(Y_coef_cp[i280],True,1)*diff(Y_coef_cp[(-n)-i280+2*i279],True,2)+diff(Y_coef_cp[i280],True,2)*diff(Y_coef_cp[(-n)-i280+2*i279],True,1))*is_seq(n-i279,i279-i280))

        return(is_seq(0,n-i279)*iota_coef[n-i279]*is_integer(n-i279)*py_sum(sum_arg_85,0,i279))

    def sum_arg_84(i279):
        # Child args for sum_arg_84
        def sum_arg_83(i280):
            # Child args for sum_arg_83
            return((diff(Y_coef_cp[i280],True,1)*diff(Y_coef_cp[(-n)-i280+2*i279],True,1,False,1)+diff(Y_coef_cp[i280],True,1,False,1)*diff(Y_coef_cp[(-n)-i280+2*i279],True,1))*is_seq(n-i279,i279-i280))

        return(is_seq(0,n-i279)*iota_coef[n-i279]*is_integer(n-i279)*py_sum(sum_arg_83,0,i279))

    def sum_arg_82(i279):
        # Child args for sum_arg_82
        def sum_arg_81(i280):
            # Child args for sum_arg_81
            return(diff(Y_coef_cp[i280],True,1)*diff(Y_coef_cp[(-n)-i280+2*i279],True,1)*is_seq(n-i279,i279-i280))

        return(is_seq(0,n-i279)*iota_coef[n-i279]*is_integer(n-i279)*py_sum(sum_arg_81,0,i279))

    def sum_arg_80(i278):
        # Child args for sum_arg_80
        return(diff(Y_coef_cp[i278],True,1)*diff(Y_coef_cp[n-i278],False,2)+diff(Y_coef_cp[i278],True,1,False,1)*diff(Y_coef_cp[n-i278],False,1))

    def sum_arg_79(i278):
        # Child args for sum_arg_79
        return(diff(Y_coef_cp[i278],True,2)*diff(Y_coef_cp[n-i278],False,1)+diff(Y_coef_cp[i278],True,1)*diff(Y_coef_cp[n-i278],True,1,False,1))

    def sum_arg_78(i278):
        # Child args for sum_arg_78
        return(diff(Y_coef_cp[i278],True,1)*diff(Y_coef_cp[n-i278],False,1))

    def sum_arg_77(i275):
        # Child args for sum_arg_77
        def sum_arg_76(i276):
            # Child args for sum_arg_76
            return((diff(X_coef_cp[i276],True,1)*diff(X_coef_cp[(-n)-i276+2*i275],True,2)+diff(X_coef_cp[i276],True,2)*diff(X_coef_cp[(-n)-i276+2*i275],True,1))*is_seq(n-i275,i275-i276))

        return(is_seq(0,n-i275)*iota_coef[n-i275]*is_integer(n-i275)*py_sum(sum_arg_76,0,i275))

    def sum_arg_75(i275):
        # Child args for sum_arg_75
        def sum_arg_74(i276):
            # Child args for sum_arg_74
            return((diff(X_coef_cp[i276],True,1)*diff(X_coef_cp[(-n)-i276+2*i275],True,1,False,1)+diff(X_coef_cp[i276],True,1,False,1)*diff(X_coef_cp[(-n)-i276+2*i275],True,1))*is_seq(n-i275,i275-i276))

        return(is_seq(0,n-i275)*iota_coef[n-i275]*is_integer(n-i275)*py_sum(sum_arg_74,0,i275))

    def sum_arg_73(i275):
        # Child args for sum_arg_73
        def sum_arg_72(i276):
            # Child args for sum_arg_72
            return(diff(X_coef_cp[i276],True,1)*diff(X_coef_cp[(-n)-i276+2*i275],True,1)*is_seq(n-i275,i275-i276))

        return(is_seq(0,n-i275)*iota_coef[n-i275]*is_integer(n-i275)*py_sum(sum_arg_72,0,i275))

    def sum_arg_71(i274):
        # Child args for sum_arg_71
        return(diff(X_coef_cp[i274],True,1)*diff(X_coef_cp[n-i274],False,2)+diff(X_coef_cp[i274],True,1,False,1)*diff(X_coef_cp[n-i274],False,1))

    def sum_arg_70(i274):
        # Child args for sum_arg_70
        return(diff(X_coef_cp[i274],True,2)*diff(X_coef_cp[n-i274],False,1)+diff(X_coef_cp[i274],True,1)*diff(X_coef_cp[n-i274],True,1,False,1))

    def sum_arg_69(i274):
        # Child args for sum_arg_69
        return(diff(X_coef_cp[i274],True,1)*diff(X_coef_cp[n-i274],False,1))

    def sum_arg_68(i271):
        # Child args for sum_arg_68
        def sum_arg_67(i272):
            # Child args for sum_arg_67
            return((i272*Z_coef_cp[i272]*diff(Z_coef_cp[(-n)-i272+2*i271],True,3)+2*i272*diff(Z_coef_cp[i272],True,1)*diff(Z_coef_cp[(-n)-i272+2*i271],True,2)+i272*diff(Z_coef_cp[i272],True,2)*diff(Z_coef_cp[(-n)-i272+2*i271],True,1))*is_seq(n-i271,i271-i272))

        return(is_seq(0,n-i271)*iota_coef[n-i271]*is_integer(n-i271)*py_sum(sum_arg_67,0,i271))

    def sum_arg_66(i271):
        # Child args for sum_arg_66
        def sum_arg_65(i272):
            # Child args for sum_arg_65
            return((i272*Z_coef_cp[i272]*diff(Z_coef_cp[(-n)-i272+2*i271],True,2,False,1)+i272*diff(Z_coef_cp[i272],False,1)*diff(Z_coef_cp[(-n)-i272+2*i271],True,2)+i272*diff(Z_coef_cp[i272],True,1)*diff(Z_coef_cp[(-n)-i272+2*i271],True,1,False,1)+i272*diff(Z_coef_cp[i272],True,1,False,1)*diff(Z_coef_cp[(-n)-i272+2*i271],True,1))*is_seq(n-i271,i271-i272))

        return(is_seq(0,n-i271)*iota_coef[n-i271]*is_integer(n-i271)*py_sum(sum_arg_65,0,i271))

    def sum_arg_64(i271):
        # Child args for sum_arg_64
        def sum_arg_63(i272):
            # Child args for sum_arg_63
            return((i272*Z_coef_cp[i272]*diff(Z_coef_cp[(-n)-i272+2*i271],True,2)+i272*diff(Z_coef_cp[i272],True,1)*diff(Z_coef_cp[(-n)-i272+2*i271],True,1))*is_seq(n-i271,i271-i272))

        return(is_seq(0,n-i271)*iota_coef[n-i271]*is_integer(n-i271)*py_sum(sum_arg_63,0,i271))

    def sum_arg_62(i269):
        # Child args for sum_arg_62
        def sum_arg_61(i270):
            # Child args for sum_arg_61
            return((i270*Y_coef_cp[i270]*diff(Y_coef_cp[(-n)-i270+2*i269],True,3)+2*i270*diff(Y_coef_cp[i270],True,1)*diff(Y_coef_cp[(-n)-i270+2*i269],True,2)+i270*diff(Y_coef_cp[i270],True,2)*diff(Y_coef_cp[(-n)-i270+2*i269],True,1))*is_seq(n-i269,i269-i270))

        return(is_seq(0,n-i269)*iota_coef[n-i269]*is_integer(n-i269)*py_sum(sum_arg_61,0,i269))

    def sum_arg_60(i269):
        # Child args for sum_arg_60
        def sum_arg_59(i270):
            # Child args for sum_arg_59
            return((i270*Y_coef_cp[i270]*diff(Y_coef_cp[(-n)-i270+2*i269],True,2,False,1)+i270*diff(Y_coef_cp[i270],False,1)*diff(Y_coef_cp[(-n)-i270+2*i269],True,2)+i270*diff(Y_coef_cp[i270],True,1)*diff(Y_coef_cp[(-n)-i270+2*i269],True,1,False,1)+i270*diff(Y_coef_cp[i270],True,1,False,1)*diff(Y_coef_cp[(-n)-i270+2*i269],True,1))*is_seq(n-i269,i269-i270))

        return(is_seq(0,n-i269)*iota_coef[n-i269]*is_integer(n-i269)*py_sum(sum_arg_59,0,i269))

    def sum_arg_58(i269):
        # Child args for sum_arg_58
        def sum_arg_57(i270):
            # Child args for sum_arg_57
            return((i270*Y_coef_cp[i270]*diff(Y_coef_cp[(-n)-i270+2*i269],True,2)+i270*diff(Y_coef_cp[i270],True,1)*diff(Y_coef_cp[(-n)-i270+2*i269],True,1))*is_seq(n-i269,i269-i270))

        return(is_seq(0,n-i269)*iota_coef[n-i269]*is_integer(n-i269)*py_sum(sum_arg_57,0,i269))

    def sum_arg_56(i267):
        # Child args for sum_arg_56
        def sum_arg_55(i268):
            # Child args for sum_arg_55
            return((i268*X_coef_cp[i268]*diff(X_coef_cp[(-n)-i268+2*i267],True,3)+2*i268*diff(X_coef_cp[i268],True,1)*diff(X_coef_cp[(-n)-i268+2*i267],True,2)+i268*diff(X_coef_cp[i268],True,2)*diff(X_coef_cp[(-n)-i268+2*i267],True,1))*is_seq(n-i267,i267-i268))

        return(is_seq(0,n-i267)*iota_coef[n-i267]*is_integer(n-i267)*py_sum(sum_arg_55,0,i267))

    def sum_arg_54(i267):
        # Child args for sum_arg_54
        def sum_arg_53(i268):
            # Child args for sum_arg_53
            return((i268*X_coef_cp[i268]*diff(X_coef_cp[(-n)-i268+2*i267],True,2,False,1)+i268*diff(X_coef_cp[i268],False,1)*diff(X_coef_cp[(-n)-i268+2*i267],True,2)+i268*diff(X_coef_cp[i268],True,1)*diff(X_coef_cp[(-n)-i268+2*i267],True,1,False,1)+i268*diff(X_coef_cp[i268],True,1,False,1)*diff(X_coef_cp[(-n)-i268+2*i267],True,1))*is_seq(n-i267,i267-i268))

        return(is_seq(0,n-i267)*iota_coef[n-i267]*is_integer(n-i267)*py_sum(sum_arg_53,0,i267))

    def sum_arg_52(i267):
        # Child args for sum_arg_52
        def sum_arg_51(i268):
            # Child args for sum_arg_51
            return((i268*X_coef_cp[i268]*diff(X_coef_cp[(-n)-i268+2*i267],True,2)+i268*diff(X_coef_cp[i268],True,1)*diff(X_coef_cp[(-n)-i268+2*i267],True,1))*is_seq(n-i267,i267-i268))

        return(is_seq(0,n-i267)*iota_coef[n-i267]*is_integer(n-i267)*py_sum(sum_arg_51,0,i267))

    def sum_arg_50(i266):
        # Child args for sum_arg_50
        return(i266*diff(Z_coef_cp[i266],True,1)*diff(Z_coef_cp[n-i266],False,2)+i266*diff(Z_coef_cp[i266],True,1,False,1)*diff(Z_coef_cp[n-i266],False,1)+i266*Z_coef_cp[i266]*diff(Z_coef_cp[n-i266],True,1,False,2)+i266*diff(Z_coef_cp[i266],False,1)*diff(Z_coef_cp[n-i266],True,1,False,1))

    def sum_arg_49(i266):
        # Child args for sum_arg_49
        return(i266*diff(Z_coef_cp[i266],True,2)*diff(Z_coef_cp[n-i266],False,1)+i266*Z_coef_cp[i266]*diff(Z_coef_cp[n-i266],True,2,False,1)+2*i266*diff(Z_coef_cp[i266],True,1)*diff(Z_coef_cp[n-i266],True,1,False,1))

    def sum_arg_48(i266):
        # Child args for sum_arg_48
        return(i266*diff(Z_coef_cp[i266],True,1)*diff(Z_coef_cp[n-i266],False,1)+i266*Z_coef_cp[i266]*diff(Z_coef_cp[n-i266],True,1,False,1))

    def sum_arg_47(i264):
        # Child args for sum_arg_47
        return(i264*diff(Y_coef_cp[i264],True,1)*diff(Y_coef_cp[n-i264],False,2)+i264*diff(Y_coef_cp[i264],True,1,False,1)*diff(Y_coef_cp[n-i264],False,1)+i264*Y_coef_cp[i264]*diff(Y_coef_cp[n-i264],True,1,False,2)+i264*diff(Y_coef_cp[i264],False,1)*diff(Y_coef_cp[n-i264],True,1,False,1))

    def sum_arg_46(i264):
        # Child args for sum_arg_46
        return(i264*diff(Y_coef_cp[i264],True,2)*diff(Y_coef_cp[n-i264],False,1)+i264*Y_coef_cp[i264]*diff(Y_coef_cp[n-i264],True,2,False,1)+2*i264*diff(Y_coef_cp[i264],True,1)*diff(Y_coef_cp[n-i264],True,1,False,1))

    def sum_arg_45(i264):
        # Child args for sum_arg_45
        return(i264*diff(Y_coef_cp[i264],True,1)*diff(Y_coef_cp[n-i264],False,1)+i264*Y_coef_cp[i264]*diff(Y_coef_cp[n-i264],True,1,False,1))

    def sum_arg_44(i262):
        # Child args for sum_arg_44
        return(i262*diff(X_coef_cp[i262],True,1)*diff(X_coef_cp[n-i262],False,2)+i262*diff(X_coef_cp[i262],True,1,False,1)*diff(X_coef_cp[n-i262],False,1)+i262*X_coef_cp[i262]*diff(X_coef_cp[n-i262],True,1,False,2)+i262*diff(X_coef_cp[i262],False,1)*diff(X_coef_cp[n-i262],True,1,False,1))

    def sum_arg_43(i262):
        # Child args for sum_arg_43
        return(i262*diff(X_coef_cp[i262],True,2)*diff(X_coef_cp[n-i262],False,1)+i262*X_coef_cp[i262]*diff(X_coef_cp[n-i262],True,2,False,1)+2*i262*diff(X_coef_cp[i262],True,1)*diff(X_coef_cp[n-i262],True,1,False,1))

    def sum_arg_42(i262):
        # Child args for sum_arg_42
        return(i262*diff(X_coef_cp[i262],True,1)*diff(X_coef_cp[n-i262],False,1)+i262*X_coef_cp[i262]*diff(X_coef_cp[n-i262],True,1,False,1))

    def sum_arg_41(i260):
        # Child args for sum_arg_41
        return(i260*diff(X_coef_cp[i260],True,1)*diff(Z_coef_cp[n-i260],False,1)+i260*X_coef_cp[i260]*diff(Z_coef_cp[n-i260],True,1,False,1)+i260*diff(X_coef_cp[i260],False,1)*diff(Z_coef_cp[n-i260],True,1)+i260*diff(X_coef_cp[i260],True,1,False,1)*Z_coef_cp[n-i260])

    def sum_arg_40(i260):
        # Child args for sum_arg_40
        return(i260*X_coef_cp[i260]*diff(Z_coef_cp[n-i260],True,2)+2*i260*diff(X_coef_cp[i260],True,1)*diff(Z_coef_cp[n-i260],True,1)+i260*diff(X_coef_cp[i260],True,2)*Z_coef_cp[n-i260])

    def sum_arg_39(i260):
        # Child args for sum_arg_39
        return(i260*X_coef_cp[i260]*diff(Z_coef_cp[n-i260],True,1)+i260*diff(X_coef_cp[i260],True,1)*Z_coef_cp[n-i260])

    def sum_arg_38(i254):
        # Child args for sum_arg_38
        return((diff(X_coef_cp[i254],True,1)*n-i254*diff(X_coef_cp[i254],True,1))*diff(Z_coef_cp[n-i254],False,1)+(X_coef_cp[i254]*n-i254*X_coef_cp[i254])*diff(Z_coef_cp[n-i254],True,1,False,1)+(diff(X_coef_cp[i254],False,1)*n-i254*diff(X_coef_cp[i254],False,1))*diff(Z_coef_cp[n-i254],True,1)+(diff(X_coef_cp[i254],True,1,False,1)*n-i254*diff(X_coef_cp[i254],True,1,False,1))*Z_coef_cp[n-i254])

    def sum_arg_37(i254):
        # Child args for sum_arg_37
        return((X_coef_cp[i254]*n-i254*X_coef_cp[i254])*diff(Z_coef_cp[n-i254],True,2)+(2*diff(X_coef_cp[i254],True,1)*n-2*i254*diff(X_coef_cp[i254],True,1))*diff(Z_coef_cp[n-i254],True,1)+(diff(X_coef_cp[i254],True,2)*n-i254*diff(X_coef_cp[i254],True,2))*Z_coef_cp[n-i254])

    def sum_arg_36(i254):
        # Child args for sum_arg_36
        return((X_coef_cp[i254]*n-i254*X_coef_cp[i254])*diff(Z_coef_cp[n-i254],True,1)+(diff(X_coef_cp[i254],True,1)*n-i254*diff(X_coef_cp[i254],True,1))*Z_coef_cp[n-i254])

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

    def sum_arg_23(i357):
        # Child args for sum_arg_23
        def sum_arg_22(i358):
            # Child args for sum_arg_22
            def sum_arg_21(i806):
                # Child args for sum_arg_21
                return(diff(B_denom_coef_c[(-i806)-i358+i357],True,1)*Delta_coef_cp[i806])

            return(is_seq(0,(-n)+i358+i357)*B_theta_coef_cp[(-n)+i358+i357]*is_integer((-n)+i358+i357)*is_seq((-n)+i358+i357,i358)*py_sum(sum_arg_21,0,i357-i358))

        return(iota_coef[n-i357]*py_sum(sum_arg_22,0,i357))

    def sum_arg_20(i355):
        # Child args for sum_arg_20
        def sum_arg_19(i238):
            # Child args for sum_arg_19
            return(Delta_coef_cp[i238]*diff(B_denom_coef_c[(-n)+2*i355-i238],True,1))

        return(is_seq(0,n-i355)*B_alpha_coef[n-i355]*is_integer(n-i355)*is_seq(n-i355,i355)*py_sum(sum_arg_19,0,2*i355-n))

    def sum_arg_18(i353):
        # Child args for sum_arg_18
        def sum_arg_17(i354):
            # Child args for sum_arg_17
            def sum_arg_16(i767):
                # Child args for sum_arg_16
                def sum_arg_15(i790):
                    # Child args for sum_arg_15
                    return(B_denom_coef_c[(-i790)-i767-i354+i353]*B_denom_coef_c[i790])

                return(diff(p_perp_coef_cp[i767],True,1)*py_sum(sum_arg_15,0,(-i767)-i354+i353))

            return(is_seq(0,(-n)+i354+i353)*B_theta_coef_cp[(-n)+i354+i353]*is_integer((-n)+i354+i353)*is_seq((-n)+i354+i353,i354)*py_sum(sum_arg_16,0,i353-i354))

        return(iota_coef[n-i353]*py_sum(sum_arg_17,0,i353))

    def sum_arg_14(i351):
        # Child args for sum_arg_14
        def sum_arg_13(i234):
            # Child args for sum_arg_13
            def sum_arg_12(i230):
                # Child args for sum_arg_12
                return(B_denom_coef_c[i230]*B_denom_coef_c[(-n)+2*i351-i234-i230])

            return(diff(p_perp_coef_cp[i234],True,1)*py_sum(sum_arg_12,0,(-n)+2*i351-i234))

        return(is_seq(0,n-i351)*B_alpha_coef[n-i351]*is_integer(n-i351)*is_seq(n-i351,i351)*py_sum(sum_arg_13,0,2*i351-n))

    def sum_arg_11(i333):
        # Child args for sum_arg_11
        def sum_arg_9(i334):
            # Child args for sum_arg_9
            return((is_seq(0,(-n)-i334+2*i333)*diff(B_denom_coef_c[i334],True,1)*B_theta_coef_cp[(-n)-i334+2*i333]+is_seq(0,(-n)-i334+2*i333)*B_denom_coef_c[i334]*diff(B_theta_coef_cp[(-n)-i334+2*i333],True,1))*is_integer((-n)-i334+2*i333)*is_seq((-n)-i334+2*i333,i333-i334))

        def sum_arg_10(i334):
            # Child args for sum_arg_10
            return((is_seq(0,(-n)-i334+2*i333)*diff(B_denom_coef_c[i334],True,1)*B_theta_coef_cp[(-n)-i334+2*i333]+is_seq(0,(-n)-i334+2*i333)*B_denom_coef_c[i334]*diff(B_theta_coef_cp[(-n)-i334+2*i333],True,1))*is_integer((-n)-i334+2*i333)*is_seq((-n)-i334+2*i333,i333-i334))

        return(iota_coef[n-i333]*(n*py_sum(sum_arg_10,0,i333)-i333*py_sum(sum_arg_9,0,i333)))

    def sum_arg_8(i317):
        # Child args for sum_arg_8
        def sum_arg_4(i318):
            # Child args for sum_arg_4
            def sum_arg_2(i316):
                # Child args for sum_arg_2
                return(is_seq(0,(-n)-i318+2*i317-i316)*Delta_coef_cp[i316]*B_theta_coef_cp[(-n)-i318+2*i317-i316]*is_integer((-n)-i318+2*i317-i316)*is_seq((-n)-i318+2*i317-i316,(-i318)+i317-i316))

            def sum_arg_3(i316):
                # Child args for sum_arg_3
                return((is_seq(0,(-n)-i318+2*i317-i316)*diff(Delta_coef_cp[i316],True,1)*B_theta_coef_cp[(-n)-i318+2*i317-i316]+is_seq(0,(-n)-i318+2*i317-i316)*Delta_coef_cp[i316]*diff(B_theta_coef_cp[(-n)-i318+2*i317-i316],True,1))*is_integer((-n)-i318+2*i317-i316)*is_seq((-n)-i318+2*i317-i316,(-i318)+i317-i316))

            return(B_denom_coef_c[i318]*py_sum(sum_arg_3,0,i317-i318)+diff(B_denom_coef_c[i318],True,1)*py_sum(sum_arg_2,0,i317-i318))

        def sum_arg_7(i318):
            # Child args for sum_arg_7
            def sum_arg_5(i316):
                # Child args for sum_arg_5
                return(is_seq(0,(-n)-i318+2*i317-i316)*Delta_coef_cp[i316]*B_theta_coef_cp[(-n)-i318+2*i317-i316]*is_integer((-n)-i318+2*i317-i316)*is_seq((-n)-i318+2*i317-i316,(-i318)+i317-i316))

            def sum_arg_6(i316):
                # Child args for sum_arg_6
                return((is_seq(0,(-n)-i318+2*i317-i316)*diff(Delta_coef_cp[i316],True,1)*B_theta_coef_cp[(-n)-i318+2*i317-i316]+is_seq(0,(-n)-i318+2*i317-i316)*Delta_coef_cp[i316]*diff(B_theta_coef_cp[(-n)-i318+2*i317-i316],True,1))*is_integer((-n)-i318+2*i317-i316)*is_seq((-n)-i318+2*i317-i316,(-i318)+i317-i316))

            return(B_denom_coef_c[i318]*py_sum(sum_arg_6,0,i317-i318)+diff(B_denom_coef_c[i318],True,1)*py_sum(sum_arg_5,0,i317-i318))

        return(iota_coef[n-i317]*(n*py_sum(sum_arg_7,0,i317)-i317*py_sum(sum_arg_4,0,i317)))

    def sum_arg_1(i309):
        # Child args for sum_arg_1
        return((is_seq(0,n-i309)*diff(B_denom_coef_c[2*i309-n],True,1)*n-is_seq(0,n-i309)*i309*diff(B_denom_coef_c[2*i309-n],True,1))*B_alpha_coef[n-i309]*is_integer(n-i309)*is_seq(n-i309,i309))


    out = -(((2*Delta_coef_cp[0]-2)*is_seq(0,n)*dl_p*n*is_integer(n)*py_sum(sum_arg_153,0,n)+(2-2*Delta_coef_cp[0])*is_seq(0,n)*dl_p*n*is_integer(n)*py_sum(sum_arg_152,0,n)+(2*Delta_coef_cp[0]-2)*is_seq(0,n)*dl_p*is_integer(n)*py_sum(sum_arg_151,0,n)+(2-2*Delta_coef_cp[0])*is_seq(0,n)*dl_p*is_integer(n)*py_sum(sum_arg_150,0,n))*diff(tau_p,False,1)+((2*Delta_coef_cp[0]-2)*iota_coef[0]*is_seq(0,n)*dl_p*n*is_integer(n)*py_sum(sum_arg_149,0,n)+(2*Delta_coef_cp[0]-2)*is_seq(0,n)*dl_p*n*is_integer(n)*py_sum(sum_arg_148,0,n)+2*is_seq(0,n)*diff(Delta_coef_cp[0],False,1)*dl_p*n*is_integer(n)*py_sum(sum_arg_147,0,n)+(2-2*Delta_coef_cp[0])*iota_coef[0]*is_seq(0,n)*dl_p*n*is_integer(n)*py_sum(sum_arg_146,0,n)+(2-2*Delta_coef_cp[0])*is_seq(0,n)*dl_p*n*is_integer(n)*py_sum(sum_arg_145,0,n)-2*is_seq(0,n)*diff(Delta_coef_cp[0],False,1)*dl_p*n*is_integer(n)*py_sum(sum_arg_144,0,n)+(2*Delta_coef_cp[0]-2)*is_seq(0,n)*dl_p*is_integer(n)*py_sum(sum_arg_143,0,n)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*is_seq(0,n)*dl_p*is_integer(n)*py_sum(sum_arg_142,0,n)+2*is_seq(0,n)*diff(Delta_coef_cp[0],False,1)*dl_p*is_integer(n)*py_sum(sum_arg_141,0,n)+(2-2*Delta_coef_cp[0])*is_seq(0,n)*dl_p*is_integer(n)*py_sum(sum_arg_140,0,n)+(2-2*Delta_coef_cp[0])*iota_coef[0]*is_seq(0,n)*dl_p*is_integer(n)*py_sum(sum_arg_139,0,n)-2*is_seq(0,n)*diff(Delta_coef_cp[0],False,1)*dl_p*is_integer(n)*py_sum(sum_arg_138,0,n))*tau_p+((2*Delta_coef_cp[0]-2)*is_seq(0,n)*dl_p*diff(kap_p,False,1)+2*is_seq(0,n)*diff(Delta_coef_cp[0],False,1)*dl_p*kap_p)*n*is_integer(n)*py_sum(sum_arg_99,0,n)+(2-2*Delta_coef_cp[0])*iota_coef[0]*is_seq(0,n)*dl_p*kap_p*n*is_integer(n)*py_sum(sum_arg_98,0,n)+(2-2*Delta_coef_cp[0])*is_seq(0,n)*dl_p*kap_p*n*is_integer(n)*py_sum(sum_arg_97,0,n)+((2-2*Delta_coef_cp[0])*is_seq(0,n)*dl_p*diff(kap_p,False,1)-2*is_seq(0,n)*diff(Delta_coef_cp[0],False,1)*dl_p*kap_p)*n*is_integer(n)*py_sum(sum_arg_96,0,n)+(2-2*Delta_coef_cp[0])*iota_coef[0]*n*py_sum(sum_arg_95,ceil(n/2),floor(n))+(2-2*Delta_coef_cp[0])*n*py_sum(sum_arg_93,ceil(n/2),floor(n))-2*diff(Delta_coef_cp[0],False,1)*n*py_sum(sum_arg_91,ceil(n/2),floor(n))+(2-2*Delta_coef_cp[0])*is_seq(0,n)*n*is_integer(n)*py_sum(sum_arg_89,0,n)+(2-2*Delta_coef_cp[0])*iota_coef[0]*is_seq(0,n)*n*is_integer(n)*py_sum(sum_arg_88,0,n)-2*is_seq(0,n)*diff(Delta_coef_cp[0],False,1)*n*is_integer(n)*py_sum(sum_arg_87,0,n)+(2-2*Delta_coef_cp[0])*iota_coef[0]*n*py_sum(sum_arg_86,ceil(n/2),floor(n))+(2-2*Delta_coef_cp[0])*n*py_sum(sum_arg_84,ceil(n/2),floor(n))-2*diff(Delta_coef_cp[0],False,1)*n*py_sum(sum_arg_82,ceil(n/2),floor(n))+(2-2*Delta_coef_cp[0])*is_seq(0,n)*n*is_integer(n)*py_sum(sum_arg_80,0,n)+4*B_alpha_coef[0]*py_sum(sum_arg_8,ceil(n/2),floor(n))+(2-2*Delta_coef_cp[0])*iota_coef[0]*is_seq(0,n)*n*is_integer(n)*py_sum(sum_arg_79,0,n)-2*is_seq(0,n)*diff(Delta_coef_cp[0],False,1)*n*is_integer(n)*py_sum(sum_arg_78,0,n)+(2-2*Delta_coef_cp[0])*iota_coef[0]*n*py_sum(sum_arg_77,ceil(n/2),floor(n))+(2-2*Delta_coef_cp[0])*n*py_sum(sum_arg_75,ceil(n/2),floor(n))-2*diff(Delta_coef_cp[0],False,1)*n*py_sum(sum_arg_73,ceil(n/2),floor(n))+(2-2*Delta_coef_cp[0])*is_seq(0,n)*n*is_integer(n)*py_sum(sum_arg_71,0,n)+(2-2*Delta_coef_cp[0])*iota_coef[0]*is_seq(0,n)*n*is_integer(n)*py_sum(sum_arg_70,0,n)-2*is_seq(0,n)*diff(Delta_coef_cp[0],False,1)*n*is_integer(n)*py_sum(sum_arg_69,0,n)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*py_sum(sum_arg_68,ceil(n/2),floor(n))+(2*Delta_coef_cp[0]-2)*py_sum(sum_arg_66,ceil(n/2),floor(n))+2*diff(Delta_coef_cp[0],False,1)*py_sum(sum_arg_64,ceil(n/2),floor(n))+(2*Delta_coef_cp[0]-2)*iota_coef[0]*py_sum(sum_arg_62,ceil(n/2),floor(n))+(2*Delta_coef_cp[0]-2)*py_sum(sum_arg_60,ceil(n/2),floor(n))+2*diff(Delta_coef_cp[0],False,1)*py_sum(sum_arg_58,ceil(n/2),floor(n))+(2*Delta_coef_cp[0]-2)*iota_coef[0]*py_sum(sum_arg_56,ceil(n/2),floor(n))+(2*Delta_coef_cp[0]-2)*py_sum(sum_arg_54,ceil(n/2),floor(n))+2*diff(Delta_coef_cp[0],False,1)*py_sum(sum_arg_52,ceil(n/2),floor(n))+(2*Delta_coef_cp[0]-2)*is_seq(0,n)*is_integer(n)*py_sum(sum_arg_50,0,n)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*is_seq(0,n)*is_integer(n)*py_sum(sum_arg_49,0,n)+2*is_seq(0,n)*diff(Delta_coef_cp[0],False,1)*is_integer(n)*py_sum(sum_arg_48,0,n)+(2*Delta_coef_cp[0]-2)*is_seq(0,n)*is_integer(n)*py_sum(sum_arg_47,0,n)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*is_seq(0,n)*is_integer(n)*py_sum(sum_arg_46,0,n)+2*is_seq(0,n)*diff(Delta_coef_cp[0],False,1)*is_integer(n)*py_sum(sum_arg_45,0,n)+(2*Delta_coef_cp[0]-2)*is_seq(0,n)*is_integer(n)*py_sum(sum_arg_44,0,n)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*is_seq(0,n)*is_integer(n)*py_sum(sum_arg_43,0,n)+2*is_seq(0,n)*diff(Delta_coef_cp[0],False,1)*is_integer(n)*py_sum(sum_arg_42,0,n)+(2*Delta_coef_cp[0]-2)*is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum(sum_arg_41,0,n)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum(sum_arg_40,0,n)+((2*Delta_coef_cp[0]-2)*is_seq(0,n)*dl_p*diff(kap_p,False,1)+2*is_seq(0,n)*diff(Delta_coef_cp[0],False,1)*dl_p*kap_p)*is_integer(n)*py_sum(sum_arg_39,0,n)+(2-2*Delta_coef_cp[0])*is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum(sum_arg_38,0,n)+(2-2*Delta_coef_cp[0])*iota_coef[0]*is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum(sum_arg_37,0,n)+((2-2*Delta_coef_cp[0])*is_seq(0,n)*dl_p*diff(kap_p,False,1)-2*is_seq(0,n)*diff(Delta_coef_cp[0],False,1)*dl_p*kap_p)*is_integer(n)*py_sum(sum_arg_36,0,n)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*n*py_sum(sum_arg_35,ceil(n/2),floor(n))+(2*Delta_coef_cp[0]-2)*n*py_sum(sum_arg_33,ceil(n/2),floor(n))+2*diff(Delta_coef_cp[0],False,1)*n*py_sum(sum_arg_31,ceil(n/2),floor(n))+(4-4*Delta_coef_cp[0])*iota_coef[0]*py_sum(sum_arg_29,ceil(n/2)-1,floor(n)-2)+(4-4*Delta_coef_cp[0])*py_sum(sum_arg_27,ceil(n/2)-1,floor(n)-2)-4*diff(Delta_coef_cp[0],False,1)*py_sum(sum_arg_25,ceil(n/2)-1,floor(n)-2)+n*((-B_alpha_coef[0]*py_sum(sum_arg_23,ceil(n/2),floor(n)))+B_alpha_coef[0]*py_sum(sum_arg_20,ceil(n/2),floor(n))+2*B_alpha_coef[0]*py_sum(sum_arg_18,ceil(n/2),floor(n))-2*B_alpha_coef[0]*py_sum(sum_arg_14,ceil(n/2),floor(n)))+2*B_alpha_coef[0]*is_seq(0,n)*n*is_integer(n)*py_sum(sum_arg_137,0,n)-2*B_alpha_coef[0]*n*py_sum(sum_arg_134,ceil(n/2),floor(n))-2*B_alpha_coef[0]*is_seq(0,n)*n*is_integer(n)*py_sum(sum_arg_131,0,n)+4*B_alpha_coef[0]*is_seq(0,n-2)*is_integer(n-2)*py_sum(sum_arg_129,0,n-2)+4*B_alpha_coef[0]*is_seq(0,n-2)*is_integer(n-2)*py_sum(sum_arg_126,0,n-2)+2*B_alpha_coef[0]*n*py_sum(sum_arg_123,ceil(n/2),floor(n))+2*B_alpha_coef[0]*is_seq(0,n)*n*is_integer(n)*py_sum(sum_arg_121,0,n)+4*B_alpha_coef[0]*py_sum(sum_arg_120,ceil(n/2)-1,floor(n)-2)+4*B_alpha_coef[0]*py_sum(sum_arg_116,ceil(n/2)-1,floor(n)-2)-4*B_alpha_coef[0]*py_sum(sum_arg_112,ceil(n/2),floor(n))-4*B_alpha_coef[0]*py_sum(sum_arg_110,ceil(n/2)-1,floor(n)-2)-4*B_alpha_coef[0]*py_sum(sum_arg_11,ceil(n/2),floor(n))-4*B_alpha_coef[0]*is_seq(0,n-2)*is_integer(n-2)*py_sum(sum_arg_108,0,n-2)-B_alpha_coef[0]*py_sum(sum_arg_107,ceil(n/2),floor(n))+2*B_alpha_coef[0]*py_sum(sum_arg_105,ceil(n/2),floor(n))+(2*Delta_coef_cp[0]-2)*iota_coef[0]*is_seq(0,n)*dl_p*kap_p*n*is_integer(n)*py_sum(sum_arg_101,0,n)+(2*Delta_coef_cp[0]-2)*is_seq(0,n)*dl_p*kap_p*n*is_integer(n)*py_sum(sum_arg_100,0,n)+4*B_alpha_coef[0]*py_sum(sum_arg_1,ceil(n/2),floor(n)))/(2*B_alpha_coef[0]*n)
    return(out)

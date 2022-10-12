# Evaluating loop. 
# Uses Xn-1, Yn-1, Zn-1,  B_theta_n, B_psi_n-3, iota_coef (n-1)/2 or (n-2)/2
# Must be evaluated with Z_coef_cp[n] = 0, p_perp_coef_cp[n] = 0
# B_psi_coef_cp[n-2] = 0 and B_theta_coef_cp[n] = 0 
from math import floor, ceil
from math_utilities import *
import chiphifunc
def eval_loop_test(n, X_coef_cp, Y_coef_cp, Z_coef_cp, \
    B_theta_coef_cp, B_psi_coef_cp, B_alpha_coef, B_denom_coef_c, \
    p_perp_coef_cp, Delta_coef_cp, kap_p, dl_p, tau_p, iota_coef):    
    def sum_arg_153(i292):
        # Child args for sum_arg_153
        return(X_coef_cp[i292]*diff(Y_coef_cp[n-i292],'chi',1))
    
    def sum_arg_152(i288):
        # Child args for sum_arg_152
        return(Y_coef_cp[i288]*diff(X_coef_cp[n-i288],'chi',1))
    
    def sum_arg_151(i260):
        # Child args for sum_arg_151
        return(i260*X_coef_cp[i260]*diff(Y_coef_cp[n-i260],'chi',1)+i260*diff(X_coef_cp[i260],'chi',1)*Y_coef_cp[n-i260])
    
    def sum_arg_150(i258):
        # Child args for sum_arg_150
        return((X_coef_cp[i258]*n-i258*X_coef_cp[i258])*diff(Y_coef_cp[n-i258],'chi',1)+(diff(X_coef_cp[i258],'chi',1)*n-i258*diff(X_coef_cp[i258],'chi',1))*Y_coef_cp[n-i258])
    
    def sum_arg_149(i292):
        # Child args for sum_arg_149
        return(X_coef_cp[i292]*diff(Y_coef_cp[n-i292],'chi',2)+diff(X_coef_cp[i292],'chi',1)*diff(Y_coef_cp[n-i292],'chi',1))
    
    def sum_arg_148(i292):
        # Child args for sum_arg_148
        return(X_coef_cp[i292]*diff(Y_coef_cp[n-i292],'chi',1,'phi',1)+diff(X_coef_cp[i292],'phi',1)*diff(Y_coef_cp[n-i292],'chi',1))
    
    def sum_arg_147(i292):
        # Child args for sum_arg_147
        return(X_coef_cp[i292]*diff(Y_coef_cp[n-i292],'chi',1))
    
    def sum_arg_146(i288):
        # Child args for sum_arg_146
        return(Y_coef_cp[i288]*diff(X_coef_cp[n-i288],'chi',2)+diff(Y_coef_cp[i288],'chi',1)*diff(X_coef_cp[n-i288],'chi',1))
    
    def sum_arg_145(i288):
        # Child args for sum_arg_145
        return(Y_coef_cp[i288]*diff(X_coef_cp[n-i288],'chi',1,'phi',1)+diff(Y_coef_cp[i288],'phi',1)*diff(X_coef_cp[n-i288],'chi',1))
    
    def sum_arg_144(i288):
        # Child args for sum_arg_144
        return(Y_coef_cp[i288]*diff(X_coef_cp[n-i288],'chi',1))
    
    def sum_arg_143(i260):
        # Child args for sum_arg_143
        return(i260*diff(X_coef_cp[i260],'chi',1)*diff(Y_coef_cp[n-i260],'phi',1)+i260*X_coef_cp[i260]*diff(Y_coef_cp[n-i260],'chi',1,'phi',1)+i260*diff(X_coef_cp[i260],'phi',1)*diff(Y_coef_cp[n-i260],'chi',1)+i260*diff(X_coef_cp[i260],'chi',1,'phi',1)*Y_coef_cp[n-i260])
    
    def sum_arg_142(i260):
        # Child args for sum_arg_142
        return(i260*X_coef_cp[i260]*diff(Y_coef_cp[n-i260],'chi',2)+2*i260*diff(X_coef_cp[i260],'chi',1)*diff(Y_coef_cp[n-i260],'chi',1)+i260*diff(X_coef_cp[i260],'chi',2)*Y_coef_cp[n-i260])
    
    def sum_arg_141(i260):
        # Child args for sum_arg_141
        return(i260*X_coef_cp[i260]*diff(Y_coef_cp[n-i260],'chi',1)+i260*diff(X_coef_cp[i260],'chi',1)*Y_coef_cp[n-i260])
    
    def sum_arg_140(i258):
        # Child args for sum_arg_140
        return((diff(X_coef_cp[i258],'chi',1)*n-i258*diff(X_coef_cp[i258],'chi',1))*diff(Y_coef_cp[n-i258],'phi',1)+(X_coef_cp[i258]*n-i258*X_coef_cp[i258])*diff(Y_coef_cp[n-i258],'chi',1,'phi',1)+(diff(X_coef_cp[i258],'phi',1)*n-i258*diff(X_coef_cp[i258],'phi',1))*diff(Y_coef_cp[n-i258],'chi',1)+(diff(X_coef_cp[i258],'chi',1,'phi',1)*n-i258*diff(X_coef_cp[i258],'chi',1,'phi',1))*Y_coef_cp[n-i258])
    
    def sum_arg_139(i258):
        # Child args for sum_arg_139
        return((X_coef_cp[i258]*n-i258*X_coef_cp[i258])*diff(Y_coef_cp[n-i258],'chi',2)+(2*diff(X_coef_cp[i258],'chi',1)*n-2*i258*diff(X_coef_cp[i258],'chi',1))*diff(Y_coef_cp[n-i258],'chi',1)+(diff(X_coef_cp[i258],'chi',2)*n-i258*diff(X_coef_cp[i258],'chi',2))*Y_coef_cp[n-i258])
    
    def sum_arg_138(i258):
        # Child args for sum_arg_138
        return((X_coef_cp[i258]*n-i258*X_coef_cp[i258])*diff(Y_coef_cp[n-i258],'chi',1)+(diff(X_coef_cp[i258],'chi',1)*n-i258*diff(X_coef_cp[i258],'chi',1))*Y_coef_cp[n-i258])
    
    def sum_arg_137(i338):
        # Child args for sum_arg_137    
        def sum_arg_135(i300):
            # Child args for sum_arg_135
            return(B_psi_coef_cp[i300]*diff(Delta_coef_cp[n-i338-i300-2],'phi',1))
            
        def sum_arg_136(i300):
            # Child args for sum_arg_136
            return(diff(B_psi_coef_cp[i300],'chi',1)*diff(Delta_coef_cp[n-i338-i300-2],'phi',1)+B_psi_coef_cp[i300]*diff(Delta_coef_cp[n-i338-i300-2],'chi',1,'phi',1))
        
        return(B_denom_coef_c[i338]*py_sum(sum_arg_136,0,n-i338-2)+diff(B_denom_coef_c[i338],'chi',1)*py_sum(sum_arg_135,0,n-i338-2))
    
    def sum_arg_134(i334):
        # Child args for sum_arg_134    
        def sum_arg_132(i332):
            # Child args for sum_arg_132
            return(Delta_coef_cp[i332]*diff(B_psi_coef_cp[n-i334-i332-2],'phi',1))
            
        def sum_arg_133(i332):
            # Child args for sum_arg_133
            return(diff(Delta_coef_cp[i332],'chi',1)*diff(B_psi_coef_cp[n-i334-i332-2],'phi',1)+Delta_coef_cp[i332]*diff(B_psi_coef_cp[n-i334-i332-2],'chi',1,'phi',1))
        
        return(B_denom_coef_c[i334]*py_sum(sum_arg_133,0,n-i334-2)+diff(B_denom_coef_c[i334],'chi',1)*py_sum(sum_arg_132,0,n-i334-2))
    
    def sum_arg_131(i1072):
        # Child args for sum_arg_131    
        def sum_arg_130(i1034):
            # Child args for sum_arg_130    
            def sum_arg_129(i1032):
                # Child args for sum_arg_129
                return(B_denom_coef_c[i1032]*B_denom_coef_c[n-i1072-i1034-i1032])
            
            return(diff(p_perp_coef_cp[i1034],'phi',1)*py_sum(sum_arg_129,0,n-i1072-i1034))
        
        return(B_theta_coef_cp[i1072]*py_sum(sum_arg_130,0,n-i1072))
    
    def sum_arg_128(i1069):
        # Child args for sum_arg_128    
        def sum_arg_127(i1070):
            # Child args for sum_arg_127    
            def sum_arg_126(i1068):
                # Child args for sum_arg_126
                return(Delta_coef_cp[i1068]*diff(B_theta_coef_cp[(-n)-i1070+2*i1069-i1068],'chi',1)*is_seq(n-i1069,(-i1070)+i1069-i1068))
            
            return(B_denom_coef_c[i1070]*py_sum(sum_arg_126,0,i1069-i1070))
        
        return(is_seq(0,n-i1069)*iota_coef[n-i1069]*is_integer(n-i1069)*py_sum(sum_arg_127,0,i1069))
    
    def sum_arg_125(i1066):
        # Child args for sum_arg_125    
        def sum_arg_124(i1064):
            # Child args for sum_arg_124
            return(Delta_coef_cp[i1064]*diff(B_theta_coef_cp[n-i1066-i1064],'phi',1))
        
        return(B_denom_coef_c[i1066]*py_sum(sum_arg_124,0,n-i1066))
    
    def sum_arg_123(i329):
        # Child args for sum_arg_123    
        def sum_arg_122(i330):
            # Child args for sum_arg_122    
            def sum_arg_120(i302):
                # Child args for sum_arg_120
                return(B_psi_coef_cp[i302]*diff(Delta_coef_cp[(-n)-i330+2*i329-i302+2],'chi',1))
                
            def sum_arg_121(i302):
                # Child args for sum_arg_121
                return(B_psi_coef_cp[i302]*diff(Delta_coef_cp[(-n)-i330+2*i329-i302+2],'chi',2)+diff(B_psi_coef_cp[i302],'chi',1)*diff(Delta_coef_cp[(-n)-i330+2*i329-i302+2],'chi',1))
            
            return(is_seq(n-i329-2,i329-i330)*(B_denom_coef_c[i330]*py_sum(sum_arg_121,0,(-n)-i330+2*i329+2)+diff(B_denom_coef_c[i330],'chi',1)*py_sum(sum_arg_120,0,(-n)-i330+2*i329+2)))
        
        return(is_seq(0,n-i329-2)*iota_coef[n-i329-2]*is_integer(n-i329-2)*py_sum(sum_arg_122,0,i329))
    
    def sum_arg_119(i325):
        # Child args for sum_arg_119    
        def sum_arg_118(i326):
            # Child args for sum_arg_118    
            def sum_arg_116(i322):
                # Child args for sum_arg_116
                return(Delta_coef_cp[i322]*diff(B_psi_coef_cp[(-n)-i326+2*i325-i322+2],'chi',1))
                
            def sum_arg_117(i322):
                # Child args for sum_arg_117
                return(Delta_coef_cp[i322]*diff(B_psi_coef_cp[(-n)-i326+2*i325-i322+2],'chi',2)+diff(Delta_coef_cp[i322],'chi',1)*diff(B_psi_coef_cp[(-n)-i326+2*i325-i322+2],'chi',1))
            
            return(is_seq(n-i325-2,i325-i326)*(B_denom_coef_c[i326]*py_sum(sum_arg_117,0,(-n)-i326+2*i325+2)+diff(B_denom_coef_c[i326],'chi',1)*py_sum(sum_arg_116,0,(-n)-i326+2*i325+2)))
        
        return(is_seq(0,n-i325-2)*iota_coef[n-i325-2]*is_integer(n-i325-2)*py_sum(sum_arg_118,0,i325))
    
    def sum_arg_115(i315):
        # Child args for sum_arg_115    
        def sum_arg_114(i316):
            # Child args for sum_arg_114
            return((diff(B_denom_coef_c[i316],'chi',1)*Delta_coef_cp[(-n)-i316+2*i315]+B_denom_coef_c[i316]*diff(Delta_coef_cp[(-n)-i316+2*i315],'chi',1))*is_seq(n-i315,i315-i316))
        
        return((is_seq(0,n-i315)*n-is_seq(0,n-i315)*i315)*B_alpha_coef[n-i315]*is_integer(n-i315)*py_sum(sum_arg_114,0,i315))
    
    def sum_arg_113(i309):
        # Child args for sum_arg_113    
        def sum_arg_112(i310):
            # Child args for sum_arg_112
            return((B_denom_coef_c[i310]*diff(B_psi_coef_cp[(-n)-i310+2*i309+2],'chi',2)+diff(B_denom_coef_c[i310],'chi',1)*diff(B_psi_coef_cp[(-n)-i310+2*i309+2],'chi',1))*is_seq(n-i309-2,i309-i310))
        
        return(is_seq(0,n-i309-2)*iota_coef[n-i309-2]*is_integer(n-i309-2)*py_sum(sum_arg_112,0,i309))
    
    def sum_arg_111(i308):
        # Child args for sum_arg_111
        return(diff(B_denom_coef_c[i308],'chi',1)*diff(B_psi_coef_cp[n-i308-2],'phi',1)+B_denom_coef_c[i308]*diff(B_psi_coef_cp[n-i308-2],'chi',1,'phi',1))
    
    def sum_arg_110(i303):
        # Child args for sum_arg_110    
        def sum_arg_109(i304):
            # Child args for sum_arg_109
            return((i304*diff(B_denom_coef_c[i304],'chi',1)*Delta_coef_cp[(-n)-i304+2*i303]+i304*B_denom_coef_c[i304]*diff(Delta_coef_cp[(-n)-i304+2*i303],'chi',1))*is_seq(n-i303,i303-i304))
        
        return(is_seq(0,n-i303)*B_alpha_coef[n-i303]*is_integer(n-i303)*py_sum(sum_arg_109,0,i303))
    
    def sum_arg_108(i295):
        # Child args for sum_arg_108    
        def sum_arg_107(i296):
            # Child args for sum_arg_107    
            def sum_arg_105(i250):
                # Child args for sum_arg_105
                return(B_denom_coef_c[i250]*B_denom_coef_c[(-n)-i296+2*i295-i250])
                
            def sum_arg_106(i250):
                # Child args for sum_arg_106
                return(diff(B_denom_coef_c[i250],'chi',1)*B_denom_coef_c[(-n)-i296+2*i295-i250]+B_denom_coef_c[i250]*diff(B_denom_coef_c[(-n)-i296+2*i295-i250],'chi',1))
            
            return(is_seq(n-i295,i295-i296)*(i296*p_perp_coef_cp[i296]*py_sum(sum_arg_106,0,(-n)-i296+2*i295)+i296*diff(p_perp_coef_cp[i296],'chi',1)*py_sum(sum_arg_105,0,(-n)-i296+2*i295)))
        
        return(is_seq(0,n-i295)*B_alpha_coef[n-i295]*is_integer(n-i295)*py_sum(sum_arg_107,0,i295))
    
    def sum_arg_104(i294):
        # Child args for sum_arg_104
        return(X_coef_cp[i294]*diff(Z_coef_cp[n-i294],'chi',2)+diff(X_coef_cp[i294],'chi',1)*diff(Z_coef_cp[n-i294],'chi',1))
    
    def sum_arg_103(i294):
        # Child args for sum_arg_103
        return(X_coef_cp[i294]*diff(Z_coef_cp[n-i294],'chi',1,'phi',1)+diff(X_coef_cp[i294],'phi',1)*diff(Z_coef_cp[n-i294],'chi',1))
    
    def sum_arg_102(i294):
        # Child args for sum_arg_102
        return(X_coef_cp[i294]*diff(Z_coef_cp[n-i294],'chi',1))
    
    def sum_arg_101(i290):
        # Child args for sum_arg_101
        return(Z_coef_cp[i290]*diff(X_coef_cp[n-i290],'chi',2)+diff(Z_coef_cp[i290],'chi',1)*diff(X_coef_cp[n-i290],'chi',1))
    
    def sum_arg_100(i290):
        # Child args for sum_arg_100
        return(Z_coef_cp[i290]*diff(X_coef_cp[n-i290],'chi',1,'phi',1)+diff(Z_coef_cp[i290],'phi',1)*diff(X_coef_cp[n-i290],'chi',1))
    
    def sum_arg_99(i290):
        # Child args for sum_arg_99
        return(Z_coef_cp[i290]*diff(X_coef_cp[n-i290],'chi',1))
    
    def sum_arg_98(i285):
        # Child args for sum_arg_98    
        def sum_arg_97(i286):
            # Child args for sum_arg_97
            return((diff(Z_coef_cp[i286],'chi',1)*diff(Z_coef_cp[(-n)-i286+2*i285],'chi',2)+diff(Z_coef_cp[i286],'chi',2)*diff(Z_coef_cp[(-n)-i286+2*i285],'chi',1))*is_seq(n-i285,i285-i286))
        
        return(is_seq(0,n-i285)*iota_coef[n-i285]*is_integer(n-i285)*py_sum(sum_arg_97,0,i285))
    
    def sum_arg_96(i285):
        # Child args for sum_arg_96    
        def sum_arg_95(i286):
            # Child args for sum_arg_95
            return((diff(Z_coef_cp[i286],'chi',1)*diff(Z_coef_cp[(-n)-i286+2*i285],'chi',1,'phi',1)+diff(Z_coef_cp[i286],'chi',1,'phi',1)*diff(Z_coef_cp[(-n)-i286+2*i285],'chi',1))*is_seq(n-i285,i285-i286))
        
        return(is_seq(0,n-i285)*iota_coef[n-i285]*is_integer(n-i285)*py_sum(sum_arg_95,0,i285))
    
    def sum_arg_94(i285):
        # Child args for sum_arg_94    
        def sum_arg_93(i286):
            # Child args for sum_arg_93
            return(diff(Z_coef_cp[i286],'chi',1)*diff(Z_coef_cp[(-n)-i286+2*i285],'chi',1)*is_seq(n-i285,i285-i286))
        
        return(is_seq(0,n-i285)*iota_coef[n-i285]*is_integer(n-i285)*py_sum(sum_arg_93,0,i285))
    
    def sum_arg_92(i284):
        # Child args for sum_arg_92
        return(diff(Z_coef_cp[i284],'chi',1)*diff(Z_coef_cp[n-i284],'phi',2)+diff(Z_coef_cp[i284],'chi',1,'phi',1)*diff(Z_coef_cp[n-i284],'phi',1))
    
    def sum_arg_91(i284):
        # Child args for sum_arg_91
        return(diff(Z_coef_cp[i284],'chi',2)*diff(Z_coef_cp[n-i284],'phi',1)+diff(Z_coef_cp[i284],'chi',1)*diff(Z_coef_cp[n-i284],'chi',1,'phi',1))
    
    def sum_arg_90(i284):
        # Child args for sum_arg_90
        return(diff(Z_coef_cp[i284],'chi',1)*diff(Z_coef_cp[n-i284],'phi',1))
    
    def sum_arg_89(i281):
        # Child args for sum_arg_89    
        def sum_arg_88(i282):
            # Child args for sum_arg_88
            return((diff(Y_coef_cp[i282],'chi',1)*diff(Y_coef_cp[(-n)-i282+2*i281],'chi',2)+diff(Y_coef_cp[i282],'chi',2)*diff(Y_coef_cp[(-n)-i282+2*i281],'chi',1))*is_seq(n-i281,i281-i282))
        
        return(is_seq(0,n-i281)*iota_coef[n-i281]*is_integer(n-i281)*py_sum(sum_arg_88,0,i281))
    
    def sum_arg_87(i281):
        # Child args for sum_arg_87    
        def sum_arg_86(i282):
            # Child args for sum_arg_86
            return((diff(Y_coef_cp[i282],'chi',1)*diff(Y_coef_cp[(-n)-i282+2*i281],'chi',1,'phi',1)+diff(Y_coef_cp[i282],'chi',1,'phi',1)*diff(Y_coef_cp[(-n)-i282+2*i281],'chi',1))*is_seq(n-i281,i281-i282))
        
        return(is_seq(0,n-i281)*iota_coef[n-i281]*is_integer(n-i281)*py_sum(sum_arg_86,0,i281))
    
    def sum_arg_85(i281):
        # Child args for sum_arg_85    
        def sum_arg_84(i282):
            # Child args for sum_arg_84
            return(diff(Y_coef_cp[i282],'chi',1)*diff(Y_coef_cp[(-n)-i282+2*i281],'chi',1)*is_seq(n-i281,i281-i282))
        
        return(is_seq(0,n-i281)*iota_coef[n-i281]*is_integer(n-i281)*py_sum(sum_arg_84,0,i281))
    
    def sum_arg_83(i280):
        # Child args for sum_arg_83
        return(diff(Y_coef_cp[i280],'chi',1)*diff(Y_coef_cp[n-i280],'phi',2)+diff(Y_coef_cp[i280],'chi',1,'phi',1)*diff(Y_coef_cp[n-i280],'phi',1))
    
    def sum_arg_82(i280):
        # Child args for sum_arg_82
        return(diff(Y_coef_cp[i280],'chi',2)*diff(Y_coef_cp[n-i280],'phi',1)+diff(Y_coef_cp[i280],'chi',1)*diff(Y_coef_cp[n-i280],'chi',1,'phi',1))
    
    def sum_arg_81(i280):
        # Child args for sum_arg_81
        return(diff(Y_coef_cp[i280],'chi',1)*diff(Y_coef_cp[n-i280],'phi',1))
    
    def sum_arg_80(i277):
        # Child args for sum_arg_80    
        def sum_arg_79(i278):
            # Child args for sum_arg_79
            return((diff(X_coef_cp[i278],'chi',1)*diff(X_coef_cp[(-n)-i278+2*i277],'chi',2)+diff(X_coef_cp[i278],'chi',2)*diff(X_coef_cp[(-n)-i278+2*i277],'chi',1))*is_seq(n-i277,i277-i278))
        
        return(is_seq(0,n-i277)*iota_coef[n-i277]*is_integer(n-i277)*py_sum(sum_arg_79,0,i277))
    
    def sum_arg_78(i277):
        # Child args for sum_arg_78    
        def sum_arg_77(i278):
            # Child args for sum_arg_77
            return((diff(X_coef_cp[i278],'chi',1)*diff(X_coef_cp[(-n)-i278+2*i277],'chi',1,'phi',1)+diff(X_coef_cp[i278],'chi',1,'phi',1)*diff(X_coef_cp[(-n)-i278+2*i277],'chi',1))*is_seq(n-i277,i277-i278))
        
        return(is_seq(0,n-i277)*iota_coef[n-i277]*is_integer(n-i277)*py_sum(sum_arg_77,0,i277))
    
    def sum_arg_76(i277):
        # Child args for sum_arg_76    
        def sum_arg_75(i278):
            # Child args for sum_arg_75
            return(diff(X_coef_cp[i278],'chi',1)*diff(X_coef_cp[(-n)-i278+2*i277],'chi',1)*is_seq(n-i277,i277-i278))
        
        return(is_seq(0,n-i277)*iota_coef[n-i277]*is_integer(n-i277)*py_sum(sum_arg_75,0,i277))
    
    def sum_arg_74(i276):
        # Child args for sum_arg_74
        return(diff(X_coef_cp[i276],'chi',1)*diff(X_coef_cp[n-i276],'phi',2)+diff(X_coef_cp[i276],'chi',1,'phi',1)*diff(X_coef_cp[n-i276],'phi',1))
    
    def sum_arg_73(i276):
        # Child args for sum_arg_73
        return(diff(X_coef_cp[i276],'chi',2)*diff(X_coef_cp[n-i276],'phi',1)+diff(X_coef_cp[i276],'chi',1)*diff(X_coef_cp[n-i276],'chi',1,'phi',1))
    
    def sum_arg_72(i276):
        # Child args for sum_arg_72
        return(diff(X_coef_cp[i276],'chi',1)*diff(X_coef_cp[n-i276],'phi',1))
    
    def sum_arg_71(i273):
        # Child args for sum_arg_71    
        def sum_arg_70(i274):
            # Child args for sum_arg_70
            return((i274*Z_coef_cp[i274]*diff(Z_coef_cp[(-n)-i274+2*i273],'chi',3)+2*i274*diff(Z_coef_cp[i274],'chi',1)*diff(Z_coef_cp[(-n)-i274+2*i273],'chi',2)+i274*diff(Z_coef_cp[i274],'chi',2)*diff(Z_coef_cp[(-n)-i274+2*i273],'chi',1))*is_seq(n-i273,i273-i274))
        
        return(is_seq(0,n-i273)*iota_coef[n-i273]*is_integer(n-i273)*py_sum(sum_arg_70,0,i273))
    
    def sum_arg_69(i273):
        # Child args for sum_arg_69    
        def sum_arg_68(i274):
            # Child args for sum_arg_68
            return((i274*Z_coef_cp[i274]*diff(Z_coef_cp[(-n)-i274+2*i273],'chi',2,'phi',1)+i274*diff(Z_coef_cp[i274],'phi',1)*diff(Z_coef_cp[(-n)-i274+2*i273],'chi',2)+i274*diff(Z_coef_cp[i274],'chi',1)*diff(Z_coef_cp[(-n)-i274+2*i273],'chi',1,'phi',1)+i274*diff(Z_coef_cp[i274],'chi',1,'phi',1)*diff(Z_coef_cp[(-n)-i274+2*i273],'chi',1))*is_seq(n-i273,i273-i274))
        
        return(is_seq(0,n-i273)*iota_coef[n-i273]*is_integer(n-i273)*py_sum(sum_arg_68,0,i273))
    
    def sum_arg_67(i273):
        # Child args for sum_arg_67    
        def sum_arg_66(i274):
            # Child args for sum_arg_66
            return((i274*Z_coef_cp[i274]*diff(Z_coef_cp[(-n)-i274+2*i273],'chi',2)+i274*diff(Z_coef_cp[i274],'chi',1)*diff(Z_coef_cp[(-n)-i274+2*i273],'chi',1))*is_seq(n-i273,i273-i274))
        
        return(is_seq(0,n-i273)*iota_coef[n-i273]*is_integer(n-i273)*py_sum(sum_arg_66,0,i273))
    
    def sum_arg_65(i271):
        # Child args for sum_arg_65    
        def sum_arg_64(i272):
            # Child args for sum_arg_64
            return((i272*Y_coef_cp[i272]*diff(Y_coef_cp[(-n)-i272+2*i271],'chi',3)+2*i272*diff(Y_coef_cp[i272],'chi',1)*diff(Y_coef_cp[(-n)-i272+2*i271],'chi',2)+i272*diff(Y_coef_cp[i272],'chi',2)*diff(Y_coef_cp[(-n)-i272+2*i271],'chi',1))*is_seq(n-i271,i271-i272))
        
        return(is_seq(0,n-i271)*iota_coef[n-i271]*is_integer(n-i271)*py_sum(sum_arg_64,0,i271))
    
    def sum_arg_63(i271):
        # Child args for sum_arg_63    
        def sum_arg_62(i272):
            # Child args for sum_arg_62
            return((i272*Y_coef_cp[i272]*diff(Y_coef_cp[(-n)-i272+2*i271],'chi',2,'phi',1)+i272*diff(Y_coef_cp[i272],'phi',1)*diff(Y_coef_cp[(-n)-i272+2*i271],'chi',2)+i272*diff(Y_coef_cp[i272],'chi',1)*diff(Y_coef_cp[(-n)-i272+2*i271],'chi',1,'phi',1)+i272*diff(Y_coef_cp[i272],'chi',1,'phi',1)*diff(Y_coef_cp[(-n)-i272+2*i271],'chi',1))*is_seq(n-i271,i271-i272))
        
        return(is_seq(0,n-i271)*iota_coef[n-i271]*is_integer(n-i271)*py_sum(sum_arg_62,0,i271))
    
    def sum_arg_61(i271):
        # Child args for sum_arg_61    
        def sum_arg_60(i272):
            # Child args for sum_arg_60
            return((i272*Y_coef_cp[i272]*diff(Y_coef_cp[(-n)-i272+2*i271],'chi',2)+i272*diff(Y_coef_cp[i272],'chi',1)*diff(Y_coef_cp[(-n)-i272+2*i271],'chi',1))*is_seq(n-i271,i271-i272))
        
        return(is_seq(0,n-i271)*iota_coef[n-i271]*is_integer(n-i271)*py_sum(sum_arg_60,0,i271))
    
    def sum_arg_59(i269):
        # Child args for sum_arg_59    
        def sum_arg_58(i270):
            # Child args for sum_arg_58
            return((i270*X_coef_cp[i270]*diff(X_coef_cp[(-n)-i270+2*i269],'chi',3)+2*i270*diff(X_coef_cp[i270],'chi',1)*diff(X_coef_cp[(-n)-i270+2*i269],'chi',2)+i270*diff(X_coef_cp[i270],'chi',2)*diff(X_coef_cp[(-n)-i270+2*i269],'chi',1))*is_seq(n-i269,i269-i270))
        
        return(is_seq(0,n-i269)*iota_coef[n-i269]*is_integer(n-i269)*py_sum(sum_arg_58,0,i269))
    
    def sum_arg_57(i269):
        # Child args for sum_arg_57    
        def sum_arg_56(i270):
            # Child args for sum_arg_56
            return((i270*X_coef_cp[i270]*diff(X_coef_cp[(-n)-i270+2*i269],'chi',2,'phi',1)+i270*diff(X_coef_cp[i270],'phi',1)*diff(X_coef_cp[(-n)-i270+2*i269],'chi',2)+i270*diff(X_coef_cp[i270],'chi',1)*diff(X_coef_cp[(-n)-i270+2*i269],'chi',1,'phi',1)+i270*diff(X_coef_cp[i270],'chi',1,'phi',1)*diff(X_coef_cp[(-n)-i270+2*i269],'chi',1))*is_seq(n-i269,i269-i270))
        
        return(is_seq(0,n-i269)*iota_coef[n-i269]*is_integer(n-i269)*py_sum(sum_arg_56,0,i269))
    
    def sum_arg_55(i269):
        # Child args for sum_arg_55    
        def sum_arg_54(i270):
            # Child args for sum_arg_54
            return((i270*X_coef_cp[i270]*diff(X_coef_cp[(-n)-i270+2*i269],'chi',2)+i270*diff(X_coef_cp[i270],'chi',1)*diff(X_coef_cp[(-n)-i270+2*i269],'chi',1))*is_seq(n-i269,i269-i270))
        
        return(is_seq(0,n-i269)*iota_coef[n-i269]*is_integer(n-i269)*py_sum(sum_arg_54,0,i269))
    
    def sum_arg_53(i268):
        # Child args for sum_arg_53
        return(i268*diff(Z_coef_cp[i268],'chi',1)*diff(Z_coef_cp[n-i268],'phi',2)+i268*diff(Z_coef_cp[i268],'chi',1,'phi',1)*diff(Z_coef_cp[n-i268],'phi',1)+i268*Z_coef_cp[i268]*diff(Z_coef_cp[n-i268],'chi',1,'phi',2)+i268*diff(Z_coef_cp[i268],'phi',1)*diff(Z_coef_cp[n-i268],'chi',1,'phi',1))
    
    def sum_arg_52(i268):
        # Child args for sum_arg_52
        return(i268*diff(Z_coef_cp[i268],'chi',2)*diff(Z_coef_cp[n-i268],'phi',1)+i268*Z_coef_cp[i268]*diff(Z_coef_cp[n-i268],'chi',2,'phi',1)+2*i268*diff(Z_coef_cp[i268],'chi',1)*diff(Z_coef_cp[n-i268],'chi',1,'phi',1))
    
    def sum_arg_51(i268):
        # Child args for sum_arg_51
        return(i268*diff(Z_coef_cp[i268],'chi',1)*diff(Z_coef_cp[n-i268],'phi',1)+i268*Z_coef_cp[i268]*diff(Z_coef_cp[n-i268],'chi',1,'phi',1))
    
    def sum_arg_50(i266):
        # Child args for sum_arg_50
        return(i266*diff(Y_coef_cp[i266],'chi',1)*diff(Y_coef_cp[n-i266],'phi',2)+i266*diff(Y_coef_cp[i266],'chi',1,'phi',1)*diff(Y_coef_cp[n-i266],'phi',1)+i266*Y_coef_cp[i266]*diff(Y_coef_cp[n-i266],'chi',1,'phi',2)+i266*diff(Y_coef_cp[i266],'phi',1)*diff(Y_coef_cp[n-i266],'chi',1,'phi',1))
    
    def sum_arg_49(i266):
        # Child args for sum_arg_49
        return(i266*diff(Y_coef_cp[i266],'chi',2)*diff(Y_coef_cp[n-i266],'phi',1)+i266*Y_coef_cp[i266]*diff(Y_coef_cp[n-i266],'chi',2,'phi',1)+2*i266*diff(Y_coef_cp[i266],'chi',1)*diff(Y_coef_cp[n-i266],'chi',1,'phi',1))
    
    def sum_arg_48(i266):
        # Child args for sum_arg_48
        return(i266*diff(Y_coef_cp[i266],'chi',1)*diff(Y_coef_cp[n-i266],'phi',1)+i266*Y_coef_cp[i266]*diff(Y_coef_cp[n-i266],'chi',1,'phi',1))
    
    def sum_arg_47(i264):
        # Child args for sum_arg_47
        return(i264*diff(X_coef_cp[i264],'chi',1)*diff(X_coef_cp[n-i264],'phi',2)+i264*diff(X_coef_cp[i264],'chi',1,'phi',1)*diff(X_coef_cp[n-i264],'phi',1)+i264*X_coef_cp[i264]*diff(X_coef_cp[n-i264],'chi',1,'phi',2)+i264*diff(X_coef_cp[i264],'phi',1)*diff(X_coef_cp[n-i264],'chi',1,'phi',1))
    
    def sum_arg_46(i264):
        # Child args for sum_arg_46
        return(i264*diff(X_coef_cp[i264],'chi',2)*diff(X_coef_cp[n-i264],'phi',1)+i264*X_coef_cp[i264]*diff(X_coef_cp[n-i264],'chi',2,'phi',1)+2*i264*diff(X_coef_cp[i264],'chi',1)*diff(X_coef_cp[n-i264],'chi',1,'phi',1))
    
    def sum_arg_45(i264):
        # Child args for sum_arg_45
        return(i264*diff(X_coef_cp[i264],'chi',1)*diff(X_coef_cp[n-i264],'phi',1)+i264*X_coef_cp[i264]*diff(X_coef_cp[n-i264],'chi',1,'phi',1))
    
    def sum_arg_44(i262):
        # Child args for sum_arg_44
        return(i262*diff(X_coef_cp[i262],'chi',1)*diff(Z_coef_cp[n-i262],'phi',1)+i262*X_coef_cp[i262]*diff(Z_coef_cp[n-i262],'chi',1,'phi',1)+i262*diff(X_coef_cp[i262],'phi',1)*diff(Z_coef_cp[n-i262],'chi',1)+i262*diff(X_coef_cp[i262],'chi',1,'phi',1)*Z_coef_cp[n-i262])
    
    def sum_arg_43(i262):
        # Child args for sum_arg_43
        return(i262*X_coef_cp[i262]*diff(Z_coef_cp[n-i262],'chi',2)+2*i262*diff(X_coef_cp[i262],'chi',1)*diff(Z_coef_cp[n-i262],'chi',1)+i262*diff(X_coef_cp[i262],'chi',2)*Z_coef_cp[n-i262])
    
    def sum_arg_42(i262):
        # Child args for sum_arg_42
        return(i262*X_coef_cp[i262]*diff(Z_coef_cp[n-i262],'chi',1)+i262*diff(X_coef_cp[i262],'chi',1)*Z_coef_cp[n-i262])
    
    def sum_arg_41(i256):
        # Child args for sum_arg_41
        return((diff(X_coef_cp[i256],'chi',1)*n-i256*diff(X_coef_cp[i256],'chi',1))*diff(Z_coef_cp[n-i256],'phi',1)+(X_coef_cp[i256]*n-i256*X_coef_cp[i256])*diff(Z_coef_cp[n-i256],'chi',1,'phi',1)+(diff(X_coef_cp[i256],'phi',1)*n-i256*diff(X_coef_cp[i256],'phi',1))*diff(Z_coef_cp[n-i256],'chi',1)+(diff(X_coef_cp[i256],'chi',1,'phi',1)*n-i256*diff(X_coef_cp[i256],'chi',1,'phi',1))*Z_coef_cp[n-i256])
    
    def sum_arg_40(i256):
        # Child args for sum_arg_40
        return((X_coef_cp[i256]*n-i256*X_coef_cp[i256])*diff(Z_coef_cp[n-i256],'chi',2)+(2*diff(X_coef_cp[i256],'chi',1)*n-2*i256*diff(X_coef_cp[i256],'chi',1))*diff(Z_coef_cp[n-i256],'chi',1)+(diff(X_coef_cp[i256],'chi',2)*n-i256*diff(X_coef_cp[i256],'chi',2))*Z_coef_cp[n-i256])
    
    def sum_arg_39(i256):
        # Child args for sum_arg_39
        return((X_coef_cp[i256]*n-i256*X_coef_cp[i256])*diff(Z_coef_cp[n-i256],'chi',1)+(diff(X_coef_cp[i256],'chi',1)*n-i256*diff(X_coef_cp[i256],'chi',1))*Z_coef_cp[n-i256])
    
    def sum_arg_38(i209):
        # Child args for sum_arg_38    
        def sum_arg_37(i210):
            # Child args for sum_arg_37
            return((diff(B_theta_coef_cp[i210],'chi',1)*B_denom_coef_c[(-n)-i210+2*i209]+B_theta_coef_cp[i210]*diff(B_denom_coef_c[(-n)-i210+2*i209],'chi',1))*is_seq(n-i209,i209-i210))
        
        return(is_seq(0,n-i209)*B_alpha_coef[n-i209]*is_integer(n-i209)*py_sum(sum_arg_37,0,i209))
    
    def sum_arg_36(i209):
        # Child args for sum_arg_36    
        def sum_arg_35(i210):
            # Child args for sum_arg_35
            return(diff(B_theta_coef_cp[i210],'phi',1)*B_denom_coef_c[(-n)-i210+2*i209]*is_seq(n-i209,i209-i210))
        
        return(is_seq(0,n-i209)*B_alpha_coef[n-i209]*is_integer(n-i209)*py_sum(sum_arg_35,0,i209))
    
    def sum_arg_34(i209):
        # Child args for sum_arg_34    
        def sum_arg_33(i210):
            # Child args for sum_arg_33
            return(B_theta_coef_cp[i210]*B_denom_coef_c[(-n)-i210+2*i209]*is_seq(n-i209,i209-i210))
        
        return(is_seq(0,n-i209)*B_alpha_coef[n-i209]*is_integer(n-i209)*py_sum(sum_arg_33,0,i209))
    
    def sum_arg_32(i201):
        # Child args for sum_arg_32    
        def sum_arg_31(i202):
            # Child args for sum_arg_31
            return((diff(B_psi_coef_cp[i202],'chi',2)*B_denom_coef_c[(-n)-i202+2*i201+2]+B_psi_coef_cp[i202]*diff(B_denom_coef_c[(-n)-i202+2*i201+2],'chi',2)+2*diff(B_psi_coef_cp[i202],'chi',1)*diff(B_denom_coef_c[(-n)-i202+2*i201+2],'chi',1))*is_seq(n-i201-2,i201-i202))
        
        return(is_seq(0,n-i201-2)*B_alpha_coef[n-i201-2]*is_integer(n-i201-2)*py_sum(sum_arg_31,0,i201))
    
    def sum_arg_30(i201):
        # Child args for sum_arg_30    
        def sum_arg_29(i202):
            # Child args for sum_arg_29
            return((diff(B_psi_coef_cp[i202],'chi',1,'phi',1)*B_denom_coef_c[(-n)-i202+2*i201+2]+diff(B_psi_coef_cp[i202],'phi',1)*diff(B_denom_coef_c[(-n)-i202+2*i201+2],'chi',1))*is_seq(n-i201-2,i201-i202))
        
        return(is_seq(0,n-i201-2)*B_alpha_coef[n-i201-2]*is_integer(n-i201-2)*py_sum(sum_arg_29,0,i201))
    
    def sum_arg_28(i201):
        # Child args for sum_arg_28    
        def sum_arg_27(i202):
            # Child args for sum_arg_27
            return((diff(B_psi_coef_cp[i202],'chi',1)*B_denom_coef_c[(-n)-i202+2*i201+2]+B_psi_coef_cp[i202]*diff(B_denom_coef_c[(-n)-i202+2*i201+2],'chi',1))*is_seq(n-i201-2,i201-i202))
        
        return(is_seq(0,n-i201-2)*B_alpha_coef[n-i201-2]*is_integer(n-i201-2)*py_sum(sum_arg_27,0,i201))
    
    def sum_arg_26(i1061):
        # Child args for sum_arg_26    
        def sum_arg_25(i1062):
            # Child args for sum_arg_25
            return(B_denom_coef_c[i1062]*diff(B_theta_coef_cp[(-n)-i1062+2*i1061],'chi',1)*is_seq(n-i1061,i1061-i1062))
        
        return(is_seq(0,n-i1061)*iota_coef[n-i1061]*is_integer(n-i1061)*py_sum(sum_arg_25,0,i1061))
    
    def sum_arg_24(i1060):
        # Child args for sum_arg_24
        return(B_denom_coef_c[i1060]*diff(B_theta_coef_cp[n-i1060],'phi',1))
    
    def sum_arg_23(i335):
        # Child args for sum_arg_23    
        def sum_arg_21(i336):
            # Child args for sum_arg_21
            return((is_seq(0,(-n)-i336+2*i335)*diff(B_denom_coef_c[i336],'chi',1)*B_theta_coef_cp[(-n)-i336+2*i335]+is_seq(0,(-n)-i336+2*i335)*B_denom_coef_c[i336]*diff(B_theta_coef_cp[(-n)-i336+2*i335],'chi',1))*is_integer((-n)-i336+2*i335)*is_seq((-n)-i336+2*i335,i335-i336))
            
        def sum_arg_22(i336):
            # Child args for sum_arg_22
            return((is_seq(0,(-n)-i336+2*i335)*diff(B_denom_coef_c[i336],'chi',1)*B_theta_coef_cp[(-n)-i336+2*i335]+is_seq(0,(-n)-i336+2*i335)*B_denom_coef_c[i336]*diff(B_theta_coef_cp[(-n)-i336+2*i335],'chi',1))*is_integer((-n)-i336+2*i335)*is_seq((-n)-i336+2*i335,i335-i336))
        
        return(iota_coef[n-i335]*(n*py_sum(sum_arg_22,0,i335)-i335*py_sum(sum_arg_21,0,i335)))
    
    def sum_arg_20(i319):
        # Child args for sum_arg_20    
        def sum_arg_16(i320):
            # Child args for sum_arg_16    
            def sum_arg_14(i318):
                # Child args for sum_arg_14
                return(is_seq(0,(-n)-i320+2*i319-i318)*Delta_coef_cp[i318]*B_theta_coef_cp[(-n)-i320+2*i319-i318]*is_integer((-n)-i320+2*i319-i318)*is_seq((-n)-i320+2*i319-i318,(-i320)+i319-i318))
                
            def sum_arg_15(i318):
                # Child args for sum_arg_15
                return((is_seq(0,(-n)-i320+2*i319-i318)*diff(Delta_coef_cp[i318],'chi',1)*B_theta_coef_cp[(-n)-i320+2*i319-i318]+is_seq(0,(-n)-i320+2*i319-i318)*Delta_coef_cp[i318]*diff(B_theta_coef_cp[(-n)-i320+2*i319-i318],'chi',1))*is_integer((-n)-i320+2*i319-i318)*is_seq((-n)-i320+2*i319-i318,(-i320)+i319-i318))
            
            return(B_denom_coef_c[i320]*py_sum(sum_arg_15,0,i319-i320)+diff(B_denom_coef_c[i320],'chi',1)*py_sum(sum_arg_14,0,i319-i320))
            
        def sum_arg_19(i320):
            # Child args for sum_arg_19    
            def sum_arg_17(i318):
                # Child args for sum_arg_17
                return(is_seq(0,(-n)-i320+2*i319-i318)*Delta_coef_cp[i318]*B_theta_coef_cp[(-n)-i320+2*i319-i318]*is_integer((-n)-i320+2*i319-i318)*is_seq((-n)-i320+2*i319-i318,(-i320)+i319-i318))
                
            def sum_arg_18(i318):
                # Child args for sum_arg_18
                return((is_seq(0,(-n)-i320+2*i319-i318)*diff(Delta_coef_cp[i318],'chi',1)*B_theta_coef_cp[(-n)-i320+2*i319-i318]+is_seq(0,(-n)-i320+2*i319-i318)*Delta_coef_cp[i318]*diff(B_theta_coef_cp[(-n)-i320+2*i319-i318],'chi',1))*is_integer((-n)-i320+2*i319-i318)*is_seq((-n)-i320+2*i319-i318,(-i320)+i319-i318))
            
            return(B_denom_coef_c[i320]*py_sum(sum_arg_18,0,i319-i320)+diff(B_denom_coef_c[i320],'chi',1)*py_sum(sum_arg_17,0,i319-i320))
        
        return(iota_coef[n-i319]*(n*py_sum(sum_arg_19,0,i319)-i319*py_sum(sum_arg_16,0,i319)))
    
    def sum_arg_13(i311):
        # Child args for sum_arg_13
        return((is_seq(0,n-i311)*diff(B_denom_coef_c[2*i311-n],'chi',1)*n-is_seq(0,n-i311)*i311*diff(B_denom_coef_c[2*i311-n],'chi',1))*B_alpha_coef[n-i311]*is_integer(n-i311)*is_seq(n-i311,i311))
    
    def sum_arg_12(i1079):
        # Child args for sum_arg_12    
        def sum_arg_11(i1080):
            # Child args for sum_arg_11    
            def sum_arg_10(i1528):
                # Child args for sum_arg_10
                return(diff(B_denom_coef_c[(-i1528)-i1080+i1079],'chi',1)*Delta_coef_cp[i1528])
            
            return(is_seq(0,(-n)+i1080+i1079)*B_theta_coef_cp[(-n)+i1080+i1079]*is_integer((-n)+i1080+i1079)*is_seq((-n)+i1080+i1079,i1080)*py_sum(sum_arg_10,0,i1079-i1080))
        
        return(iota_coef[n-i1079]*py_sum(sum_arg_11,0,i1079))
    
    def sum_arg_9(i1077):
        # Child args for sum_arg_9    
        def sum_arg_8(i1044):
            # Child args for sum_arg_8
            return(Delta_coef_cp[i1044]*diff(B_denom_coef_c[(-n)+2*i1077-i1044],'chi',1))
        
        return(is_seq(0,n-i1077)*B_alpha_coef[n-i1077]*is_integer(n-i1077)*is_seq(n-i1077,i1077)*py_sum(sum_arg_8,0,2*i1077-n))
    
    def sum_arg_7(i1075):
        # Child args for sum_arg_7    
        def sum_arg_6(i1076):
            # Child args for sum_arg_6    
            def sum_arg_5(i1489):
                # Child args for sum_arg_5    
                def sum_arg_4(i1512):
                    # Child args for sum_arg_4
                    return(B_denom_coef_c[(-i1512)-i1489-i1076+i1075]*B_denom_coef_c[i1512])
                
                return(diff(p_perp_coef_cp[i1489],'chi',1)*py_sum(sum_arg_4,0,(-i1489)-i1076+i1075))
            
            return(is_seq(0,(-n)+i1076+i1075)*B_theta_coef_cp[(-n)+i1076+i1075]*is_integer((-n)+i1076+i1075)*is_seq((-n)+i1076+i1075,i1076)*py_sum(sum_arg_5,0,i1075-i1076))
        
        return(iota_coef[n-i1075]*py_sum(sum_arg_6,0,i1075))
    
    def sum_arg_3(i1073):
        # Child args for sum_arg_3    
        def sum_arg_2(i1040):
            # Child args for sum_arg_2    
            def sum_arg_1(i1036):
                # Child args for sum_arg_1
                return(B_denom_coef_c[i1036]*B_denom_coef_c[(-n)+2*i1073-i1040-i1036])
            
            return(diff(p_perp_coef_cp[i1040],'chi',1)*py_sum(sum_arg_1,0,(-n)+2*i1073-i1040))
        
        return(is_seq(0,n-i1073)*B_alpha_coef[n-i1073]*is_integer(n-i1073)*is_seq(n-i1073,i1073)*py_sum(sum_arg_2,0,2*i1073-n))
    
    
    out = -(((2*Delta_coef_cp[0]-2)*is_seq(0,n)*dl_p*n*is_integer(n)*py_sum_parallel(sum_arg_153,0,n)+(2-2*Delta_coef_cp[0])*is_seq(0,n)*dl_p*n*is_integer(n)*py_sum_parallel(sum_arg_152,0,n)+(2*Delta_coef_cp[0]-2)*is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_151,0,n)+(2-2*Delta_coef_cp[0])*is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_150,0,n))*diff(tau_p,'phi',1)+((2*Delta_coef_cp[0]-2)*iota_coef[0]*is_seq(0,n)*dl_p*n*is_integer(n)*py_sum_parallel(sum_arg_149,0,n)+(2*Delta_coef_cp[0]-2)*is_seq(0,n)*dl_p*n*is_integer(n)*py_sum_parallel(sum_arg_148,0,n)+((2*Delta_coef_cp[0]-2)*is_seq(0,n)*diff(dl_p,'phi',1)+2*is_seq(0,n)*diff(Delta_coef_cp[0],'phi',1)*dl_p)*n*is_integer(n)*py_sum_parallel(sum_arg_147,0,n)+(2-2*Delta_coef_cp[0])*iota_coef[0]*is_seq(0,n)*dl_p*n*is_integer(n)*py_sum_parallel(sum_arg_146,0,n)+(2-2*Delta_coef_cp[0])*is_seq(0,n)*dl_p*n*is_integer(n)*py_sum_parallel(sum_arg_145,0,n)+((2-2*Delta_coef_cp[0])*is_seq(0,n)*diff(dl_p,'phi',1)-2*is_seq(0,n)*diff(Delta_coef_cp[0],'phi',1)*dl_p)*n*is_integer(n)*py_sum_parallel(sum_arg_144,0,n)+(2*Delta_coef_cp[0]-2)*is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_143,0,n)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_142,0,n)+((2*Delta_coef_cp[0]-2)*is_seq(0,n)*diff(dl_p,'phi',1)+2*is_seq(0,n)*diff(Delta_coef_cp[0],'phi',1)*dl_p)*is_integer(n)*py_sum_parallel(sum_arg_141,0,n)+(2-2*Delta_coef_cp[0])*is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_140,0,n)+(2-2*Delta_coef_cp[0])*iota_coef[0]*is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_139,0,n)+((2-2*Delta_coef_cp[0])*is_seq(0,n)*diff(dl_p,'phi',1)-2*is_seq(0,n)*diff(Delta_coef_cp[0],'phi',1)*dl_p)*is_integer(n)*py_sum_parallel(sum_arg_138,0,n))*tau_p+((2-2*Delta_coef_cp[0])*is_seq(0,n)*dl_p*diff(kap_p,'phi',1)+((2-2*Delta_coef_cp[0])*is_seq(0,n)*diff(dl_p,'phi',1)-2*is_seq(0,n)*diff(Delta_coef_cp[0],'phi',1)*dl_p)*kap_p)*n*is_integer(n)*py_sum_parallel(sum_arg_99,0,n)+(2-2*Delta_coef_cp[0])*iota_coef[0]*n*py_sum_parallel(sum_arg_98,ceil(n/2),floor(n))+(2-2*Delta_coef_cp[0])*n*py_sum_parallel(sum_arg_96,ceil(n/2),floor(n))-2*diff(Delta_coef_cp[0],'phi',1)*n*py_sum_parallel(sum_arg_94,ceil(n/2),floor(n))+(2-2*Delta_coef_cp[0])*is_seq(0,n)*n*is_integer(n)*py_sum_parallel(sum_arg_92,0,n)+(2-2*Delta_coef_cp[0])*iota_coef[0]*is_seq(0,n)*n*is_integer(n)*py_sum_parallel(sum_arg_91,0,n)-2*is_seq(0,n)*diff(Delta_coef_cp[0],'phi',1)*n*is_integer(n)*py_sum_parallel(sum_arg_90,0,n)+n*(B_alpha_coef[0]*py_sum_parallel(sum_arg_9,ceil(n/2),floor(n))+2*B_alpha_coef[0]*py_sum_parallel(sum_arg_7,ceil(n/2),floor(n))-2*B_alpha_coef[0]*py_sum_parallel(sum_arg_3,ceil(n/2),floor(n))-B_alpha_coef[0]*py_sum_parallel(sum_arg_12,ceil(n/2),floor(n)))+(2-2*Delta_coef_cp[0])*iota_coef[0]*n*py_sum_parallel(sum_arg_89,ceil(n/2),floor(n))+(2-2*Delta_coef_cp[0])*n*py_sum_parallel(sum_arg_87,ceil(n/2),floor(n))-2*diff(Delta_coef_cp[0],'phi',1)*n*py_sum_parallel(sum_arg_85,ceil(n/2),floor(n))+(2-2*Delta_coef_cp[0])*is_seq(0,n)*n*is_integer(n)*py_sum_parallel(sum_arg_83,0,n)+(2-2*Delta_coef_cp[0])*iota_coef[0]*is_seq(0,n)*n*is_integer(n)*py_sum_parallel(sum_arg_82,0,n)-2*is_seq(0,n)*diff(Delta_coef_cp[0],'phi',1)*n*is_integer(n)*py_sum_parallel(sum_arg_81,0,n)+(2-2*Delta_coef_cp[0])*iota_coef[0]*n*py_sum_parallel(sum_arg_80,ceil(n/2),floor(n))+(2-2*Delta_coef_cp[0])*n*py_sum_parallel(sum_arg_78,ceil(n/2),floor(n))-2*diff(Delta_coef_cp[0],'phi',1)*n*py_sum_parallel(sum_arg_76,ceil(n/2),floor(n))+(2-2*Delta_coef_cp[0])*is_seq(0,n)*n*is_integer(n)*py_sum_parallel(sum_arg_74,0,n)+(2-2*Delta_coef_cp[0])*iota_coef[0]*is_seq(0,n)*n*is_integer(n)*py_sum_parallel(sum_arg_73,0,n)-2*is_seq(0,n)*diff(Delta_coef_cp[0],'phi',1)*n*is_integer(n)*py_sum_parallel(sum_arg_72,0,n)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*py_sum_parallel(sum_arg_71,ceil(n/2),floor(n))+(2*Delta_coef_cp[0]-2)*py_sum_parallel(sum_arg_69,ceil(n/2),floor(n))+2*diff(Delta_coef_cp[0],'phi',1)*py_sum_parallel(sum_arg_67,ceil(n/2),floor(n))+(2*Delta_coef_cp[0]-2)*iota_coef[0]*py_sum_parallel(sum_arg_65,ceil(n/2),floor(n))+(2*Delta_coef_cp[0]-2)*py_sum_parallel(sum_arg_63,ceil(n/2),floor(n))+2*diff(Delta_coef_cp[0],'phi',1)*py_sum_parallel(sum_arg_61,ceil(n/2),floor(n))+(2*Delta_coef_cp[0]-2)*iota_coef[0]*py_sum_parallel(sum_arg_59,ceil(n/2),floor(n))+(2*Delta_coef_cp[0]-2)*py_sum_parallel(sum_arg_57,ceil(n/2),floor(n))+2*diff(Delta_coef_cp[0],'phi',1)*py_sum_parallel(sum_arg_55,ceil(n/2),floor(n))+(2*Delta_coef_cp[0]-2)*is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_53,0,n)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_52,0,n)+2*is_seq(0,n)*diff(Delta_coef_cp[0],'phi',1)*is_integer(n)*py_sum_parallel(sum_arg_51,0,n)+(2*Delta_coef_cp[0]-2)*is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_50,0,n)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_49,0,n)+2*is_seq(0,n)*diff(Delta_coef_cp[0],'phi',1)*is_integer(n)*py_sum_parallel(sum_arg_48,0,n)+(2*Delta_coef_cp[0]-2)*is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_47,0,n)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_46,0,n)+2*is_seq(0,n)*diff(Delta_coef_cp[0],'phi',1)*is_integer(n)*py_sum_parallel(sum_arg_45,0,n)+(2*Delta_coef_cp[0]-2)*is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_44,0,n)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_43,0,n)+((2*Delta_coef_cp[0]-2)*is_seq(0,n)*dl_p*diff(kap_p,'phi',1)+((2*Delta_coef_cp[0]-2)*is_seq(0,n)*diff(dl_p,'phi',1)+2*is_seq(0,n)*diff(Delta_coef_cp[0],'phi',1)*dl_p)*kap_p)*is_integer(n)*py_sum_parallel(sum_arg_42,0,n)+(2-2*Delta_coef_cp[0])*is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_41,0,n)+(2-2*Delta_coef_cp[0])*iota_coef[0]*is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_40,0,n)+((2-2*Delta_coef_cp[0])*is_seq(0,n)*dl_p*diff(kap_p,'phi',1)+((2-2*Delta_coef_cp[0])*is_seq(0,n)*diff(dl_p,'phi',1)-2*is_seq(0,n)*diff(Delta_coef_cp[0],'phi',1)*dl_p)*kap_p)*is_integer(n)*py_sum_parallel(sum_arg_39,0,n)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*n*py_sum_parallel(sum_arg_38,ceil(n/2),floor(n))+(2*Delta_coef_cp[0]-2)*n*py_sum_parallel(sum_arg_36,ceil(n/2),floor(n))+2*diff(Delta_coef_cp[0],'phi',1)*n*py_sum_parallel(sum_arg_34,ceil(n/2),floor(n))+(4-4*Delta_coef_cp[0])*iota_coef[0]*py_sum_parallel(sum_arg_32,ceil(n/2)-1,floor(n)-2)+(4-4*Delta_coef_cp[0])*py_sum_parallel(sum_arg_30,ceil(n/2)-1,floor(n)-2)-4*diff(Delta_coef_cp[0],'phi',1)*py_sum_parallel(sum_arg_28,ceil(n/2)-1,floor(n)-2)+2*B_alpha_coef[0]*n*py_sum_parallel(sum_arg_26,ceil(n/2),floor(n))+2*B_alpha_coef[0]*is_seq(0,n)*n*is_integer(n)*py_sum_parallel(sum_arg_24,0,n)-4*B_alpha_coef[0]*py_sum_parallel(sum_arg_23,ceil(n/2),floor(n))+4*B_alpha_coef[0]*py_sum_parallel(sum_arg_20,ceil(n/2),floor(n))+4*B_alpha_coef[0]*is_seq(0,n-2)*is_integer(n-2)*py_sum_parallel(sum_arg_137,0,n-2)+4*B_alpha_coef[0]*is_seq(0,n-2)*is_integer(n-2)*py_sum_parallel(sum_arg_134,0,n-2)+2*B_alpha_coef[0]*is_seq(0,n)*n*is_integer(n)*py_sum_parallel(sum_arg_131,0,n)+4*B_alpha_coef[0]*py_sum_parallel(sum_arg_13,ceil(n/2),floor(n))-2*B_alpha_coef[0]*n*py_sum_parallel(sum_arg_128,ceil(n/2),floor(n))-2*B_alpha_coef[0]*is_seq(0,n)*n*is_integer(n)*py_sum_parallel(sum_arg_125,0,n)+4*B_alpha_coef[0]*py_sum_parallel(sum_arg_123,ceil(n/2)-1,floor(n)-2)+4*B_alpha_coef[0]*py_sum_parallel(sum_arg_119,ceil(n/2)-1,floor(n)-2)-4*B_alpha_coef[0]*py_sum_parallel(sum_arg_115,ceil(n/2),floor(n))-4*B_alpha_coef[0]*py_sum_parallel(sum_arg_113,ceil(n/2)-1,floor(n)-2)-4*B_alpha_coef[0]*is_seq(0,n-2)*is_integer(n-2)*py_sum_parallel(sum_arg_111,0,n-2)-B_alpha_coef[0]*py_sum_parallel(sum_arg_110,ceil(n/2),floor(n))+2*B_alpha_coef[0]*py_sum_parallel(sum_arg_108,ceil(n/2),floor(n))+(2*Delta_coef_cp[0]-2)*iota_coef[0]*is_seq(0,n)*dl_p*kap_p*n*is_integer(n)*py_sum_parallel(sum_arg_104,0,n)+(2*Delta_coef_cp[0]-2)*is_seq(0,n)*dl_p*kap_p*n*is_integer(n)*py_sum_parallel(sum_arg_103,0,n)+((2*Delta_coef_cp[0]-2)*is_seq(0,n)*dl_p*diff(kap_p,'phi',1)+((2*Delta_coef_cp[0]-2)*is_seq(0,n)*diff(dl_p,'phi',1)+2*is_seq(0,n)*diff(Delta_coef_cp[0],'phi',1)*dl_p)*kap_p)*n*is_integer(n)*py_sum_parallel(sum_arg_102,0,n)+(2-2*Delta_coef_cp[0])*iota_coef[0]*is_seq(0,n)*dl_p*kap_p*n*is_integer(n)*py_sum_parallel(sum_arg_101,0,n)+(2-2*Delta_coef_cp[0])*is_seq(0,n)*dl_p*kap_p*n*is_integer(n)*py_sum_parallel(sum_arg_100,0,n))/(2*B_alpha_coef[0]*n)
    return(out)

# Evaluates Zn+1. Requires X[..., n], Y[..., n], Z[..., n], 
# B_theta_coef_cp[..., n+1], B_psi_coef_cp[..., n], 
# B_alpha_coef [..., (n-1)/2 or (n-2)/2] 
# iota_coef [..., (n-1)/2 or (n-2)/2] 
# kap_p, dl_p, tau_p
from math import floor, ceil
from math_utilities import *
import chiphifunc
def eval_Znp1_cp(n, X_coef_cp, Y_coef_cp, Z_coef_cp, \
    B_theta_coef_cp, B_psi_coef_cp, \
    B_alpha_coef, \
    kap_p, dl_p, tau_p, iota_coef):    
    def sum_arg_62(i1356):
        # Child args for sum_arg_62    
        def sum_arg_61(i1354):
            # Child args for sum_arg_61
            return(B_theta_coef_cp[i1354]*Y_coef_cp[n-i1356-i1354+2])
        
        return(i1356*Z_coef_cp[i1356]*py_sum(sum_arg_61,0,n-i1356+2))
    
    def sum_arg_60(i1352):
        # Child args for sum_arg_60    
        def sum_arg_59(i1350):
            # Child args for sum_arg_59
            return(B_psi_coef_cp[i1350]*Y_coef_cp[n-i1352-i1350])
        
        return(diff(Z_coef_cp[i1352],'chi',1)*py_sum(sum_arg_59,0,n-i1352))
    
    def sum_arg_58(i1934):
        # Child args for sum_arg_58    
        def sum_arg_57(i1378):
            # Child args for sum_arg_57
            return(B_psi_coef_cp[i1378]*diff(X_coef_cp[n-i1934-i1378],'chi',1))
        
        return(diff(Z_coef_cp[i1934],'phi',1)*py_sum(sum_arg_57,0,n-i1934))
    
    def sum_arg_56(i1928):
        # Child args for sum_arg_56    
        def sum_arg_55(i1382):
            # Child args for sum_arg_55
            return(B_psi_coef_cp[i1382]*diff(X_coef_cp[n-i1928-i1382],'phi',1))
        
        return(diff(Z_coef_cp[i1928],'chi',1)*py_sum(sum_arg_55,0,n-i1928))
    
    def sum_arg_54(i1931):
        # Child args for sum_arg_54    
        def sum_arg_53(i1932):
            # Child args for sum_arg_53
            return(diff(Z_coef_cp[i1932],'chi',1)*X_coef_cp[(-n)-i1932+2*i1931-2]*((-n)-i1932+2*i1931-2)*is_seq(n-i1931+2,i1931-i1932))
        
        return(is_seq(0,n-i1931+2)*B_alpha_coef[n-i1931+2]*is_integer(n-i1931+2)*py_sum(sum_arg_53,0,i1931))
    
    def sum_arg_52(i1930):
        # Child args for sum_arg_52
        return(B_psi_coef_cp[i1930]*diff(X_coef_cp[n-i1930],'chi',1))
    
    def sum_arg_51(i1924):
        # Child args for sum_arg_51    
        def sum_arg_50(i1380):
            # Child args for sum_arg_50
            return(B_theta_coef_cp[i1380]*(n-i1924-i1380+2)*X_coef_cp[n-i1924-i1380+2])
        
        return(diff(Z_coef_cp[i1924],'phi',1)*py_sum(sum_arg_50,0,n-i1924+2))
    
    def sum_arg_49(i1921):
        # Child args for sum_arg_49    
        def sum_arg_48(i1922):
            # Child args for sum_arg_48
            return(diff(X_coef_cp[i1922],'chi',1)*Z_coef_cp[(-n)-i1922+2*i1921-2]*((-n)-i1922+2*i1921-2)*is_seq(n-i1921+2,i1921-i1922))
        
        return(is_seq(0,n-i1921+2)*B_alpha_coef[n-i1921+2]*is_integer(n-i1921+2)*py_sum(sum_arg_48,0,i1921))
    
    def sum_arg_47(i1388):
        # Child args for sum_arg_47    
        def sum_arg_46(i1386):
            # Child args for sum_arg_46
            return(B_theta_coef_cp[i1386]*(n-i1388-i1386+2)*Z_coef_cp[n-i1388-i1386+2])
        
        return(diff(X_coef_cp[i1388],'phi',1)*py_sum(sum_arg_46,0,n-i1388+2))
    
    def sum_arg_45(i1372):
        # Child args for sum_arg_45    
        def sum_arg_44(i1370):
            # Child args for sum_arg_44
            return(B_theta_coef_cp[i1370]*X_coef_cp[n-i1372-i1370+2])
        
        return(i1372*X_coef_cp[i1372]*py_sum(sum_arg_44,0,n-i1372+2))
    
    def sum_arg_43(i1368):
        # Child args for sum_arg_43    
        def sum_arg_42(i1366):
            # Child args for sum_arg_42
            return(B_theta_coef_cp[i1366]*Z_coef_cp[n-i1368-i1366+2])
        
        return(i1368*Z_coef_cp[i1368]*py_sum(sum_arg_42,0,n-i1368+2))
    
    def sum_arg_41(i1364):
        # Child args for sum_arg_41    
        def sum_arg_40(i1362):
            # Child args for sum_arg_40
            return(B_psi_coef_cp[i1362]*X_coef_cp[n-i1364-i1362])
        
        return(diff(X_coef_cp[i1364],'chi',1)*py_sum(sum_arg_40,0,n-i1364))
    
    def sum_arg_39(i1360):
        # Child args for sum_arg_39    
        def sum_arg_38(i1358):
            # Child args for sum_arg_38
            return(B_psi_coef_cp[i1358]*Z_coef_cp[n-i1360-i1358])
        
        return(diff(Z_coef_cp[i1360],'chi',1)*py_sum(sum_arg_38,0,n-i1360))
    
    def sum_arg_37(i1935):
        # Child args for sum_arg_37    
        def sum_arg_36(i1936):
            # Child args for sum_arg_36    
            def sum_arg_35(i2392):
                # Child args for sum_arg_35
                return(diff(X_coef_cp[(-i2392)-i1936+i1935],'chi',1)*i2392*Z_coef_cp[i2392])
            
            return(is_seq(0,(-n)+i1936+i1935-2)*B_theta_coef_cp[(-n)+i1936+i1935-2]*is_integer((-n)+i1936+i1935-2)*is_seq((-n)+i1936+i1935-2,i1936)*py_sum(sum_arg_35,0,i1935-i1936))
        
        return(iota_coef[n-i1935+2]*py_sum(sum_arg_36,0,i1935))
    
    def sum_arg_34(i1925):
        # Child args for sum_arg_34    
        def sum_arg_33(i1926):
            # Child args for sum_arg_33    
            def sum_arg_32(i2376):
                # Child args for sum_arg_32
                return(diff(Z_coef_cp[(-i2376)-i1926+i1925],'chi',1)*i2376*X_coef_cp[i2376])
            
            return(is_seq(0,(-n)+i1926+i1925-2)*B_theta_coef_cp[(-n)+i1926+i1925-2]*is_integer((-n)+i1926+i1925-2)*is_seq((-n)+i1926+i1925-2,i1926)*py_sum(sum_arg_32,0,i1925-i1926))
        
        return(iota_coef[n-i1925+2]*py_sum(sum_arg_33,0,i1925))
    
    def sum_arg_31(i1392):
        # Child args for sum_arg_31
        return(B_theta_coef_cp[i1392]*(n-i1392+2)*X_coef_cp[n-i1392+2])
    
    def sum_arg_30(i1316):
        # Child args for sum_arg_30    
        def sum_arg_29(i1314):
            # Child args for sum_arg_29
            return(B_theta_coef_cp[i1314]*X_coef_cp[n-i1316-i1314+2])
        
        return(i1316*Z_coef_cp[i1316]*py_sum(sum_arg_29,0,n-i1316+2))
    
    def sum_arg_28(i1312):
        # Child args for sum_arg_28    
        def sum_arg_27(i1310):
            # Child args for sum_arg_27
            return(B_psi_coef_cp[i1310]*X_coef_cp[n-i1312-i1310])
        
        return(diff(Z_coef_cp[i1312],'chi',1)*py_sum(sum_arg_27,0,n-i1312))
    
    def sum_arg_26(i1460):
        # Child args for sum_arg_26    
        def sum_arg_25(i1334):
            # Child args for sum_arg_25
            return(B_psi_coef_cp[i1334]*diff(Y_coef_cp[n-i1460-i1334],'phi',1))
        
        return(diff(Z_coef_cp[i1460],'chi',1)*py_sum(sum_arg_25,0,n-i1460))
    
    def sum_arg_24(i1454):
        # Child args for sum_arg_24    
        def sum_arg_23(i1330):
            # Child args for sum_arg_23
            return(B_psi_coef_cp[i1330]*diff(Y_coef_cp[n-i1454-i1330],'chi',1))
        
        return(diff(Z_coef_cp[i1454],'phi',1)*py_sum(sum_arg_23,0,n-i1454))
    
    def sum_arg_22(i1462):
        # Child args for sum_arg_22    
        def sum_arg_21(i1332):
            # Child args for sum_arg_21
            return(B_theta_coef_cp[i1332]*(n-i1462-i1332+2)*Y_coef_cp[n-i1462-i1332+2])
        
        return(diff(Z_coef_cp[i1462],'phi',1)*py_sum(sum_arg_21,0,n-i1462+2))
    
    def sum_arg_20(i1457):
        # Child args for sum_arg_20    
        def sum_arg_19(i1458):
            # Child args for sum_arg_19
            return(diff(Y_coef_cp[i1458],'chi',1)*Z_coef_cp[(-n)-i1458+2*i1457-2]*((-n)-i1458+2*i1457-2)*is_seq(n-i1457+2,i1457-i1458))
        
        return(is_seq(0,n-i1457+2)*B_alpha_coef[n-i1457+2]*is_integer(n-i1457+2)*py_sum(sum_arg_19,0,i1457))
    
    def sum_arg_18(i1456):
        # Child args for sum_arg_18
        return(B_psi_coef_cp[i1456]*diff(Y_coef_cp[n-i1456],'chi',1))
    
    def sum_arg_17(i1449):
        # Child args for sum_arg_17    
        def sum_arg_16(i1450):
            # Child args for sum_arg_16
            return(diff(Z_coef_cp[i1450],'chi',1)*Y_coef_cp[(-n)-i1450+2*i1449-2]*((-n)-i1450+2*i1449-2)*is_seq(n-i1449+2,i1449-i1450))
        
        return(is_seq(0,n-i1449+2)*B_alpha_coef[n-i1449+2]*is_integer(n-i1449+2)*py_sum(sum_arg_16,0,i1449))
    
    def sum_arg_15(i1340):
        # Child args for sum_arg_15    
        def sum_arg_14(i1338):
            # Child args for sum_arg_14
            return(B_theta_coef_cp[i1338]*(n-i1340-i1338+2)*Z_coef_cp[n-i1340-i1338+2])
        
        return(diff(Y_coef_cp[i1340],'phi',1)*py_sum(sum_arg_14,0,n-i1340+2))
    
    def sum_arg_13(i1324):
        # Child args for sum_arg_13    
        def sum_arg_12(i1322):
            # Child args for sum_arg_12
            return(B_theta_coef_cp[i1322]*X_coef_cp[n-i1324-i1322+2])
        
        return(i1324*Y_coef_cp[i1324]*py_sum(sum_arg_12,0,n-i1324+2))
    
    def sum_arg_11(i1320):
        # Child args for sum_arg_11    
        def sum_arg_10(i1318):
            # Child args for sum_arg_10
            return(B_psi_coef_cp[i1318]*X_coef_cp[n-i1320-i1318])
        
        return(diff(Y_coef_cp[i1320],'chi',1)*py_sum(sum_arg_10,0,n-i1320))
    
    def sum_arg_9(i1463):
        # Child args for sum_arg_9    
        def sum_arg_8(i1464):
            # Child args for sum_arg_8    
            def sum_arg_7(i1920):
                # Child args for sum_arg_7
                return(diff(Z_coef_cp[(-i1920)-i1464+i1463],'chi',1)*i1920*Y_coef_cp[i1920])
            
            return(is_seq(0,(-n)+i1464+i1463-2)*B_theta_coef_cp[(-n)+i1464+i1463-2]*is_integer((-n)+i1464+i1463-2)*is_seq((-n)+i1464+i1463-2,i1464)*py_sum(sum_arg_7,0,i1463-i1464))
        
        return(iota_coef[n-i1463+2]*py_sum(sum_arg_8,0,i1463))
    
    def sum_arg_6(i1451):
        # Child args for sum_arg_6    
        def sum_arg_5(i1452):
            # Child args for sum_arg_5    
            def sum_arg_4(i1904):
                # Child args for sum_arg_4
                return(diff(Y_coef_cp[(-i1904)-i1452+i1451],'chi',1)*i1904*Z_coef_cp[i1904])
            
            return(is_seq(0,(-n)+i1452+i1451-2)*B_theta_coef_cp[(-n)+i1452+i1451-2]*is_integer((-n)+i1452+i1451-2)*is_seq((-n)+i1452+i1451-2,i1452)*py_sum(sum_arg_4,0,i1451-i1452))
        
        return(iota_coef[n-i1451+2]*py_sum(sum_arg_5,0,i1451))
    
    def sum_arg_3(i1344):
        # Child args for sum_arg_3
        return(B_theta_coef_cp[i1344]*(n-i1344+2)*Y_coef_cp[n-i1344+2])
    
    def sum_arg_2(i1345):
        # Child args for sum_arg_2
        return(is_seq(0,n-i1345)*diff(X_coef_cp[2*i1345-n],'chi',1)*iota_coef[n-i1345]*is_integer(n-i1345)*is_seq(n-i1345,i1345))
    
    def sum_arg_1(i1393):
        # Child args for sum_arg_1
        return(is_seq(0,n-i1393)*diff(Y_coef_cp[2*i1393-n],'chi',1)*iota_coef[n-i1393]*is_integer(n-i1393)*is_seq(n-i1393,i1393))
    
    
    out = ((-Y_coef_cp[1]*((-(is_seq(0,n+2)*dl_p*is_integer(n+2)*py_sum(sum_arg_62,0,n+2)*tau_p)/2)+is_seq(0,n)*dl_p*is_integer(n)*py_sum(sum_arg_60,0,n)*tau_p-is_seq(0,n)*is_integer(n)*py_sum(sum_arg_58,0,n)+is_seq(0,n)*is_integer(n)*py_sum(sum_arg_56,0,n)-py_sum(sum_arg_54,ceil(0.5*n)+1,floor(n)+2)/2-is_seq(0,n)*dl_p*is_integer(n)*py_sum(sum_arg_52,0,n)+(is_seq(0,n+2)*is_integer(n+2)*py_sum(sum_arg_51,0,n+2))/2+py_sum(sum_arg_49,ceil(0.5*n)+1,floor(n)+2)/2-(is_seq(0,n+2)*is_integer(n+2)*py_sum(sum_arg_47,0,n+2))/2-(is_seq(0,n+2)*dl_p*kap_p*is_integer(n+2)*py_sum(sum_arg_45,0,n+2))/2-(is_seq(0,n+2)*dl_p*kap_p*is_integer(n+2)*py_sum(sum_arg_43,0,n+2))/2+is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum(sum_arg_41,0,n)+is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum(sum_arg_39,0,n)-py_sum(sum_arg_37,ceil(0.5*n)+1,floor(n)+2)/2+py_sum(sum_arg_34,ceil(0.5*n)+1,floor(n)+2)/2+(is_seq(0,n+2)*dl_p*is_integer(n+2)*py_sum(sum_arg_31,0,n+2))/2))-X_coef_cp[1]*((-(is_seq(0,n+2)*dl_p*is_integer(n+2)*py_sum(sum_arg_30,0,n+2)*tau_p)/2)+is_seq(0,n)*dl_p*is_integer(n)*py_sum(sum_arg_28,0,n)*tau_p-py_sum(sum_arg_9,ceil(0.5*n)+1,floor(n)+2)/2+py_sum(sum_arg_6,ceil(0.5*n)+1,floor(n)+2)/2-(is_seq(0,n+2)*dl_p*is_integer(n+2)*py_sum(sum_arg_3,0,n+2))/2-is_seq(0,n)*is_integer(n)*py_sum(sum_arg_26,0,n)+is_seq(0,n)*is_integer(n)*py_sum(sum_arg_24,0,n)-(is_seq(0,n+2)*is_integer(n+2)*py_sum(sum_arg_22,0,n+2))/2-py_sum(sum_arg_20,ceil(0.5*n)+1,floor(n)+2)/2+is_seq(0,n)*dl_p*is_integer(n)*py_sum(sum_arg_18,0,n)+py_sum(sum_arg_17,ceil(0.5*n)+1,floor(n)+2)/2+(is_seq(0,n+2)*is_integer(n+2)*py_sum(sum_arg_15,0,n+2))/2+(is_seq(0,n+2)*dl_p*kap_p*is_integer(n+2)*py_sum(sum_arg_13,0,n+2))/2-is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum(sum_arg_11,0,n))+X_coef_cp[1]*(is_seq(0,n)*dl_p*Y_coef_cp[n]*is_integer(n)*tau_p+py_sum(sum_arg_2,ceil(0.5*n),floor(n))+is_seq(0,n)*is_integer(n)*diff(X_coef_cp[n],'phi',1)+is_seq(0,n)*dl_p*kap_p*Z_coef_cp[n]*is_integer(n))+Y_coef_cp[1]*((-is_seq(0,n)*dl_p*X_coef_cp[n]*is_integer(n)*tau_p)+py_sum(sum_arg_1,ceil(0.5*n),floor(n))+is_seq(0,n)*is_integer(n)*diff(Y_coef_cp[n],'phi',1)))/(dl_p*((-n)-1))
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


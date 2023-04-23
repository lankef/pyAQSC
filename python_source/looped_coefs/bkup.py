from math_utilities import *
from chiphifunc import *


coef_B_psi_dphi_0_dchi_0_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:((((2*Delta_coef_cp[0]-2)*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1)*diff(Y_coef_cp[1],False,1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,2)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(Y_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1,False,1)+(2-2*Delta_coef_cp[0])*diff(X_coef_cp[1],False,1)*(diff(Y_coef_cp[1],True,1))**2+((2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1))*diff(Y_coef_cp[1],True,1))*dl_p*kap_p**2*n_eval+(((2-2*Delta_coef_cp[0])*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],True,2))*diff(Y_coef_cp[1],False,1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,3)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,2,False,1)+((2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],True,1)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],False,1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,2)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(Y_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1,False,1)+(2*Delta_coef_cp[0]-2)*diff(X_coef_cp[1],False,1)*(diff(Y_coef_cp[1],True,1))**2+((4-4*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1))*diff(Y_coef_cp[1],True,1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,3)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,2,False,1))*dl_p*kap_p**2)*diff(tau_p,False,1)+((((2-2*Delta_coef_cp[0])*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1)*diff(Y_coef_cp[1],False,1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,2)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(Y_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1,False,1)+(2*Delta_coef_cp[0]-2)*diff(X_coef_cp[1],False,1)*(diff(Y_coef_cp[1],True,1))**2+((2-2*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1))*diff(Y_coef_cp[1],True,1))*dl_p*kap_p*diff(kap_p,False,1)+((2*Delta_coef_cp[0]-2)*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1)*diff(Y_coef_cp[1],False,2)+((2*Delta_coef_cp[0]-2)*iota_coef[0]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,2)+(2*Delta_coef_cp[0]-2)*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1,False,1)+((4*Delta_coef_cp[0]-4)*iota_coef[0]*diff(X_coef_cp[1],True,2)+(4*Delta_coef_cp[0]-4)*diff(X_coef_cp[1],True,1,False,1)+2*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,1))*diff(Y_coef_cp[1],False,1)+(2-2*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(Y_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,3)+(4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,2,False,1)+(2-2*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*(diff(Y_coef_cp[1],True,2))**2+((4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],True,1,False,1)+((6-6*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],False,1)+(2-2*Delta_coef_cp[0])*iota_coef[0]**2*diff(X_coef_cp[1],True,1)-2*iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(Y_coef_cp[1],True,1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],True,2)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1))*diff(Y_coef_cp[1],True,2)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(Y_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1,False,2)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*(diff(Y_coef_cp[1],True,1,False,1))**2+(((6-6*Delta_coef_cp[0])*diff(X_coef_cp[1],False,1)-2*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(Y_coef_cp[1],True,1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1))*diff(Y_coef_cp[1],True,1,False,1)+((2-2*Delta_coef_cp[0])*diff(X_coef_cp[1],False,2)-2*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],False,1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]**2*diff(X_coef_cp[1],True,2))*(diff(Y_coef_cp[1],True,1))**2+((2*Delta_coef_cp[0]-2)*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],True,3)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2,False,1)+2*iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,2)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,2)+2*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1,False,1))*diff(Y_coef_cp[1],True,1))*dl_p*kap_p**2)*n_eval+(((2*Delta_coef_cp[0]-2)*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],True,2))*diff(Y_coef_cp[1],False,1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,3)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,2,False,1)+((2-2*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],True,1)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],False,1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,2)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(Y_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1,False,1)+(2-2*Delta_coef_cp[0])*diff(X_coef_cp[1],False,1)*(diff(Y_coef_cp[1],True,1))**2+((4*Delta_coef_cp[0]-4)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1))*diff(Y_coef_cp[1],True,1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,3)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,2,False,1))*dl_p*kap_p*diff(kap_p,False,1)+(((2-2*Delta_coef_cp[0])*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],True,2))*diff(Y_coef_cp[1],False,2)+(2-2*Delta_coef_cp[0])*diff(X_coef_cp[1],True,2)*(diff(Y_coef_cp[1],False,1))**2+((2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],True,3)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(Y_coef_cp[1],True,2,False,1)+(2*Delta_coef_cp[0]-2)*diff(X_coef_cp[1],False,1)*diff(Y_coef_cp[1],True,2)+(2-2*Delta_coef_cp[0])*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1,False,1)+((8-8*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],True,2)+(4-4*Delta_coef_cp[0])*diff(X_coef_cp[1],True,1,False,1)-2*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,1)+(6-6*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,3)+(6-6*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],True,2,False,1)-2*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,2))*diff(Y_coef_cp[1],False,1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,4)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,3,False,1)+((4*Delta_coef_cp[0]-4)*iota_coef[0]**2*X_coef_cp[1]*diff(Y_coef_cp[1],True,1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],False,1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)+2*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(Y_coef_cp[1],True,3)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,2,False,2)+((6*Delta_coef_cp[0]-6)*iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],True,1)+(4*Delta_coef_cp[0]-4)*Y_coef_cp[1]*diff(X_coef_cp[1],False,1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)+2*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(Y_coef_cp[1],True,2,False,1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]**2*X_coef_cp[1]*(diff(Y_coef_cp[1],True,2))**2+((4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],True,1,False,1)+((8*Delta_coef_cp[0]-8)*iota_coef[0]*diff(X_coef_cp[1],False,1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]**2*diff(X_coef_cp[1],True,1)+2*iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(Y_coef_cp[1],True,1)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],False,2)+2*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],False,1)+(2-2*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],True,2)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1)+2*iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,2)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(Y_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1,False,2)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*(diff(Y_coef_cp[1],True,1,False,1))**2+(((6*Delta_coef_cp[0]-6)*diff(X_coef_cp[1],False,1)+2*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(Y_coef_cp[1],True,1)+(6-6*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1))*diff(Y_coef_cp[1],True,1,False,1)+((2*Delta_coef_cp[0]-2)*diff(X_coef_cp[1],False,2)+2*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],False,1)+(4-4*Delta_coef_cp[0])*iota_coef[0]**2*diff(X_coef_cp[1],True,2))*(diff(Y_coef_cp[1],True,1))**2+((8-8*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],True,3)+(10-10*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2,False,1)-4*iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,2)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,2)-2*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1,False,1))*diff(Y_coef_cp[1],True,1)+(2-2*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,4)+(4-4*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,3,False,1)-2*iota_coef[0]*Y_coef_cp[1]**2*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,3)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,2,False,2)-2*Y_coef_cp[1]**2*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,2,False,1))*dl_p*kap_p**2)*tau_p+(((Delta_coef_cp[0]-1)*(diff(X_coef_cp[1],True,1))**2*diff(Y_coef_cp[1],False,1)+(1-Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,2)+(1-Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1,False,1)+(1-Delta_coef_cp[0])*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)*diff(Y_coef_cp[1],True,1)+(Delta_coef_cp[0]-1)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,2)+(Delta_coef_cp[0]-1)*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,1,False,1))*kap_p*diff(kap_p,False,2)+((2-2*Delta_coef_cp[0])*(diff(X_coef_cp[1],True,1))**2*diff(Y_coef_cp[1],False,1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,2)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1,False,1)+(2*Delta_coef_cp[0]-2)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)*diff(Y_coef_cp[1],True,1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,2)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,1,False,1))*(diff(kap_p,False,1))**2+((2*Delta_coef_cp[0]-2)*(diff(X_coef_cp[1],True,1))**2*diff(Y_coef_cp[1],False,2)+((4*Delta_coef_cp[0]-4)*iota_coef[0]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,2)+(4*Delta_coef_cp[0]-4)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,1,False,1)+diff(Delta_coef_cp[0],False,1)*(diff(X_coef_cp[1],True,1))**2)*diff(Y_coef_cp[1],False,1)+(2-2*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,3)+(4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,2,False,1)+((4-4*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)+(2-2*Delta_coef_cp[0])*iota_coef[0]**2*(diff(X_coef_cp[1],True,1))**2-iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,2)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1,False,2)+((4-4*Delta_coef_cp[0])*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)-X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,1,False,1)+((2-2*Delta_coef_cp[0])*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,2)-diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]**2*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,2))*diff(Y_coef_cp[1],True,1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,3)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,2,False,1)+iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,2)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,1,False,2)+Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,1,False,1))*kap_p*diff(kap_p,False,1)+((1-Delta_coef_cp[0])*(diff(X_coef_cp[1],True,1))**2*diff(Y_coef_cp[1],False,3)+((3-3*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,2)+(3-3*Delta_coef_cp[0])*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,1,False,1)-diff(Delta_coef_cp[0],False,1)*(diff(X_coef_cp[1],True,1))**2)*diff(Y_coef_cp[1],False,2)+((2-2*Delta_coef_cp[0])*iota_coef[0]**2*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,3)+(4-4*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,2,False,1)-iota_coef[0]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,2)+(2-2*Delta_coef_cp[0])*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,1,False,2)-diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,1,False,1))*diff(Y_coef_cp[1],False,1)+(Delta_coef_cp[0]-1)*iota_coef[0]**3*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,4)+(3*Delta_coef_cp[0]-3)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,3,False,1)+((3*Delta_coef_cp[0]-3)*iota_coef[0]**2*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]**3*(diff(X_coef_cp[1],True,1))**2+iota_coef[0]**2*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,3)+(3*Delta_coef_cp[0]-3)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,2,False,2)+((6*Delta_coef_cp[0]-6)*iota_coef[0]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)+(3*Delta_coef_cp[0]-3)*iota_coef[0]**2*(diff(X_coef_cp[1],True,1))**2+2*iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,2,False,1)+((3*Delta_coef_cp[0]-3)*iota_coef[0]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,2)+2*iota_coef[0]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)+(1-Delta_coef_cp[0])*iota_coef[0]**3*X_coef_cp[1]*diff(X_coef_cp[1],True,3)+(2-2*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,2,False,1)-iota_coef[0]**2*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,2)+(1-Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1,False,2)+((3*Delta_coef_cp[0]-3)*iota_coef[0]**2*diff(X_coef_cp[1],True,1)-iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,1,False,1)+iota_coef[0]**2*diff(Delta_coef_cp[0],False,1)*(diff(X_coef_cp[1],True,1))**2)*diff(Y_coef_cp[1],True,2)+(Delta_coef_cp[0]-1)*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1,False,3)+((3*Delta_coef_cp[0]-3)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)+X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,1,False,2)+((3*Delta_coef_cp[0]-3)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,2)+2*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)+(1-Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,3)+(2-2*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,2,False,1)+((3-3*Delta_coef_cp[0])*iota_coef[0]**2*diff(X_coef_cp[1],True,1)-iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,2)+(1-Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],True,1,False,2)-X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1,False,1))*diff(Y_coef_cp[1],True,1,False,1)+((Delta_coef_cp[0]-1)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,3)+diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,2)+((1-Delta_coef_cp[0])*iota_coef[0]**2*diff(X_coef_cp[1],True,3)+(2-2*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],True,2,False,1)-iota_coef[0]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,2)+(1-Delta_coef_cp[0])*diff(X_coef_cp[1],True,1,False,2)-diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1,False,1))*diff(X_coef_cp[1],False,1)+(2-2*Delta_coef_cp[0])*iota_coef[0]**3*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,3)+(3-3*Delta_coef_cp[0])*iota_coef[0]**2*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,2,False,1)-iota_coef[0]**2*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,2))*diff(Y_coef_cp[1],True,1)+(1-Delta_coef_cp[0])*iota_coef[0]**3*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,4)+(3-3*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,3,False,1)+((Delta_coef_cp[0]-1)*iota_coef[0]**3*Y_coef_cp[1]*diff(X_coef_cp[1],True,2)+(Delta_coef_cp[0]-1)*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1)-iota_coef[0]**2*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(X_coef_cp[1],True,3)+(3-3*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,2,False,2)+((2*Delta_coef_cp[0]-2)*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],True,2)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1)-2*iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(X_coef_cp[1],True,2,False,1)+iota_coef[0]**2*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*(diff(X_coef_cp[1],True,2))**2+((Delta_coef_cp[0]-1)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,2)+2*iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1,False,1))*diff(X_coef_cp[1],True,2)+(1-Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,1,False,3)+((Delta_coef_cp[0]-1)*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1)-Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(X_coef_cp[1],True,1,False,2)+Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*(diff(X_coef_cp[1],True,1,False,1))**2)*kap_p**2)*n_eval+(((1-Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],True,2)+(1-Delta_coef_cp[0])*(diff(X_coef_cp[1],True,1))**2)*diff(Y_coef_cp[1],False,1)+(Delta_coef_cp[0]-1)*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,3)+(Delta_coef_cp[0]-1)*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,2,False,1)+((Delta_coef_cp[0]-1)*X_coef_cp[1]*diff(X_coef_cp[1],False,1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,2)+(Delta_coef_cp[0]-1)*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1,False,1)+((Delta_coef_cp[0]-1)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)+(1-Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,2))*diff(Y_coef_cp[1],True,1)+(1-Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,3)+(1-Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2,False,1)+(1-Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,2)+(1-Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,1,False,1))*kap_p*diff(kap_p,False,2)+(((2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(X_coef_cp[1],True,2)+(2*Delta_coef_cp[0]-2)*(diff(X_coef_cp[1],True,1))**2)*diff(Y_coef_cp[1],False,1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,3)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,2,False,1)+((2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],False,1)+(4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,2)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1,False,1)+((2-2*Delta_coef_cp[0])*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,2))*diff(Y_coef_cp[1],True,1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,3)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2,False,1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,2)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,1,False,1))*(diff(kap_p,False,1))**2+(((2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],True,2)+(2-2*Delta_coef_cp[0])*(diff(X_coef_cp[1],True,1))**2)*diff(Y_coef_cp[1],False,2)+((4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,3)+(4-4*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],True,2,False,1)+((4-4*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],True,1)-X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,2)+(4-4*Delta_coef_cp[0])*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,1,False,1)-diff(Delta_coef_cp[0],False,1)*(diff(X_coef_cp[1],True,1))**2)*diff(Y_coef_cp[1],False,1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]**2*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,4)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,3,False,1)+((4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],False,1)+(6*Delta_coef_cp[0]-6)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,1)+iota_coef[0]*X_coef_cp[1]**2*diff(Delta_coef_cp[0],False,1))*diff(Y_coef_cp[1],True,3)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,2,False,2)+((4*Delta_coef_cp[0]-4)*X_coef_cp[1]*diff(X_coef_cp[1],False,1)+(8*Delta_coef_cp[0]-8)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1)+X_coef_cp[1]**2*diff(Delta_coef_cp[0],False,1))*diff(Y_coef_cp[1],True,2,False,1)+((2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(X_coef_cp[1],False,2)+((4*Delta_coef_cp[0]-4)*iota_coef[0]*diff(X_coef_cp[1],True,1)+X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],False,1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]**2*(diff(X_coef_cp[1],True,1))**2+2*iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,2)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1,False,2)+((4*Delta_coef_cp[0]-4)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)+(4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,2)+X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,1,False,1)+((2*Delta_coef_cp[0]-2)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,2)+diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)+(4-4*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,3)+(4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,2,False,1)+((2-2*Delta_coef_cp[0])*iota_coef[0]**2*diff(X_coef_cp[1],True,1)-iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,2))*diff(Y_coef_cp[1],True,1)+(2-2*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,4)+(4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,3,False,1)+((2-2*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)-iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,3)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2,False,2)+((4-4*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)-X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,2,False,1)-iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,2)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,1,False,2)-Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,1,False,1))*kap_p*diff(kap_p,False,1)+(((Delta_coef_cp[0]-1)*X_coef_cp[1]*diff(X_coef_cp[1],True,2)+(Delta_coef_cp[0]-1)*(diff(X_coef_cp[1],True,1))**2)*diff(Y_coef_cp[1],False,3)+((3*Delta_coef_cp[0]-3)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,3)+(3*Delta_coef_cp[0]-3)*X_coef_cp[1]*diff(X_coef_cp[1],True,2,False,1)+((3*Delta_coef_cp[0]-3)*iota_coef[0]*diff(X_coef_cp[1],True,1)+X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,2)+(3*Delta_coef_cp[0]-3)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,1,False,1)+diff(Delta_coef_cp[0],False,1)*(diff(X_coef_cp[1],True,1))**2)*diff(Y_coef_cp[1],False,2)+((1-Delta_coef_cp[0])*diff(X_coef_cp[1],True,2)*diff(X_coef_cp[1],False,2)-diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,2)*diff(X_coef_cp[1],False,1)+(3*Delta_coef_cp[0]-3)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,4)+(6*Delta_coef_cp[0]-6)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,3,False,1)+((2*Delta_coef_cp[0]-2)*iota_coef[0]**2*diff(X_coef_cp[1],True,1)+2*iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,3)+(3*Delta_coef_cp[0]-3)*X_coef_cp[1]*diff(X_coef_cp[1],True,2,False,2)+((4*Delta_coef_cp[0]-4)*iota_coef[0]*diff(X_coef_cp[1],True,1)+2*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,2,False,1)+(1-Delta_coef_cp[0])*iota_coef[0]**2*(diff(X_coef_cp[1],True,2))**2+(2-2*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],True,1,False,1)*diff(X_coef_cp[1],True,2)+(2*Delta_coef_cp[0]-2)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,1,False,2)+diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,1,False,1))*diff(Y_coef_cp[1],False,1)+(1-Delta_coef_cp[0])*iota_coef[0]**3*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,5)+(3-3*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,4,False,1)+((3-3*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],False,1)+(4-4*Delta_coef_cp[0])*iota_coef[0]**3*X_coef_cp[1]*diff(X_coef_cp[1],True,1)-iota_coef[0]**2*X_coef_cp[1]**2*diff(Delta_coef_cp[0],False,1))*diff(Y_coef_cp[1],True,4)+(3-3*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,3,False,2)+((6-6*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],False,1)+(9-9*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,1)-2*iota_coef[0]*X_coef_cp[1]**2*diff(Delta_coef_cp[0],False,1))*diff(Y_coef_cp[1],True,3,False,1)+((2-2*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],False,2)+((3-3*Delta_coef_cp[0])*iota_coef[0]**2*diff(X_coef_cp[1],True,1)-iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],False,1)+(1-Delta_coef_cp[0])*iota_coef[0]**3*X_coef_cp[1]*diff(X_coef_cp[1],True,2)+(4-4*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1)+(2-2*Delta_coef_cp[0])*iota_coef[0]**3*(diff(X_coef_cp[1],True,1))**2-2*iota_coef[0]**2*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,3)+(1-Delta_coef_cp[0])*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,2,False,3)+((3-3*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],False,1)+(6-6*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1)-X_coef_cp[1]**2*diff(Delta_coef_cp[0],False,1))*diff(Y_coef_cp[1],True,2,False,2)+((2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],False,2)+((6-6*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],True,1)-X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],False,1)+(Delta_coef_cp[0]-1)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,2)+(4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1)+(3-3*Delta_coef_cp[0])*iota_coef[0]**2*(diff(X_coef_cp[1],True,1))**2-3*iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,2,False,1)+((1-Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],False,3)+((Delta_coef_cp[0]-1)*diff(X_coef_cp[1],False,1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],True,1)-X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],False,2)+diff(Delta_coef_cp[0],False,1)*(diff(X_coef_cp[1],False,1))**2+((Delta_coef_cp[0]-1)*iota_coef[0]**2*diff(X_coef_cp[1],True,2)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*diff(X_coef_cp[1],True,1,False,1))*diff(X_coef_cp[1],False,1)+(3*Delta_coef_cp[0]-3)*iota_coef[0]**3*X_coef_cp[1]*diff(X_coef_cp[1],True,3)+(2*Delta_coef_cp[0]-2)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,2,False,1)+((Delta_coef_cp[0]-1)*iota_coef[0]**3*diff(X_coef_cp[1],True,1)+iota_coef[0]**2*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,2)+(2-2*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1,False,2)+((1-Delta_coef_cp[0])*iota_coef[0]**2*diff(X_coef_cp[1],True,1)-iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,1,False,1))*diff(Y_coef_cp[1],True,2)+(1-Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1,False,3)+((3-3*Delta_coef_cp[0])*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)+(3*Delta_coef_cp[0]-3)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,2)-X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,1,False,2)+((3-3*Delta_coef_cp[0])*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,2)-2*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)+(7*Delta_coef_cp[0]-7)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,3)+(8*Delta_coef_cp[0]-8)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,2,False,1)+((3*Delta_coef_cp[0]-3)*iota_coef[0]**2*diff(X_coef_cp[1],True,1)+3*iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,2)+(Delta_coef_cp[0]-1)*X_coef_cp[1]*diff(X_coef_cp[1],True,1,False,2)+X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1,False,1))*diff(Y_coef_cp[1],True,1,False,1)+((1-Delta_coef_cp[0])*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,3)+((1-Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],True,2)-diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(X_coef_cp[1],False,2)+((Delta_coef_cp[0]-1)*iota_coef[0]**2*diff(X_coef_cp[1],True,3)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*diff(X_coef_cp[1],True,2,False,1)+(Delta_coef_cp[0]-1)*diff(X_coef_cp[1],True,1,False,2)+diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1,False,1))*diff(X_coef_cp[1],False,1)+(3*Delta_coef_cp[0]-3)*iota_coef[0]**3*X_coef_cp[1]*diff(X_coef_cp[1],True,4)+(6*Delta_coef_cp[0]-6)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,3,False,1)+((2*Delta_coef_cp[0]-2)*iota_coef[0]**3*diff(X_coef_cp[1],True,1)+2*iota_coef[0]**2*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,3)+(3*Delta_coef_cp[0]-3)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,2,False,2)+((3*Delta_coef_cp[0]-3)*iota_coef[0]**2*diff(X_coef_cp[1],True,1)+2*iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,2,False,1)+(1-Delta_coef_cp[0])*iota_coef[0]**3*(diff(X_coef_cp[1],True,2))**2+(2-2*Delta_coef_cp[0])*iota_coef[0]**2*diff(X_coef_cp[1],True,1,False,1)*diff(X_coef_cp[1],True,2))*diff(Y_coef_cp[1],True,1)+((1-Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,3)+(1-Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],True,2,False,1))*diff(X_coef_cp[1],False,2)+((-iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,3))-Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,2,False,1))*diff(X_coef_cp[1],False,1)+(Delta_coef_cp[0]-1)*iota_coef[0]**3*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,5)+(3*Delta_coef_cp[0]-3)*iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,4,False,1)+((Delta_coef_cp[0]-1)*iota_coef[0]**3*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)+iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,4)+(3*Delta_coef_cp[0]-3)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,3,False,2)+((3*Delta_coef_cp[0]-3)*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)+2*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,3,False,1)+((2-2*Delta_coef_cp[0])*iota_coef[0]**3*Y_coef_cp[1]*diff(X_coef_cp[1],True,2)+(3-3*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1))*diff(X_coef_cp[1],True,3)+(Delta_coef_cp[0]-1)*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2,False,3)+((3*Delta_coef_cp[0]-3)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)+X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,2,False,2)+((3-3*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],True,2)+(4-4*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1)+iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(X_coef_cp[1],True,2,False,1)-iota_coef[0]**2*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*(diff(X_coef_cp[1],True,2))**2+((1-Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,2)-2*iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1,False,1))*diff(X_coef_cp[1],True,2)+(Delta_coef_cp[0]-1)*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,1,False,3)+((1-Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1)+Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(X_coef_cp[1],True,1,False,2)-Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*(diff(X_coef_cp[1],True,1,False,1))**2)*kap_p**2)/(B_alpha_coef[0]*dl_p*kap_p**3*n_eval**2-B_alpha_coef[0]*dl_p*kap_p**3*n_eval)+((((2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]*(diff(X_coef_cp[1],True,1))**2)*dl_p*diff(kap_p,False,1)+((2-2*Delta_coef_cp[0])*(diff(X_coef_cp[1],True,1))**2*diff(Y_coef_cp[1],False,1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,2)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1,False,1)+((2*Delta_coef_cp[0]-2)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,2)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1)+2*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,1)+(4-4*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,2)+(4-4*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,1,False,1)-2*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*(diff(X_coef_cp[1],True,1))**2)*dl_p*kap_p)*n_eval+((2-2*Delta_coef_cp[0])*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,2)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*(diff(X_coef_cp[1],True,1))**2)*dl_p*diff(kap_p,False,1)+(((2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(X_coef_cp[1],True,2)+(2*Delta_coef_cp[0]-2)*(diff(X_coef_cp[1],True,1))**2)*diff(Y_coef_cp[1],False,1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,3)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,2,False,1)+((4-4*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],False,1)+(6-6*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1)-2*X_coef_cp[1]**2*diff(Delta_coef_cp[0],False,1))*diff(Y_coef_cp[1],True,2)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1,False,1)+((2-2*Delta_coef_cp[0])*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1)-2*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,1)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],True,2)*diff(X_coef_cp[1],False,1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,3)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2,False,1)+((6*Delta_coef_cp[0]-6)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)+2*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,2)+(4*Delta_coef_cp[0]-4)*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,1,False,1)+2*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*(diff(X_coef_cp[1],True,1))**2)*dl_p*kap_p)/(B_alpha_coef[0]*n_eval**2-B_alpha_coef[0]*n_eval)-(2*B_denom_coef_c[0]*iota_coef[0]*diff(Delta_coef_cp[1],True,2)+2*B_denom_coef_c[0]*diff(Delta_coef_cp[1],True,1,False,1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*diff(B_denom_coef_c[1],True,2))/n_eval+(3*diff(Delta_coef_cp[0],False,1)*diff(B_denom_coef_c[1],True,1))/n_eval
coef_B_psi_dphi_0_dchi_0_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:((B_alpha_coef[0]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1)*diff(Y_coef_cp[1],False,1)-B_alpha_coef[0]*iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,2)-B_alpha_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1,False,1)-B_alpha_coef[0]*diff(X_coef_cp[1],False,1)*(diff(Y_coef_cp[1],True,1))**2+(B_alpha_coef[0]*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2)+B_alpha_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1))*diff(Y_coef_cp[1],True,1))*n_eval+((-B_alpha_coef[0]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1))-B_alpha_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2))*diff(Y_coef_cp[1],False,1)+B_alpha_coef[0]*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,3)+B_alpha_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,2,False,1)+(B_alpha_coef[0]*iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],True,1)+B_alpha_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],False,1)+B_alpha_coef[0]*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,2)+B_alpha_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1,False,1)+B_alpha_coef[0]*diff(X_coef_cp[1],False,1)*(diff(Y_coef_cp[1],True,1))**2+((-2*B_alpha_coef[0]*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2))-B_alpha_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1))*diff(Y_coef_cp[1],True,1)-B_alpha_coef[0]*iota_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,3)-B_alpha_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,2,False,1))/(2*dl_p*kap_p*n_eval-2*dl_p*kap_p)

coef_B_psi_dphi_0_dchi_1_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:(-((((2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*(diff(Y_coef_cp[1],True,1))**2+(2-2*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1))*dl_p*kap_p**2*n_eval+((2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],False,1)+(4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,2)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,1,False,1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*(diff(Y_coef_cp[1],True,1))**2+((2-2*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],False,1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,2)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,1,False,1))*dl_p*kap_p**2)*diff(tau_p,False,1)+((((2-2*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*(diff(Y_coef_cp[1],True,1))**2+(2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1))*dl_p*kap_p*diff(kap_p,False,1)+((4-4*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1)*diff(Y_coef_cp[1],False,1)+((6*Delta_coef_cp[0]-6)*iota_coef[0]**2*X_coef_cp[1]*diff(Y_coef_cp[1],True,1)+(2-2*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,2)+((6*Delta_coef_cp[0]-6)*iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],True,1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,1,False,1)+((4*Delta_coef_cp[0]-4)*iota_coef[0]*diff(X_coef_cp[1],False,1)+2*iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*(diff(Y_coef_cp[1],True,1))**2+((4-4*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],True,2)+(4-4*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1)-2*iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,1))*dl_p*kap_p**2)*n_eval+((2-2*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],False,1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,2)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,1,False,1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*(diff(Y_coef_cp[1],True,1))**2+((2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],False,1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,1)+(4-4*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,2)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,1,False,1))*dl_p*kap_p*diff(kap_p,False,1)+((2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],False,2)+(2*Delta_coef_cp[0]-2)*diff(X_coef_cp[1],True,1)*(diff(Y_coef_cp[1],False,1))**2+((4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],True,2)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(Y_coef_cp[1],True,1,False,1)+((2-2*Delta_coef_cp[0])*diff(X_coef_cp[1],False,1)+(6*Delta_coef_cp[0]-6)*iota_coef[0]*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,1)+(12*Delta_coef_cp[0]-12)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2)+(6*Delta_coef_cp[0]-6)*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1)+2*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],False,1)+(6-6*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,3)+(8-8*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,2,False,1)+((10-10*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(Y_coef_cp[1],True,1)+(8-8*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],False,1)+(4-4*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)-4*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(Y_coef_cp[1],True,2)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,1,False,2)+((8-8*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],True,1)+(4-4*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],False,1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)-2*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(Y_coef_cp[1],True,1,False,1)+((6-6*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],False,1)-2*iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*(diff(Y_coef_cp[1],True,1))**2+((2-2*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],False,2)-2*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],False,1)+(14*Delta_coef_cp[0]-14)*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],True,2)+(6*Delta_coef_cp[0]-6)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1)+2*iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,1)+(6*Delta_coef_cp[0]-6)*iota_coef[0]**2*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,3)+(8*Delta_coef_cp[0]-8)*iota_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,2,False,1)+4*iota_coef[0]*Y_coef_cp[1]**2*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,2)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,1,False,2)+2*Y_coef_cp[1]**2*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1,False,1))*dl_p*kap_p**2)*tau_p+(((Delta_coef_cp[0]-1)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1)+(1-Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*(diff(X_coef_cp[1],True,1))**2)*kap_p*diff(kap_p,False,2)+((2-2*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]*(diff(X_coef_cp[1],True,1))**2)*(diff(kap_p,False,1))**2+((4-4*Delta_coef_cp[0])*iota_coef[0]*(diff(X_coef_cp[1],True,1))**2*diff(Y_coef_cp[1],False,1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,2)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1,False,1)+((4*Delta_coef_cp[0]-4)*iota_coef[0]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)+iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,1)+(4-4*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,2)+(4-4*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,1,False,1)-iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*(diff(X_coef_cp[1],True,1))**2)*kap_p*diff(kap_p,False,1)+((3*Delta_coef_cp[0]-3)*iota_coef[0]*(diff(X_coef_cp[1],True,1))**2*diff(Y_coef_cp[1],False,2)+((6*Delta_coef_cp[0]-6)*iota_coef[0]**2*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,2)+(6*Delta_coef_cp[0]-6)*iota_coef[0]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,1,False,1)+2*iota_coef[0]*diff(Delta_coef_cp[0],False,1)*(diff(X_coef_cp[1],True,1))**2)*diff(Y_coef_cp[1],False,1)+(3-3*Delta_coef_cp[0])*iota_coef[0]**3*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,3)+(6-6*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,2,False,1)+((6-6*Delta_coef_cp[0])*iota_coef[0]**2*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)+(3-3*Delta_coef_cp[0])*iota_coef[0]**3*(diff(X_coef_cp[1],True,1))**2-2*iota_coef[0]**2*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,2)+(3-3*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1,False,2)+((6-6*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)-2*iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,1,False,1)+((3-3*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,2)-2*iota_coef[0]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)+(Delta_coef_cp[0]-1)*iota_coef[0]**3*X_coef_cp[1]*diff(X_coef_cp[1],True,3)+(2*Delta_coef_cp[0]-2)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,2,False,1)+((3*Delta_coef_cp[0]-3)*iota_coef[0]**3*diff(X_coef_cp[1],True,1)+iota_coef[0]**2*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,2)+(Delta_coef_cp[0]-1)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1,False,2)+iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1,False,1))*diff(Y_coef_cp[1],True,1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]**3*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,3)+(4*Delta_coef_cp[0]-4)*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,2,False,1)+iota_coef[0]**2*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,2)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,1,False,2)+iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,1,False,1))*kap_p**2)*n_eval+((Delta_coef_cp[0]-1)*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],False,1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,2)+(1-Delta_coef_cp[0])*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,1,False,1)+((1-Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],False,1)+(1-Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2)+(Delta_coef_cp[0]-1)*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1)+(Delta_coef_cp[0]-1)*iota_coef[0]*Y_coef_cp[1]*(diff(X_coef_cp[1],True,1))**2)*kap_p*diff(kap_p,False,2)+((2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],False,1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,2)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,1,False,1)+((2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(X_coef_cp[1],False,1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,1)+(4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*(diff(X_coef_cp[1],True,1))**2)*(diff(kap_p,False,1))**2+((2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],False,2)+((8*Delta_coef_cp[0]-8)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,2)+(4*Delta_coef_cp[0]-4)*X_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*(diff(X_coef_cp[1],True,1))**2+X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],False,1)+(6-6*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,3)+(8-8*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,2,False,1)+((8-8*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],False,1)+(10-10*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,1)-2*iota_coef[0]*X_coef_cp[1]**2*diff(Delta_coef_cp[0],False,1))*diff(Y_coef_cp[1],True,2)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,1,False,2)+((4-4*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],False,1)+(4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1)-X_coef_cp[1]**2*diff(Delta_coef_cp[0],False,1))*diff(Y_coef_cp[1],True,1,False,1)+((2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],False,2)+((4-4*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],True,1)-X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],False,1)+(6*Delta_coef_cp[0]-6)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,2)-iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,1)+(6*Delta_coef_cp[0]-6)*iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,3)+(8*Delta_coef_cp[0]-8)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2,False,1)+((4*Delta_coef_cp[0]-4)*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)+2*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,2)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,2)+((4*Delta_coef_cp[0]-4)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)+X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,1,False,1)+iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*(diff(X_coef_cp[1],True,1))**2)*kap_p*diff(kap_p,False,1)+((1-Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],False,3)+((6-6*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,2)+(3-3*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1)+(3-3*Delta_coef_cp[0])*iota_coef[0]*(diff(X_coef_cp[1],True,1))**2-X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],False,2)+((Delta_coef_cp[0]-1)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,2)+diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)+(9-9*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,3)+(12-12*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,2,False,1)+((5-5*Delta_coef_cp[0])*iota_coef[0]**2*diff(X_coef_cp[1],True,1)-4*iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,2)+(3-3*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],True,1,False,2)+((4-4*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],True,1)-2*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,1,False,1)-iota_coef[0]*diff(Delta_coef_cp[0],False,1)*(diff(X_coef_cp[1],True,1))**2)*diff(Y_coef_cp[1],False,1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]**3*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,4)+(9*Delta_coef_cp[0]-9)*iota_coef[0]**2*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,3,False,1)+((9*Delta_coef_cp[0]-9)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],False,1)+(11*Delta_coef_cp[0]-11)*iota_coef[0]**3*X_coef_cp[1]*diff(X_coef_cp[1],True,1)+3*iota_coef[0]**2*X_coef_cp[1]**2*diff(Delta_coef_cp[0],False,1))*diff(Y_coef_cp[1],True,3)+(6*Delta_coef_cp[0]-6)*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,2,False,2)+((12*Delta_coef_cp[0]-12)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],False,1)+(15*Delta_coef_cp[0]-15)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,1)+4*iota_coef[0]*X_coef_cp[1]**2*diff(Delta_coef_cp[0],False,1))*diff(Y_coef_cp[1],True,2,False,1)+((4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],False,2)+((6*Delta_coef_cp[0]-6)*iota_coef[0]**2*diff(X_coef_cp[1],True,1)+2*iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],False,1)+(2-2*Delta_coef_cp[0])*iota_coef[0]**3*X_coef_cp[1]*diff(X_coef_cp[1],True,2)+(5*Delta_coef_cp[0]-5)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1)+(3*Delta_coef_cp[0]-3)*iota_coef[0]**3*(diff(X_coef_cp[1],True,1))**2+3*iota_coef[0]**2*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,2)+(Delta_coef_cp[0]-1)*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,1,False,3)+((3*Delta_coef_cp[0]-3)*X_coef_cp[1]*diff(X_coef_cp[1],False,1)+(3*Delta_coef_cp[0]-3)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1)+X_coef_cp[1]**2*diff(Delta_coef_cp[0],False,1))*diff(Y_coef_cp[1],True,1,False,2)+((2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(X_coef_cp[1],False,2)+((6*Delta_coef_cp[0]-6)*iota_coef[0]*diff(X_coef_cp[1],True,1)+X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],False,1)+(10-10*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,2)+(2-2*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1)+iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,1,False,1)+((Delta_coef_cp[0]-1)*X_coef_cp[1]*diff(X_coef_cp[1],False,3)+((1-Delta_coef_cp[0])*diff(X_coef_cp[1],False,1)+(3*Delta_coef_cp[0]-3)*iota_coef[0]*diff(X_coef_cp[1],True,1)+X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],False,2)-diff(Delta_coef_cp[0],False,1)*(diff(X_coef_cp[1],False,1))**2+((1-Delta_coef_cp[0])*iota_coef[0]**2*diff(X_coef_cp[1],True,2)+(2-2*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],True,1,False,1)+iota_coef[0]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(X_coef_cp[1],False,1)+(9-9*Delta_coef_cp[0])*iota_coef[0]**3*X_coef_cp[1]*diff(X_coef_cp[1],True,3)+(11-11*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,2,False,1)+((3-3*Delta_coef_cp[0])*iota_coef[0]**3*diff(X_coef_cp[1],True,1)-4*iota_coef[0]**2*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,2)+(1-Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1,False,2)-iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1,False,1))*diff(Y_coef_cp[1],True,1)+((2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2)+(Delta_coef_cp[0]-1)*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1))*diff(X_coef_cp[1],False,2)+(2*iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,2)+Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1,False,1))*diff(X_coef_cp[1],False,1)+(4-4*Delta_coef_cp[0])*iota_coef[0]**3*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,4)+(9-9*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,3,False,1)+((2-2*Delta_coef_cp[0])*iota_coef[0]**3*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)-3*iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,3)+(6-6*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2,False,2)+((4-4*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)-4*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,2,False,1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]**3*Y_coef_cp[1]*(diff(X_coef_cp[1],True,2))**2+((5*Delta_coef_cp[0]-5)*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1)+iota_coef[0]**2*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(X_coef_cp[1],True,2)+(1-Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,3)+((2-2*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)-X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,1,False,2)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]*(diff(X_coef_cp[1],True,1,False,1))**2)*kap_p**2)/(B_alpha_coef[0]*dl_p*kap_p**3*n_eval**2-B_alpha_coef[0]*dl_p*kap_p**3*n_eval))+(((2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*(diff(X_coef_cp[1],True,1))**2)*dl_p*kap_p*n_eval+((2-2*Delta_coef_cp[0])*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,1)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1))*dl_p*diff(kap_p,False,1)+((2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],False,1)+(4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,2)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,1,False,1)+((4-4*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],False,1)+(4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1)-2*X_coef_cp[1]**2*diff(Delta_coef_cp[0],False,1))*diff(Y_coef_cp[1],True,1)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*Y_coef_cp[1]*(diff(X_coef_cp[1],True,1))**2+2*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*dl_p*kap_p)/(B_alpha_coef[0]*n_eval**2-B_alpha_coef[0]*n_eval)+((4*Delta_coef_cp[0]-4)*iota_coef[0]*diff(B_denom_coef_c[1],True,1)*n_eval+(4-4*Delta_coef_cp[0])*iota_coef[0]*diff(B_denom_coef_c[1],True,1)-4*B_denom_coef_c[1]*diff(Delta_coef_cp[0],False,1))/(n_eval**2-n_eval)-((Delta_coef_cp[0]-1)*iota_coef[0]*diff(B_denom_coef_c[1],True,1)*n_eval+(1-Delta_coef_cp[0])*iota_coef[0]*diff(B_denom_coef_c[1],True,1)-B_denom_coef_c[1]*diff(Delta_coef_cp[0],False,1))/(n_eval**2-n_eval)-(2*B_denom_coef_c[0]*diff(Delta_coef_cp[1],False,1)+4*B_denom_coef_c[0]*iota_coef[0]*diff(Delta_coef_cp[1],True,1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*diff(B_denom_coef_c[1],True,1))/n_eval
coef_B_psi_dphi_0_dchi_1_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:-((B_alpha_coef[0]*iota_coef[0]*X_coef_cp[1]*(diff(Y_coef_cp[1],True,1))**2-B_alpha_coef[0]*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1))*n_eval+B_alpha_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],False,1)-2*B_alpha_coef[0]*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,2)-B_alpha_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,1,False,1)-B_alpha_coef[0]*iota_coef[0]*X_coef_cp[1]*(diff(Y_coef_cp[1],True,1))**2+(B_alpha_coef[0]*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)-B_alpha_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],False,1))*diff(Y_coef_cp[1],True,1)+2*B_alpha_coef[0]*iota_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,2)+B_alpha_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,1,False,1))/(2*dl_p*kap_p*n_eval-2*dl_p*kap_p)

coef_B_psi_dphi_0_dchi_2_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:(((2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,1))*dl_p*kap_p**2*diff(tau_p,False,1)+(((2-2*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*(diff(Y_coef_cp[1],True,1))**2+(2*Delta_coef_cp[0]-2)*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1))*dl_p*kap_p**2*n_eval+((2-2*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,1))*dl_p*kap_p*diff(kap_p,False,1)+(((2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],True,1)+(6-6*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],False,1)+(6*Delta_coef_cp[0]-6)*iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,2)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,1,False,1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]**2*X_coef_cp[1]*(diff(Y_coef_cp[1],True,1))**2+((4*Delta_coef_cp[0]-4)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],False,1)+(4-4*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)+2*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(Y_coef_cp[1],True,1)+(6-6*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,2)+(4-4*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,1,False,1)-2*iota_coef[0]*Y_coef_cp[1]**2*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*dl_p*kap_p**2)*tau_p+(((2-2*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]**2*Y_coef_cp[1]*(diff(X_coef_cp[1],True,1))**2)*kap_p*diff(kap_p,False,1)+((3-3*Delta_coef_cp[0])*iota_coef[0]**2*(diff(X_coef_cp[1],True,1))**2*diff(Y_coef_cp[1],False,1)+(3*Delta_coef_cp[0]-3)*iota_coef[0]**3*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,2)+(3*Delta_coef_cp[0]-3)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1,False,1)+((3*Delta_coef_cp[0]-3)*iota_coef[0]**2*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)+iota_coef[0]**2*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,1)+(3-3*Delta_coef_cp[0])*iota_coef[0]**3*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,2)+(3-3*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,1,False,1)-iota_coef[0]**2*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*(diff(X_coef_cp[1],True,1))**2)*kap_p**2)*n_eval+((Delta_coef_cp[0]-1)*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,1)+(1-Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1))*kap_p*diff(kap_p,False,2)+((2-2*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1))*(diff(kap_p,False,1))**2+((4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],False,1)+(6*Delta_coef_cp[0]-6)*iota_coef[0]**2*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,2)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,1,False,1)+((4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],False,1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,1)+iota_coef[0]*X_coef_cp[1]**2*diff(Delta_coef_cp[0],False,1))*diff(Y_coef_cp[1],True,1)+(6-6*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2)+(4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1)+(2-2*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]*(diff(X_coef_cp[1],True,1))**2-iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*kap_p*diff(kap_p,False,1)+((3*Delta_coef_cp[0]-3)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],False,2)+((9*Delta_coef_cp[0]-9)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,2)+(6*Delta_coef_cp[0]-6)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1)+(3*Delta_coef_cp[0]-3)*iota_coef[0]**2*(diff(X_coef_cp[1],True,1))**2+2*iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],False,1)+(6-6*Delta_coef_cp[0])*iota_coef[0]**3*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,3)+(9-9*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,2,False,1)+((9-9*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],False,1)+(9-9*Delta_coef_cp[0])*iota_coef[0]**3*X_coef_cp[1]*diff(X_coef_cp[1],True,1)-3*iota_coef[0]**2*X_coef_cp[1]**2*diff(Delta_coef_cp[0],False,1))*diff(Y_coef_cp[1],True,2)+(3-3*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,1,False,2)+((6-6*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],False,1)+(3-3*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,1)-2*iota_coef[0]*X_coef_cp[1]**2*diff(Delta_coef_cp[0],False,1))*diff(Y_coef_cp[1],True,1,False,1)+((2-2*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],False,2)+((3-3*Delta_coef_cp[0])*iota_coef[0]**2*diff(X_coef_cp[1],True,1)-iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],False,1)+(7*Delta_coef_cp[0]-7)*iota_coef[0]**3*X_coef_cp[1]*diff(X_coef_cp[1],True,2)+(2*Delta_coef_cp[0]-2)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1))*diff(Y_coef_cp[1],True,1)+(1-Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,2)-iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)+(6*Delta_coef_cp[0]-6)*iota_coef[0]**3*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,3)+(9*Delta_coef_cp[0]-9)*iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2,False,1)+((2*Delta_coef_cp[0]-2)*iota_coef[0]**3*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)+3*iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,2)+(3*Delta_coef_cp[0]-3)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,2)+((Delta_coef_cp[0]-1)*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)+2*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,1,False,1))*kap_p**2)/(B_alpha_coef[0]*dl_p*kap_p**3*n_eval**2-B_alpha_coef[0]*dl_p*kap_p**3*n_eval)-(((2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1))*dl_p*kap_p)/(B_alpha_coef[0]*n_eval**2-B_alpha_coef[0]*n_eval)-((4*Delta_coef_cp[0]-4)*iota_coef[0]*B_denom_coef_c[1])/(n_eval**2-n_eval)+((Delta_coef_cp[0]-1)*iota_coef[0]*B_denom_coef_c[1])/(n_eval**2-n_eval)-(2*B_denom_coef_c[0]*iota_coef[0]*Delta_coef_cp[1])/n_eval
coef_B_psi_dphi_0_dchi_2_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:(B_alpha_coef[0]*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,1)-B_alpha_coef[0]*iota_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,1))/(2*dl_p*kap_p*n_eval-2*dl_p*kap_p)

coef_B_psi_dphi_0_dchi_3_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:(((2*Delta_coef_cp[0]-2)*iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,1)+(2-2*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,1))*dl_p*kap_p*tau_p+((Delta_coef_cp[0]-1)*iota_coef[0]**3*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1)+(1-Delta_coef_cp[0])*iota_coef[0]**3*Y_coef_cp[1]*(diff(X_coef_cp[1],True,1))**2)*kap_p*n_eval+((2*Delta_coef_cp[0]-2)*iota_coef[0]**2*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,1)+(2-2*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1))*diff(kap_p,False,1)+((3*Delta_coef_cp[0]-3)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],False,1)+(4-4*Delta_coef_cp[0])*iota_coef[0]**3*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,2)+(3-3*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,1,False,1)+((3-3*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],False,1)+(1-Delta_coef_cp[0])*iota_coef[0]**3*X_coef_cp[1]*diff(X_coef_cp[1],True,1)-iota_coef[0]**2*X_coef_cp[1]**2*diff(Delta_coef_cp[0],False,1))*diff(Y_coef_cp[1],True,1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]**3*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2)+(3*Delta_coef_cp[0]-3)*iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1)+(Delta_coef_cp[0]-1)*iota_coef[0]**3*Y_coef_cp[1]*(diff(X_coef_cp[1],True,1))**2+iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*kap_p)/(B_alpha_coef[0]*dl_p*kap_p**2*n_eval**2-B_alpha_coef[0]*dl_p*kap_p**2*n_eval)
coef_B_psi_dphi_0_dchi_3_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_0_dchi_4_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:-((Delta_coef_cp[0]-1)*iota_coef[0]**3*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,1)+(1-Delta_coef_cp[0])*iota_coef[0]**3*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1))/(B_alpha_coef[0]*dl_p*kap_p*n_eval**2-B_alpha_coef[0]*dl_p*kap_p*n_eval)
coef_B_psi_dphi_0_dchi_4_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_0_dchi_5_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_0_dchi_5_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_0_dchi_6_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_0_dchi_6_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_0_dchi_7_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_0_dchi_7_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_1_dchi_0_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:(-((((2*Delta_coef_cp[0]-2)*X_coef_cp[1]*(diff(Y_coef_cp[1],True,1))**2+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1))*dl_p*kap_p**2*n_eval+((2-2*Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,2)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*(diff(Y_coef_cp[1],True,1))**2+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,2))*dl_p*kap_p**2)*diff(tau_p,False,1)+((((2-2*Delta_coef_cp[0])*X_coef_cp[1]*(diff(Y_coef_cp[1],True,1))**2+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1))*dl_p*kap_p*diff(kap_p,False,1)+((4-4*Delta_coef_cp[0])*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1)*diff(Y_coef_cp[1],False,1)+((6*Delta_coef_cp[0]-6)*iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],True,1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,2)+((6*Delta_coef_cp[0]-6)*X_coef_cp[1]*diff(Y_coef_cp[1],True,1)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,1,False,1)+((4*Delta_coef_cp[0]-4)*diff(X_coef_cp[1],False,1)+2*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*(diff(Y_coef_cp[1],True,1))**2+((4-4*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2)+(4-4*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1)-2*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,1))*dl_p*kap_p**2)*n_eval+((2*Delta_coef_cp[0]-2)*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,2)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*(diff(Y_coef_cp[1],True,1))**2+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,2))*dl_p*kap_p*diff(kap_p,False,1)+(((2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(Y_coef_cp[1],True,2)+(4*Delta_coef_cp[0]-4)*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1)+(6*Delta_coef_cp[0]-6)*Y_coef_cp[1]*diff(X_coef_cp[1],True,2))*diff(Y_coef_cp[1],False,1)+(4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,3)+(4-4*Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,2,False,1)+((8-8*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(Y_coef_cp[1],True,1)+(4-4*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],False,1)+(2-2*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)-2*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(Y_coef_cp[1],True,2)+((6-6*Delta_coef_cp[0])*X_coef_cp[1]*diff(Y_coef_cp[1],True,1)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,1,False,1)+((4-4*Delta_coef_cp[0])*diff(X_coef_cp[1],False,1)-2*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*(diff(Y_coef_cp[1],True,1))**2+((10*Delta_coef_cp[0]-10)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2)+(4*Delta_coef_cp[0]-4)*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1)+2*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,3)+(4*Delta_coef_cp[0]-4)*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,2,False,1)+2*Y_coef_cp[1]**2*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,2))*dl_p*kap_p**2)*tau_p+(((Delta_coef_cp[0]-1)*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1)+(1-Delta_coef_cp[0])*Y_coef_cp[1]*(diff(X_coef_cp[1],True,1))**2)*kap_p*diff(kap_p,False,2)+((2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*(diff(X_coef_cp[1],True,1))**2)*(diff(kap_p,False,1))**2+((4-4*Delta_coef_cp[0])*(diff(X_coef_cp[1],True,1))**2*diff(Y_coef_cp[1],False,1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,2)+(4*Delta_coef_cp[0]-4)*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1,False,1)+((4*Delta_coef_cp[0]-4)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)+X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,1)+(4-4*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,2)+(4-4*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,1,False,1)-Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*(diff(X_coef_cp[1],True,1))**2)*kap_p*diff(kap_p,False,1)+((3*Delta_coef_cp[0]-3)*(diff(X_coef_cp[1],True,1))**2*diff(Y_coef_cp[1],False,2)+((6*Delta_coef_cp[0]-6)*iota_coef[0]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,2)+(6*Delta_coef_cp[0]-6)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,1,False,1)+2*diff(Delta_coef_cp[0],False,1)*(diff(X_coef_cp[1],True,1))**2)*diff(Y_coef_cp[1],False,1)+(3-3*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,3)+(6-6*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,2,False,1)+((6-6*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)+(3-3*Delta_coef_cp[0])*iota_coef[0]**2*(diff(X_coef_cp[1],True,1))**2-2*iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,2)+(3-3*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1,False,2)+((6-6*Delta_coef_cp[0])*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)-2*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,1,False,1)+((3-3*Delta_coef_cp[0])*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,2)-2*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)+(Delta_coef_cp[0]-1)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,3)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,2,False,1)+((3*Delta_coef_cp[0]-3)*iota_coef[0]**2*diff(X_coef_cp[1],True,1)+iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,2)+(Delta_coef_cp[0]-1)*X_coef_cp[1]*diff(X_coef_cp[1],True,1,False,2)+X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1,False,1))*diff(Y_coef_cp[1],True,1)+(2*Delta_coef_cp[0]-2)*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,3)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,2,False,1)+iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,2)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,1,False,2)+Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,1,False,1))*kap_p**2)*n_eval+((1-Delta_coef_cp[0])*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,2)+(1-Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1)+(Delta_coef_cp[0]-1)*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2)+(Delta_coef_cp[0]-1)*Y_coef_cp[1]*(diff(X_coef_cp[1],True,1))**2)*kap_p*diff(kap_p,False,2)+((2*Delta_coef_cp[0]-2)*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,2)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]*(diff(X_coef_cp[1],True,1))**2)*(diff(kap_p,False,1))**2+(((4*Delta_coef_cp[0]-4)*X_coef_cp[1]*diff(X_coef_cp[1],True,2)+(4*Delta_coef_cp[0]-4)*(diff(X_coef_cp[1],True,1))**2)*diff(Y_coef_cp[1],False,1)+(4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,3)+(4-4*Delta_coef_cp[0])*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,2,False,1)+((4-4*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],False,1)+(8-8*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1)-X_coef_cp[1]**2*diff(Delta_coef_cp[0],False,1))*diff(Y_coef_cp[1],True,2)+(4-4*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1,False,1)+((4-4*Delta_coef_cp[0])*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,2)-X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,3)+(4*Delta_coef_cp[0]-4)*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2,False,1)+((4*Delta_coef_cp[0]-4)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)+X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,2)+(4*Delta_coef_cp[0]-4)*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,1,False,1)+Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*(diff(X_coef_cp[1],True,1))**2)*kap_p*diff(kap_p,False,1)+(((3-3*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],True,2)+(3-3*Delta_coef_cp[0])*(diff(X_coef_cp[1],True,1))**2)*diff(Y_coef_cp[1],False,2)+((6-6*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,3)+(6-6*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],True,2,False,1)+((6-6*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],True,1)-2*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,2)+(6-6*Delta_coef_cp[0])*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,1,False,1)-2*diff(Delta_coef_cp[0],False,1)*(diff(X_coef_cp[1],True,1))**2)*diff(Y_coef_cp[1],False,1)+(3*Delta_coef_cp[0]-3)*iota_coef[0]**2*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,4)+(6*Delta_coef_cp[0]-6)*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,3,False,1)+((6*Delta_coef_cp[0]-6)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],False,1)+(9*Delta_coef_cp[0]-9)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,1)+2*iota_coef[0]*X_coef_cp[1]**2*diff(Delta_coef_cp[0],False,1))*diff(Y_coef_cp[1],True,3)+(3*Delta_coef_cp[0]-3)*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,2,False,2)+((6*Delta_coef_cp[0]-6)*X_coef_cp[1]*diff(X_coef_cp[1],False,1)+(12*Delta_coef_cp[0]-12)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1)+2*X_coef_cp[1]**2*diff(Delta_coef_cp[0],False,1))*diff(Y_coef_cp[1],True,2,False,1)+((2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(X_coef_cp[1],False,2)+((6*Delta_coef_cp[0]-6)*iota_coef[0]*diff(X_coef_cp[1],True,1)+X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],False,1)+(1-Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,2)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1)+(3*Delta_coef_cp[0]-3)*iota_coef[0]**2*(diff(X_coef_cp[1],True,1))**2+3*iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,2)+(3*Delta_coef_cp[0]-3)*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1,False,2)+((6*Delta_coef_cp[0]-6)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)+(6-6*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,2)+2*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,1,False,1)+((3*Delta_coef_cp[0]-3)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,2)+2*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)+(7-7*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,3)+(8-8*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,2,False,1)+((3-3*Delta_coef_cp[0])*iota_coef[0]**2*diff(X_coef_cp[1],True,1)-3*iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,2)+(1-Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],True,1,False,2)-X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1,False,1))*diff(Y_coef_cp[1],True,1)+(Delta_coef_cp[0]-1)*Y_coef_cp[1]*diff(X_coef_cp[1],True,2)*diff(X_coef_cp[1],False,2)+Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,2)*diff(X_coef_cp[1],False,1)+(3-3*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,4)+(6-6*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,3,False,1)+((2-2*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)-2*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,3)+(3-3*Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2,False,2)+((4-4*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)-2*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,2,False,1)+(Delta_coef_cp[0]-1)*iota_coef[0]**2*Y_coef_cp[1]*(diff(X_coef_cp[1],True,2))**2+(2*Delta_coef_cp[0]-2)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1)*diff(X_coef_cp[1],True,2)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,1,False,2)-Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,1,False,1))*kap_p**2)/(B_alpha_coef[0]*dl_p*kap_p**3*n_eval**2-B_alpha_coef[0]*dl_p*kap_p**3*n_eval))+(((2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]*(diff(X_coef_cp[1],True,1))**2)*dl_p*kap_p*n_eval+((2-2*Delta_coef_cp[0])*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,2)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*(diff(X_coef_cp[1],True,1))**2)*dl_p*kap_p)/(B_alpha_coef[0]*n_eval**2-B_alpha_coef[0]*n_eval)-(2*B_denom_coef_c[0]*diff(Delta_coef_cp[1],True,1))/n_eval+((4*Delta_coef_cp[0]-4)*diff(B_denom_coef_c[1],True,1))/n_eval-((Delta_coef_cp[0]-1)*diff(B_denom_coef_c[1],True,1))/n_eval
coef_B_psi_dphi_1_dchi_0_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:-((B_alpha_coef[0]*X_coef_cp[1]*(diff(Y_coef_cp[1],True,1))**2-B_alpha_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1))*n_eval-B_alpha_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,2)-B_alpha_coef[0]*X_coef_cp[1]*(diff(Y_coef_cp[1],True,1))**2+B_alpha_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1)+B_alpha_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,2))/(2*dl_p*kap_p*n_eval-2*dl_p*kap_p)

coef_B_psi_dphi_1_dchi_1_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:(((2*Delta_coef_cp[0]-2)*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,1)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,1))*dl_p*kap_p**2*diff(tau_p,False,1)+(((4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*(diff(Y_coef_cp[1],True,1))**2+(4*Delta_coef_cp[0]-4)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1))*dl_p*kap_p**2*n_eval+((2-2*Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,1)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,1))*dl_p*kap_p*diff(kap_p,False,1)+(((2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(Y_coef_cp[1],True,1)+(6-6*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],False,1)+(8*Delta_coef_cp[0]-8)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,2)+(4*Delta_coef_cp[0]-4)*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,1,False,1)+(6*Delta_coef_cp[0]-6)*iota_coef[0]*X_coef_cp[1]*(diff(Y_coef_cp[1],True,1))**2+((4*Delta_coef_cp[0]-4)*Y_coef_cp[1]*diff(X_coef_cp[1],False,1)+(6-6*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)+2*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(Y_coef_cp[1],True,1)+(8-8*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,2)+(4-4*Delta_coef_cp[0])*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,1,False,1)-2*Y_coef_cp[1]**2*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*dl_p*kap_p**2)*tau_p+(((4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*Y_coef_cp[1]*(diff(X_coef_cp[1],True,1))**2)*kap_p*diff(kap_p,False,1)+((6-6*Delta_coef_cp[0])*iota_coef[0]*(diff(X_coef_cp[1],True,1))**2*diff(Y_coef_cp[1],False,1)+(6*Delta_coef_cp[0]-6)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,2)+(6*Delta_coef_cp[0]-6)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1,False,1)+((6*Delta_coef_cp[0]-6)*iota_coef[0]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)+2*iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,1)+(6-6*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,2)+(6-6*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,1,False,1)-2*iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*(diff(X_coef_cp[1],True,1))**2)*kap_p**2)*n_eval+((Delta_coef_cp[0]-1)*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,1)+(1-Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1))*kap_p*diff(kap_p,False,2)+((2-2*Delta_coef_cp[0])*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,1)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1))*(diff(kap_p,False,1))**2+((4-4*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],False,1)+(8*Delta_coef_cp[0]-8)*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,2)+(4*Delta_coef_cp[0]-4)*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,1,False,1)+((4*Delta_coef_cp[0]-4)*X_coef_cp[1]*diff(X_coef_cp[1],False,1)+(4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1)+X_coef_cp[1]**2*diff(Delta_coef_cp[0],False,1))*diff(Y_coef_cp[1],True,1)+(8-8*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2)+(4-4*Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1)+(4-4*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*(diff(X_coef_cp[1],True,1))**2-X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*kap_p*diff(kap_p,False,1)+((3*Delta_coef_cp[0]-3)*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],False,2)+((12*Delta_coef_cp[0]-12)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,2)+(6*Delta_coef_cp[0]-6)*X_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1)+(6*Delta_coef_cp[0]-6)*iota_coef[0]*(diff(X_coef_cp[1],True,1))**2+2*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],False,1)+(9-9*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,3)+(12-12*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,2,False,1)+((12-12*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],False,1)+(15-15*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,1)-4*iota_coef[0]*X_coef_cp[1]**2*diff(Delta_coef_cp[0],False,1))*diff(Y_coef_cp[1],True,2)+(3-3*Delta_coef_cp[0])*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,1,False,2)+((6-6*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],False,1)+(6-6*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1)-2*X_coef_cp[1]**2*diff(Delta_coef_cp[0],False,1))*diff(Y_coef_cp[1],True,1,False,1)+((2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],False,2)+((6-6*Delta_coef_cp[0])*iota_coef[0]*diff(X_coef_cp[1],True,1)-X_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],False,1)+(10*Delta_coef_cp[0]-10)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,2)+(2*Delta_coef_cp[0]-2)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1)-iota_coef[0]*X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,1)+(1-Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,2)-Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)+(9*Delta_coef_cp[0]-9)*iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,3)+(12*Delta_coef_cp[0]-12)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2,False,1)+((5*Delta_coef_cp[0]-5)*iota_coef[0]**2*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)+4*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,2)+(3*Delta_coef_cp[0]-3)*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,2)+((4*Delta_coef_cp[0]-4)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)+2*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,1,False,1)+iota_coef[0]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*(diff(X_coef_cp[1],True,1))**2)*kap_p**2)/(B_alpha_coef[0]*dl_p*kap_p**3*n_eval**2-B_alpha_coef[0]*dl_p*kap_p**3*n_eval)-(((2*Delta_coef_cp[0]-2)*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,1)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1))*dl_p*kap_p)/(B_alpha_coef[0]*n_eval**2-B_alpha_coef[0]*n_eval)-((4*Delta_coef_cp[0]-4)*B_denom_coef_c[1])/(n_eval**2-n_eval)+((Delta_coef_cp[0]-1)*B_denom_coef_c[1])/(n_eval**2-n_eval)-(2*B_denom_coef_c[0]*Delta_coef_cp[1])/n_eval
coef_B_psi_dphi_1_dchi_1_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:(B_alpha_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,1)-B_alpha_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,1))/(2*dl_p*kap_p*n_eval-2*dl_p*kap_p)

coef_B_psi_dphi_1_dchi_2_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:(((4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,1)+(4-4*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,1))*dl_p*kap_p*tau_p+((3*Delta_coef_cp[0]-3)*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1)+(3-3*Delta_coef_cp[0])*iota_coef[0]**2*Y_coef_cp[1]*(diff(X_coef_cp[1],True,1))**2)*kap_p*n_eval+((4*Delta_coef_cp[0]-4)*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,1)+(4-4*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1))*diff(kap_p,False,1)+((6*Delta_coef_cp[0]-6)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],False,1)+(9-9*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,2)+(6-6*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,1,False,1)+((6-6*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],False,1)+(3-3*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*diff(X_coef_cp[1],True,1)-2*iota_coef[0]*X_coef_cp[1]**2*diff(Delta_coef_cp[0],False,1))*diff(Y_coef_cp[1],True,1)+(9*Delta_coef_cp[0]-9)*iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2)+(6*Delta_coef_cp[0]-6)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1)+(3*Delta_coef_cp[0]-3)*iota_coef[0]**2*Y_coef_cp[1]*(diff(X_coef_cp[1],True,1))**2+2*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*kap_p)/(B_alpha_coef[0]*dl_p*kap_p**2*n_eval**2-B_alpha_coef[0]*dl_p*kap_p**2*n_eval)
coef_B_psi_dphi_1_dchi_2_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_1_dchi_3_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:-((3*Delta_coef_cp[0]-3)*iota_coef[0]**2*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,1)+(3-3*Delta_coef_cp[0])*iota_coef[0]**2*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1))/(B_alpha_coef[0]*dl_p*kap_p*n_eval**2-B_alpha_coef[0]*dl_p*kap_p*n_eval)
coef_B_psi_dphi_1_dchi_3_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_1_dchi_4_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_1_dchi_4_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_1_dchi_5_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_1_dchi_5_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_1_dchi_6_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_1_dchi_6_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_1_dchi_7_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_1_dchi_7_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_2_dchi_0_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:-((((2*Delta_coef_cp[0]-2)*X_coef_cp[1]*(diff(Y_coef_cp[1],True,1))**2+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1))*dl_p*kap_p*n_eval+((2-2*Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,2)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*(diff(Y_coef_cp[1],True,1))**2+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,2))*dl_p*kap_p)*tau_p+(((2*Delta_coef_cp[0]-2)*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]*(diff(X_coef_cp[1],True,1))**2)*diff(kap_p,False,1)+((3*Delta_coef_cp[0]-3)*(diff(X_coef_cp[1],True,1))**2*diff(Y_coef_cp[1],False,1)+(3-3*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,2)+(3-3*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1,False,1)+((3-3*Delta_coef_cp[0])*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)-X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,1)+(3*Delta_coef_cp[0]-3)*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,2)+(3*Delta_coef_cp[0]-3)*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,1,False,1)+Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*(diff(X_coef_cp[1],True,1))**2)*kap_p)*n_eval+((2-2*Delta_coef_cp[0])*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,2)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1)+(2*Delta_coef_cp[0]-2)*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2)+(2*Delta_coef_cp[0]-2)*Y_coef_cp[1]*(diff(X_coef_cp[1],True,1))**2)*diff(kap_p,False,1)+(((3-3*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],True,2)+(3-3*Delta_coef_cp[0])*(diff(X_coef_cp[1],True,1))**2)*diff(Y_coef_cp[1],False,1)+(3*Delta_coef_cp[0]-3)*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,3)+(3*Delta_coef_cp[0]-3)*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,2,False,1)+((3*Delta_coef_cp[0]-3)*X_coef_cp[1]*diff(X_coef_cp[1],False,1)+(6*Delta_coef_cp[0]-6)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1)+X_coef_cp[1]**2*diff(Delta_coef_cp[0],False,1))*diff(Y_coef_cp[1],True,2)+(3*Delta_coef_cp[0]-3)*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1,False,1)+((3*Delta_coef_cp[0]-3)*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],False,1)+(3-3*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,2)+X_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*diff(Y_coef_cp[1],True,1)+(3-3*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,3)+(3-3*Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2,False,1)+((3-3*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)-X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1))*diff(X_coef_cp[1],True,2)+(3-3*Delta_coef_cp[0])*Y_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(X_coef_cp[1],True,1,False,1)-Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*(diff(X_coef_cp[1],True,1))**2)*kap_p)/(B_alpha_coef[0]*dl_p*kap_p**2*n_eval**2-B_alpha_coef[0]*dl_p*kap_p**2*n_eval)
coef_B_psi_dphi_2_dchi_0_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_2_dchi_1_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:(((2*Delta_coef_cp[0]-2)*X_coef_cp[1]*Y_coef_cp[1]*diff(Y_coef_cp[1],True,1)+(2-2*Delta_coef_cp[0])*Y_coef_cp[1]**2*diff(X_coef_cp[1],True,1))*dl_p*kap_p*tau_p+((3*Delta_coef_cp[0]-3)*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1)+(3-3*Delta_coef_cp[0])*iota_coef[0]*Y_coef_cp[1]*(diff(X_coef_cp[1],True,1))**2)*kap_p*n_eval+((2*Delta_coef_cp[0]-2)*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,1)+(2-2*Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1))*diff(kap_p,False,1)+((3*Delta_coef_cp[0]-3)*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],False,1)+(6-6*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,2)+(3-3*Delta_coef_cp[0])*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,1,False,1)+((3-3*Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],False,1)+(3-3*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*diff(X_coef_cp[1],True,1)-X_coef_cp[1]**2*diff(Delta_coef_cp[0],False,1))*diff(Y_coef_cp[1],True,1)+(6*Delta_coef_cp[0]-6)*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2)+(3*Delta_coef_cp[0]-3)*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1,False,1)+(3*Delta_coef_cp[0]-3)*iota_coef[0]*Y_coef_cp[1]*(diff(X_coef_cp[1],True,1))**2+X_coef_cp[1]*Y_coef_cp[1]*diff(Delta_coef_cp[0],False,1)*diff(X_coef_cp[1],True,1))*kap_p)/(B_alpha_coef[0]*dl_p*kap_p**2*n_eval**2-B_alpha_coef[0]*dl_p*kap_p**2*n_eval)
coef_B_psi_dphi_2_dchi_1_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_2_dchi_2_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:-((3*Delta_coef_cp[0]-3)*iota_coef[0]*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,1)+(3-3*Delta_coef_cp[0])*iota_coef[0]*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1))/(B_alpha_coef[0]*dl_p*kap_p*n_eval**2-B_alpha_coef[0]*dl_p*kap_p*n_eval)
coef_B_psi_dphi_2_dchi_2_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_2_dchi_3_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_2_dchi_3_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_2_dchi_4_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_2_dchi_4_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_2_dchi_5_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_2_dchi_5_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_2_dchi_6_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_2_dchi_6_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_2_dchi_7_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_2_dchi_7_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_3_dchi_0_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:(((Delta_coef_cp[0]-1)*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1)+(1-Delta_coef_cp[0])*Y_coef_cp[1]*(diff(X_coef_cp[1],True,1))**2)*n_eval+(1-Delta_coef_cp[0])*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,2)+(1-Delta_coef_cp[0])*X_coef_cp[1]*diff(X_coef_cp[1],True,1)*diff(Y_coef_cp[1],True,1)+(Delta_coef_cp[0]-1)*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,2)+(Delta_coef_cp[0]-1)*Y_coef_cp[1]*(diff(X_coef_cp[1],True,1))**2)/(B_alpha_coef[0]*dl_p*kap_p*n_eval**2-B_alpha_coef[0]*dl_p*kap_p*n_eval)
coef_B_psi_dphi_3_dchi_0_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_3_dchi_1_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:-((Delta_coef_cp[0]-1)*X_coef_cp[1]**2*diff(Y_coef_cp[1],True,1)+(1-Delta_coef_cp[0])*X_coef_cp[1]*Y_coef_cp[1]*diff(X_coef_cp[1],True,1))/(B_alpha_coef[0]*dl_p*kap_p*n_eval**2-B_alpha_coef[0]*dl_p*kap_p*n_eval)
coef_B_psi_dphi_3_dchi_1_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_3_dchi_2_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_3_dchi_2_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_3_dchi_3_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_3_dchi_3_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_3_dchi_4_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_3_dchi_4_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_3_dchi_5_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_3_dchi_5_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_3_dchi_6_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_3_dchi_6_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_3_dchi_7_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_3_dchi_7_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_4_dchi_0_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_4_dchi_0_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_4_dchi_1_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_4_dchi_1_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_4_dchi_2_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_4_dchi_2_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_4_dchi_3_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_4_dchi_3_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_4_dchi_4_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_4_dchi_4_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_4_dchi_5_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_4_dchi_5_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_4_dchi_6_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_4_dchi_6_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_4_dchi_7_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_4_dchi_7_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_5_dchi_0_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_5_dchi_0_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_5_dchi_1_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_5_dchi_1_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_5_dchi_2_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_5_dchi_2_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_5_dchi_3_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_5_dchi_3_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_5_dchi_4_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_5_dchi_4_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_5_dchi_5_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_5_dchi_5_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_5_dchi_6_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_5_dchi_6_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_5_dchi_7_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_5_dchi_7_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_6_dchi_0_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_6_dchi_0_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_6_dchi_1_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_6_dchi_1_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_6_dchi_2_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_6_dchi_2_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_6_dchi_3_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_6_dchi_3_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_6_dchi_4_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_6_dchi_4_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_6_dchi_5_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_6_dchi_5_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_6_dchi_6_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_6_dchi_6_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_6_dchi_7_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_6_dchi_7_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_7_dchi_0_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_7_dchi_0_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_7_dchi_1_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_7_dchi_1_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_7_dchi_2_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_7_dchi_2_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_7_dchi_3_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_7_dchi_3_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_7_dchi_4_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_7_dchi_4_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_7_dchi_5_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_7_dchi_5_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_7_dchi_6_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_7_dchi_6_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

coef_B_psi_dphi_7_dchi_7_all_but_Y = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0
coef_B_psi_dphi_7_dchi_7_in_Y_RHS = lambda n_eval, X_coef_cp, Y_coef_cp,\
    Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
    dl_p, tau_p, kap_p, iota_coef:0

# for tracking B_theta dependence
tensor_fft_op_B_psi_in_all_but_Y = \
    lambda n_eval, X_coef_cp, Y_coef_cp,\
        Delta_coef_cp, B_alpha_coef, B_denom_coef_c,\
        dl_p, tau_p, kap_p, iota_coef,\
        num_mode, cap_axis0, len_tensor, nfp:\
        to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_0_dchi_0_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=0, dchi=0,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_0_dchi_1_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=0, dchi=1,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_0_dchi_2_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=0, dchi=2,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_0_dchi_3_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=0, dchi=3,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_0_dchi_4_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=0, dchi=4,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_0_dchi_5_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=0, dchi=5,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_0_dchi_6_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=0, dchi=6,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_0_dchi_7_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=0, dchi=7,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_1_dchi_0_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=1, dchi=0,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_1_dchi_1_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=1, dchi=1,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_1_dchi_2_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=1, dchi=2,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_1_dchi_3_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=1, dchi=3,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_1_dchi_4_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=1, dchi=4,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_1_dchi_5_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=1, dchi=5,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_1_dchi_6_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=1, dchi=6,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_1_dchi_7_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=1, dchi=7,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_2_dchi_0_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=2, dchi=0,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_2_dchi_1_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=2, dchi=1,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_2_dchi_2_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=2, dchi=2,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_2_dchi_3_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=2, dchi=3,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_2_dchi_4_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=2, dchi=4,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_2_dchi_5_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=2, dchi=5,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_2_dchi_6_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=2, dchi=6,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_2_dchi_7_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=2, dchi=7,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_3_dchi_0_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=3, dchi=0,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_3_dchi_1_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=3, dchi=1,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_3_dchi_2_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=3, dchi=2,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_3_dchi_3_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=3, dchi=3,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_3_dchi_4_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=3, dchi=4,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_3_dchi_5_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=3, dchi=5,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_3_dchi_6_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=3, dchi=6,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_3_dchi_7_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=3, dchi=7,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_4_dchi_0_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=4, dchi=0,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_4_dchi_1_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=4, dchi=1,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_4_dchi_2_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=4, dchi=2,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_4_dchi_3_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=4, dchi=3,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_4_dchi_4_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=4, dchi=4,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_4_dchi_5_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=4, dchi=5,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_4_dchi_6_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=4, dchi=6,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_4_dchi_7_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=4, dchi=7,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_5_dchi_0_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=5, dchi=0,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_5_dchi_1_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=5, dchi=1,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_5_dchi_2_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=5, dchi=2,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_5_dchi_3_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=5, dchi=3,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_5_dchi_4_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=5, dchi=4,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_5_dchi_5_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=5, dchi=5,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_5_dchi_6_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=5, dchi=6,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_5_dchi_7_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=5, dchi=7,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_6_dchi_0_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=6, dchi=0,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_6_dchi_1_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=6, dchi=1,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_6_dchi_2_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=6, dchi=2,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_6_dchi_3_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=6, dchi=3,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_6_dchi_4_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=6, dchi=4,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_6_dchi_5_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=6, dchi=5,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_6_dchi_6_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=6, dchi=6,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_6_dchi_7_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=6, dchi=7,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_7_dchi_0_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=7, dchi=0,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_7_dchi_1_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=7, dchi=1,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_7_dchi_2_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=7, dchi=2,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_7_dchi_3_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=7, dchi=3,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_7_dchi_4_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=7, dchi=4,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_7_dchi_5_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=7, dchi=5,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_7_dchi_6_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=7, dchi=6,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +to_tensor_fft_op_multi_dim(
            coef_B_psi_dphi_7_dchi_7_all_but_Y(
                n_eval=n_eval,
                X_coef_cp=X_coef_cp, Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef, B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p, tau_p=tau_p, kap_p=kap_p, iota_coef=iota_coef
            ),
            dphi=7, dchi=7,
            num_mode=num_mode, cap_axis0=cap_axis0,
            len_tensor=len_tensor, nfp=nfp
        )\
        +0

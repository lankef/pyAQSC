''' Parsed files '''
# Coeff of B_psi terms and B_theta terms they carry in the looped equation.
# from 'Looped study B_theta through B_psi.wxmx'
from .lambda_B_psi_coef import *
# Coeff of B_psi in other variables for expediated evaluation. Also used in .
# from 'Looped study B_theta through B_psi.wxmx'
from .lambda_B_psi_coef_in_vars import *
# The coefficient of B_psi in D3.
from .B_psi_coefs_in_D3 import *
''' Hardcoded files '''
# The coefficient of Y1p in D3. Used in both MHD and magnetic iterations.
from .Y_coefs_in_D3 import *
# The coefficient of Y1p in the looped equation. Used in both MHD and magnetic iterations.
from .Y_coefs import *
# Coefficients of terms directly containing B_theta in the looped equation.
from .direct_B_theta_coefs import *

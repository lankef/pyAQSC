''' Where global settings are defined '''
# APES can run without pyQSC, but can have additional features
#  when pyQSC is available. Setting to True enables:
# chiphifunc.py
# - 'pseudo_spectral' mode for ChiPhiFunc.dphi()
# chiphifunc_test_suite.py
# - import_from_stel(): Loading equilibria from pyQSC
# - read_first_three_orders(): Loading equilibria from ref [1-3]
use_pyQSC = False

''' Numerical settings (chiphifunc.py) '''
double_precision = True
diff_mode = 1 # 1 for fft, 2 for pseudo_spectral
# Currently, It takes ~1hr to compile iterate_2 (most of the time taken by
# looped_solver.py) to order 4 for each combination of nfp and number of
# harmonics. Compiling to higher order may require ~10 hrs.
# However, it offers ~ 20x speed up for each case.
# The compile can be disabled to only compile recursion relations for individual
# variables and save
compile_MHD_iteration = False

''' Caching and Joblib settings '''
# Caching -----
# Maximum size of LRU cache in ChiPhiFunc
max_size = 1000

''' Plotting and output settings (for chiphifunc_test_suite.py) '''
# Number of grid points when:
# 1. Generating random ChiPhiFunc and ChiPhiEpsFunc for testing
# using rand_ChiPhiFunc() rand_ChiPhiEpsFunc()
# 2. Evaluating the values represented by power series or power series
# coefficients using evaluate(lambda), evaluate_ChiPhiFunc(ChiPhiFunc) and
# evaluate_ChiPhiEpsFunc(ChiPhiEpsFunc)
n_grid_phi = 1000
n_grid_chi = 500

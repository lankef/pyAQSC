''' Import settings '''
# APES can run without pyQSC, but can have additional features
#  when pyQSC is available. Setting to True enables:
# chiphifunc.py
# - 'pseudo_spectral' mode for ChiPhiFunc.dphi()
# chiphifunc_test_suite.py
# - import_from_stel(): Loading equilibria from pyQSC
# - read_first_three_orders(): Loading equilibria from ref [1-3]
use_pyQSC = True

''' Numerical settings (chiphifunc.py) '''
# Default numerical methods -----
diff_mode = 'fft' # available: pseudo_spectral, finite_difference, fft, spline
integral_mode = 'fft' # avalable: spline, simpson, fft
two_pi_integral_mode = 'fft' # available: spline, simpson, fft
# Asymptotic series settings -----
# solve_integration_factor() can automatically switch to using asymptotic
# series for solving y'+py=f under integration_mode='auto' when the
# average amplitude of p is greater than a threshold set by this value.
asymptotic_threshold = 30
# When solving y'+py=f with an asymptotic series, stop at this order if the
# optimal truncation is not yet reached.
asymptotic_order = 6

''' Caching and Joblib settings '''
# Caching -----
# Maximum size of LRU cache in ChiPhiFunc
max_size = 1000
# Joblib -----
# By default, the number of threads launched is set
# to #cpu/2.
import multiprocessing
print('Detected', multiprocessing.cpu_count(), 'CPU\'s. Setting n_jobs to #CPU/2.')
# scipy.integrate is based on compiled codes. 'threading' is the best backend.
n_jobs_chiphifunc = multiprocessing.cpu_count()//2 # for integration factor
n_jobs_math_utilities = multiprocessing.cpu_count()//2 # for summations
backend_chiphifunc = 'threading'
backend_math_utilities = 'threading'


''' Plotting and output settings (for chiphifunc_test_suite.py) '''
# Number of grid points when:
# 1. Generating random ChiPhiFunc and ChiPhiEpsFunc for testing
# using rand_ChiPhiFunc() rand_ChiPhiEpsFunc()
# 2. Evaluating the values represented by power series or power series
# coefficients using evaluate(lambda), evaluate_ChiPhiFunc(ChiPhiFunc) and
# evaluate_ChiPhiEpsFunc(ChiPhiEpsFunc)
n_grid_phi = 1000
n_grid_chi = 500

''' Debug mode (chiphifunc.py) '''
# When enabled, all ChiPhiFunc __init__ calls become tracked, and
# can be accessed in the lists chiphifunc.debug_max_value[] and
# chiphifunc.debug_avg_value[].
# WARNING: These lists need TO BE PERIODICALLY PURGED MANUALLY.
# WARNING: Debug mode SUBSTALTIALLY SLOWS DOWN the code by disabling numba
# in various places.
debug_mode = False

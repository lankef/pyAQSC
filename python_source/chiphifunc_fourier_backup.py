''' III. Fourier representation for phi dependence - DEPRECIATED '''

# Implementation of ChiPhiFunc using FULL, exponential Fourier series to represent
# free functions of phi.
# When fourier_mode is enabled during initialization, content would be treated
# as fourier coefficients of format:
# [
#      s_chi_n = [s_k, ..., s_2, s_1, const, c_1, c_2, ..., c_k],
#      ...
#      s_chi_1 = [s_k, ..., s_2, s_1, const, c_1, c_2, ..., c_k],
#      c_chi_1 = [s_k, ..., s_2, s_1, const, c_1, c_2, ..., c_k],
#      ...
#      c_chi_n = [s_k, ..., s_2, s_1, const, c_1, c_2, ..., c_k],
# ]
class ChiPhiFuncFourier(ChiPhiFunc):
    def __init__(self, content, fourier_mode = False):
        super().__init__(content, fourier_mode)
        if content.shape[1]%2 == 0:
            raise ValueError('Phi coefficients should be a full fourier series. Even phi_dim detected.')
    # Operator handlers -------------------------------------------------
    # NOTE: will not be passed items awaiting for conditions.

    # Addition of 2 ChiPhiFuncFourier's.
    # Wrapper for numba method.
    # -- Input: self and another ChiPhiFuncFourier
    # -- Output: a new ChiPhiFuncFourier
    def add_ChiPhiFunc(self, other):
        return ChiPhiFuncFourier(ChiPhiFunc.add_jit(self.content, other.content))

    # Addition of a constant with a ChiPhiFuncFourier.
    # Wrapper for numba method.
    def add_const(self, other):
        return ChiPhiFuncFourier(ChiPhiFunc.add_jit(self.content, np.complex128([[other]])))

    # Handles pointwise multiplication (* operator)
    # Convolve2d is already compiled. No need to jit.
    # -- Input: self and other
    # -- Output: a new ChiPhiFuncFourier
    def multiply(self, other, div=False):
        if div:
            raise NotImplementedError()
        return(ChiPhiFuncFourier(scipy.signal.convolve2d(self.content, other.content)))

    # Calculates an integer power of a ChiPhiFuncFourier
    # Convolve2d is already compiled. No need to jit.
    # Also here we assume all powers are to fairly low orders (2)
    # -- Input: self and power (int)
    # -- Output: self and other
    def pow(self, int_pow):
        new_content = self.content.copy()
        for n in range(int_pow-1):
            new_content = scipy.signal.convolve2d(new_content, new_content)
        return(ChiPhiFuncFourier(new_content))

    # Get a 2-argument lamnda function for plotting this term
    def get_lambda(self):
        len_chi = self.get_shape()[0]
        len_phi = self.get_shape()[1]

        if len_phi%2!=1:
            raise ValueError('coeffs_chi must have an odd number of components on phi axis')

        ind_phi = int((len_phi-1)/2)
        mode_phi = np.linspace(-ind_phi, ind_phi, len_phi)

        ind_chi = len_chi-1
        mode_chi = np.linspace(-ind_chi, ind_chi, len_chi).reshape(-1,1)
        # The outer dot product is summing along axis 0.
        # The inner @ (equivalent to dot product) is summing along axis 1.
        return(np.vectorize(lambda chi, phi : np.dot(self.content@(np.e**(1j*phi*mode_phi)), (np.e**(1j*(-chi)*mode_chi)))))


    # Utilities ---------------------------------------------------

    # Converting fourier coefficients into exponential coeffs used in
    # ChiPhiFunc's internal representation. Only used during super().__init__
    # Does not copy.
    def trig_to_exp(self):
        util_matrix_chi = fourier_to_exp_op(self.get_shape()[0])
        util_matrix_phi = fourier_to_exp_op(self.get_shape()[1])
        # Apply the conversion matrix on phi axis
        # (two T's because np.matmul can't choose axis)
#         self.content = (util_matrix_phi @ self.content.T).T
        self.content = self.content @ util_matrix_phi.T
        # Apply the conversion matrix on chi axis
        self.content = util_matrix_chi @ self.content

    # DEPRECIATED
    # Generate phi central difference operator diff_matrixT. content @ diff_matrixT.T = dchi(f).
    # (.T actually not needed since this the operator is diagonal)
    # -- Input: len_phi: length of phi series.
    # -- Output: 2d matrix.
    @njit
    def dphi_op(len_phi, invert = False):
        ind_phi = int((len_phi-1)/2)
        mode_phi = np.linspace(-ind_phi, ind_phi, len_phi)
        if invert:
            return(np.diag(-1j/mode_phi))
        return(np.diag(1j*mode_phi))

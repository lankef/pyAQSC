''' chipifunc.py '''
''' Spline dphi '''
class ChiPhiFunc:
    @partial(jit, static_argnums=(1, 2,))
    def dphi(self, order:int=1, mode=0):  # nfp-sensitive!!
        if order<0:
            return(ChiPhiFuncSpecial(-11))
        if mode==0:
            mode = diff_mode
        if mode==1:
            len_phi = self.content.shape[1]
            content_fft = jnp.fft.fft(self.content, axis=1)
            fftfreq_temp = jit_fftfreq_int(len_phi)*1j
            out_content_fft = content_fft*fftfreq_temp[None, :]**order
            out = jnp.fft.ifft(out_content_fft,axis=1)
        elif mode==2:
            out = self.content
            for i in range(order):
                out = (dphi_op_pseudospectral(self.content.shape[1]) @ out.T).T
        # Disabled because it's terrible.
        # elif mode==3:
        #     out = jnp.gradient(self.content, axis=1)/(jnp.pi*2/self.content.shape[1])
        # if mode[-6:]=='spline':
        #     if order>0:
        #         out = integrate_phi_spline(content, mode, periodic=False,
        #             diff=True, diff_order=order)
        #     if order<0:
        #         out = integrate_phi_spline(content, mode, periodic=False,
        #             diff=False, diff_order=-order)
        #     return(out)
        else:
            return(ChiPhiFuncSpecial(-11))
        return(ChiPhiFunc(out*self.nfp**order, self.nfp))
# Not supported by JAX
def integrate_phi_simpson(content, dx = 'default', periodic = False):
    '''
    Integrates a function on a grid using Simpson's method.
    Produces a content where values along axis 1 is the input content's
    integral.
    periodic is a special mode that assumes integrates a content over a period.
    It assumes that the grid function is periodic, and does not repeat the first
    grid's value.
    A usual content has the first cell's LEFT edge at 0
    and the last cell's RIGHT edge at 2pi.
    A specal dx is provided for a grid that has the first cell's LEFT edge at 0
    and the last cell's LEFT edge at 2pi.
    NOTE
    Cell values are ALWAYS taken at the left edge.
    nfp dependence is NOT HANDLED HERE.
    '''
    raise NotImplementedError('Simpson integrals not implemented in JAX')
    # len_chi = content.shape[0]
    # len_phi = content.shape[1]
    # if dx == 'default':
    #     dx = 2*np.pi/len_phi
    # if periodic:
    #     # The result of the integral is an 1d array of chi coeffs.
    #     # This integrates the full period, and needs to be wrapped.
    #     # the periodic=False option does not integrate the full period and
    #     # does not wrap.
    #     new_content = scipy.integrate.simpson(\
    #         wrap_grid_content_jit(content),\
    #         dx=dx,\
    #         axis=1\
    #         )
    #     return(np.array([new_content]).T)
    # else:
    #     # Integrate up to each element's grid
    #     integrate = lambda i_phi : scipy.integrate.simpson(content[:,:i_phi+1], dx=dx)
    #     out_list = Parallel(n_jobs=n_jobs, backend=backend, require=require)(
    #         delayed(integrate)(i_phi) for i_phi in range(len_phi)
    #     )
    #     return(np.array(out_list).T) # not nfp-dependent

# Not supported by JAX
def integrate_phi_spline(content, mode, dx = 'default', periodic=False,
    diff=False, diff_order=None):
    '''
    Implementation of spline-based integrate_phi using Parallel.
    nfp dependence is NOT HANDLED HERE.
    mode can be 'b_spline' or 'cubic_spline'
    '''
    raise NotImplementedError('Spline integral not implemented in JAX')
    # len_chi = content.shape[0]
    # len_phi = content.shape[1]
    # # if dx == 'default':
    # #     dx = 2*np.pi/len_phi
    # #     # purely real.
    #
    # content_looped = wrap_grid_content_jit(content)
    # # Separating real and imag components
    # content_re = np.real(content_looped)
    # content_im = np.imag(content_looped)
    # content_looped = np.concatenate((content_re, content_im), axis=0)
    # phis = np.linspace(0, np.pi*2, len_phi+1)
    #
    # def generate_and_integrate_spline(i_chi):
    #     if mode == 'b_spline':
    #         new_spline = scipy.interpolate.make_interp_spline(phis, content_looped[i_chi], bc_type = 'periodic')
    #         if diff:
    #             return(scipy.interpolate.splder(new_spline, n=diff_order))
    #         print('Waring! B-spline antiderivative is known to produce a small constant offset to the result.')
    #         return(scipy.interpolate.splantider(new_spline))
    #     elif mode == 'cubic_spline':
    #         new_spline = scipy.interpolate.CubicSpline(phis, content_looped[i_chi], bc_type = 'periodic')
    #         if diff:
    #             return(new_spline.derivative(diff_order))
    #         return(new_spline.antiderivative())
    #     else:
    #         raise AttributeError('Spline mode \''+str(mode)+'\' is not recognized.')
    # A list of integrated splines
    # integrate_spline_list = Parallel(n_jobs=n_jobs, backend=backend, require=require)(
    #     delayed(generate_and_integrate_spline)(i_chi) for i_chi in range(len_chi*2)
    # )
    #
    # if periodic:
    #     # The result of the integral is an 1d array of chi coeffs.
    #     # This integrates the full period, and needs to be wrapped.
    #     # the periodic=False option does not integrate the full period and
    #     # does not wrap.
    #     evaluate_spline_2pi = lambda spline: spline(2*np.pi)
    #     out_list = Parallel(n_jobs=n_jobs, backend=backend, require=require)(
    #         delayed(evaluate_spline_2pi)(spline) for spline in integrate_spline_list
    #     )
    #     out_list = np.array(out_list)
    #     out_list = out_list[:len_chi]+1j*out_list[len_chi:]
    #     return(out_list[:, None])
    # else:
    #     evaluate_spline = lambda spline, phis : spline(phis)
    #     out_list = Parallel(n_jobs=n_jobs, backend=backend, require=require)(
    #         delayed(evaluate_spline)(spline, phis[:-1]) for spline in integrate_spline_list
    #     )
    #     out_list = np.array(out_list)
    #     out_list = out_list[:len_chi]+1j*out_list[len_chi:]
    #     return(out_list) # not nfp-dependent

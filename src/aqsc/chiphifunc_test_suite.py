import numpy as np
import scipy.signal
from matplotlib import pyplot as plt

# for importing parsed codes
from .chiphifunc import *
from .chiphiepsfunc import *
from .equilibrium import *
from .math_utilities import is_seq,py_sum,is_integer,diff
from .config import *

# Detect whether pyQSC is enabled.
if use_pyQSC:
    from qsc import Qsc

def rand_splines(len_chi, amp_range = (0.5,2), n_points=5):
    amplitude = np.random.random()*(amp_range[1]-amp_range[0])+amp_range[0]
    # Random anchor points
    y = np.random.random((len_chi, n_points))*amplitude*2-amplitude
    y = np.concatenate([y, (y[:, 0])[:, None]], axis=1)
    x = np.linspace(0,2*np.pi, n_points+1)

    # Shift the anchor points onto a non-uniform grid
    random_shift = (np.random.random(n_points+1)-0.5)*0.8*np.pi*2/n_points
    random_shift[0] = 0
    random_shift[-1] = 0
    x = x+random_shift

    # Fit periodic cubic spline
    splines = scipy.interpolate.CubicSpline(x, y, axis=1, bc_type='periodic')
    return(splines)

def ChiPhiFunc_from_splines(splines, nfp=1, dphi_order=0, len_phi=1000):
    '''
    Constructs a real ChiPhiFunc from a set of known cubic splines.

    Inputs: -----
    spline: set of cubic splines.
    dphi_order: take derivatives before constructing the ChiPhiFunc.
    nfp: field period number,
    len_phi: number of grid points in phi.
    Output: -----
    A ChiPhiFunc.
    '''
    points = np.linspace(0, 2*np.pi*(1-1/len_phi), len_phi)
    content = splines.derivative(dphi_order)(points)*nfp**dphi_order

    return(ChiPhiFunc(content, nfp, trig_mode=True))

def rand_ChiPhiFunc(len_chi, nfp=1, len_phi=1000, amp_range = (0.5,2), n_points=5):
    return(
        ChiPhiFunc_from_splines(
            rand_splines(len_chi, amp_range, n_points),
            dphi_order=0,
            nfp=nfp,
            len_phi=len_phi
        )
    )

# nfp-dependent!!
def rand_ChiPhiEpsFunc(order, nfp=1, zero_outer=False):
    list_out = [0]
    for i in range(order):
        new_content = rand_ChiPhiFunc(i+1, nfp).content
        if zero_outer:
            new_content[0,:]=0
            new_content[-1,:]=0
        list_out.append(ChiPhiFunc(new_content, nfp))
    return(ChiPhiEpsFunc(list_out, nfp, True))

# Evaluate a callable on 'points' (defined above)
# not nfp-dependent
def evaluate(func):
    phi = np.linspace(0, 2*np.pi*(1-1/n_grid_phi), n_grid_phi) 
    chi = np.linspace(0, 2*np.pi*(1-1/n_grid_chi), n_grid_chi)
    return(func(chi[:, None], phi[None, :]))

# Evaluate a ChiPhiFunc on 'points' (defined above)
# not nfp-dependent
def evaluate_ChiPhiFunc(chiphifunc_in):
    return(evaluate(chiphifunc_in.eval))

# # Evaluate every elements of a ChiPhiEpsFunc on 'points', and returns
# # a ChiPhiEpsFunc where all elements are np arrays storing evaluation results
# # on 'points'.
# # not nfp-dependent
# def evaluate_ChiPhiEpsFunc(chiphepsfunc_in):
#     if not isinstance(chiphepsfunc_in, ChiPhiEpsFunc):
#         raise TypeError('Input must be a ChiPhiEpsFunc')
#     new_list = []
#     for item in chiphepsfunc_in.chiphifunc_list:
#         if isinstance(item, ChiPhiFunc):
#             new_list.append(evaluate_ChiPhiFunc(item))
#         else:
#             new_list.append(item)
#     return(ChiPhiEpsFunc(new_list))

# Display an array result from evaluate() or evaluate_ChiPhiFunc()
# not nfp-dependent
def display(array, complex=True):
    plt.pcolormesh(chi, phi, np.real(array).T)
    plt.colorbar()
    plt.show()
    if complex:
        plt.pcolormesh(chi, phi, np.imag(array).T)
        plt.colorbar()
        plt.show()

# Plots the content of two ChiPhiFunc's and compare.
# nfp-dependent!!
def compare_chiphifunc(
        A:ChiPhiFunc, B:ChiPhiFunc, 
        trig_mode:bool=False, 
        simple_mode:bool=True, 
        colormap_mode:bool=False):
    if not simple_mode:
        print('A')
        A.display_content(trig_mode=trig_mode, colormap_mode=colormap_mode)
        print('B')
        B.display_content(trig_mode=trig_mode, colormap_mode=colormap_mode)
    if A.content.shape[1]!=B.content.shape[1]:
        print('Phi grid number not matched.')
        return()

    if A.nfp != B.nfp:
        if A.nfp == 1 or B.nfp == 1:
            print('One of A and B is converted to have nfp=1.')
            print('A.nfp =', A.nfp)
            print('B.nfp =', B.nfp)
            compare_chiphifunc(A.export_single_nfp(), B.export_single_nfp(),
            trig_mode, simple_mode, colormap_mode)
            return()
        elif A.nfp != 0 and B.nfp != 0:
            print('A, B has different non-zero, non-one nfp!')
            print('A.nfp =', A.nfp)
            print('B.nfp =', B.nfp)
            return()
    else:
        nfp = max(A.nfp, B.nfp)



    diff_AB = A-B
    # A or B has extra components, plot those components separately
    if A.content.shape[0]!=B.content.shape[0]:
        amount_to_trim = abs(A.content.shape[0]-B.content.shape[0])//2
        center_content = diff_AB.content[amount_to_trim: -amount_to_trim].copy()
        trimmed_content = diff_AB.content.copy()
        trimmed_content[amount_to_trim: -amount_to_trim] = np.zeros_like(center_content)
        diff_AB_center = ChiPhiFunc(center_content, nfp)
        diff_AB_trimmed = ChiPhiFunc(trimmed_content, nfp)
        print('A and B has different number of components.')
        print('Difference')
        diff_AB_center.display_content(trig_mode=trig_mode)
        print('Extra components')
        diff_AB_trimmed.display_content(trig_mode=trig_mode)
    else:
        print('Difference')
        diff_AB.display_content(trig_mode=trig_mode)

    print('fractional errors b/w data and general formula')

    # Sometimes 2 ChiPhiFuncs being compared will have different row/col numbers.
    if A.content.shape[0]%2!=B.content.shape[0]%2:
        raise AttributeError('2 ChiPhiFunc\'s being compared have different'\
        'even/oddness.')

    shape = (max(A.content.shape[0], B.content.shape[0]),max(A.content.shape[1],B.content.shape[1]))
    A_content_padded = np.zeros(shape, np.complex128)
    B_content_padded = np.zeros(shape, np.complex128)

    a_pad_row = (shape[0] - A.content.shape[0])//2
    a_pad_col = (shape[1] - A.content.shape[1])//2
    b_pad_row = (shape[0] - B.content.shape[0])//2
    b_pad_col = (shape[1] - B.content.shape[1])//2
    A_content_padded[a_pad_row:shape[0]-a_pad_row,a_pad_col:shape[1]-a_pad_col] = A.content
    B_content_padded[b_pad_row:shape[0]-b_pad_row,b_pad_col:shape[1]-b_pad_col] = B.content
    print_fractional_error(A_content_padded, B_content_padded)

# Compare 2 arrays and print out absolute and fractional error.
# Used for comparing evaluation results or contents
# not nfp-dependent
def print_fractional_error(guess, ans):
    if np.any(ans):
        frac = np.abs((guess-ans)/ans)
    else:
        frac = np.nan
    actual = np.abs((guess-ans))
    print('Max amplitude of arg a:',np.max(np.abs(guess)))
    print('Max amplitude of arg b:',np.max(np.abs(ans)))
    print('{:<15} {:<15} {:<15}'.format(
        'Error type:',
        'Fractional',
        'Total'
    ))
    print('{:<15} {:<15} {:<15}'.format(
        'Avg:',
        np.format_float_scientific(np.average(frac),3),
        np.format_float_scientific(np.average(actual),3)
    ))
    print('{:<15} {:<15} {:<15}'.format(
        'Worst:',
        np.format_float_scientific(np.nanmax(frac),3),
        np.format_float_scientific(np.nanmax(actual),3)
    ))
    print('{:<15} {:<15} {:<15}'.format(
        'Std',
        np.format_float_scientific(np.std(frac),3),
        np.format_float_scientific(np.std(actual),3)
    ))
    print('Total imaginary component')
    print(np.sum(np.imag(frac)))
    print('')

# Compare the cumulative error from repeated calls of a single-argument callable
# on a ChiPhiFunc to a single-argument callable on an EVALUATED array.
# not nfp-dependent
def cumulative_error(chiphifunc_in, callable_chiphifunc, callable_array, num_steps):
    guess = chiphifunc_in
    ans = evaluate_ChiPhiFunc(chiphifunc_in)
    errors = []
    for i in range(num_steps):
        errors.append(np.average(evaluate_ChiPhiFunc(guess)-ans))
        guess = callable_chiphifunc(guess)
        ans = callable_array(ans)

    plt.plot(errors)
    plt.ylabel('Error')
    plt.xlabel('# execution')
    plt.show()

# Generates R or Z from Eduardo's axis shape array for pyqsc
# The notation represents [number of harmonics, 0 harm, value,
# nth harmonics, value_cos,value_sin,...].
# Outputs Matt Landreman's format:
# rc = [const, <coeff of mode=i*nfp>]
# rs = [0,     <coeff of mode=i*nfp>]
# not nfp-dependent
def rodriguez_to_landreman(in_array, nfp):
    num_modes = in_array[0]
    cos_modes = {}
    sin_modes = {}
    offset = 1
    const = 0
    if in_array[1]==0:
        const = in_array[2]
        offset = 3
    for i in range((len(in_array)-offset)//3):
        cos_modes[in_array[i*3+offset]] = in_array[i*3+offset+1]
        sin_modes[in_array[i*3+offset]] = in_array[i*3+offset+2]

    c_out = [const]
    s_out = [0]

    # Matt Landreman's rc, rs, zc, zs are of format:
    #
    # rc = [const, <coeff of mode=i*nfp>]
    # rs = [0,     <coeff of mode=i*nfp>]
    if np.any(np.array(list(sin_modes.keys()))%nfp!=0) or np.any(np.array(list(cos_modes.keys()))%nfp!=0):
        print('sin_modes.keys()')
        print(sin_modes.keys())
        print('cos_modes.keys()')
        print(cos_modes.keys())
        raise ValueError('Mode incompatible with nfp detected.')
    max_i = int(max(sin_modes.keys())//nfp)
    for i in range(max_i):
        mode_num = (i+1)*nfp
        if mode_num in cos_modes.keys():
            c_out.append(cos_modes[mode_num])
        else:
            c_out.append(0)
        if mode_num in sin_modes.keys():
            s_out.append(sin_modes[mode_num])
        else:
            s_out.append(0)

    return(c_out, s_out)

# Eduardo provided hand-calculated values to the 3rd order.
# each folder always contain:
# Bp0.dat		X31c.dat	Y22s.dat	Z20.dat		inputs.dat
# Bpc11.dat		X31s.dat	Y31c.dat	Z22c.dat	kappa.dat
# Bps11.dat		X33c.dat	Y31s.dat	Z22s.dat	tau.dat
# Btc20.dat		X33s.dat	Y33c.dat	Z31c.dat	outputs.dat
# X20c.dat		Xc1.dat		Y33s.dat	Z31s.dat
# X22c.dat		Y20.dat		Yc1.dat		Z33c.dat
# X22s.dat		Y22c.dat	Ys1.dat		Z33s.dat
# nfp-dependent!!
def read_first_three_orders(path, R_array, Z_array, numerical_mode = False, nfp_enabled=False, plot_axis=False):
    if not use_pyQSC:
        raise AttributeError(
            'use_pyQSC must be enabled to use test datasets from Eduardo Rodriguez.'
        )


    nfp_read, Xi_0, eta, B20, B22c, B22s, B31c, B31s, B33c, B33s, Ba0, Ba1 = np.loadtxt(path+'inputs.dat')
    print('Configuration has',nfp_read,'field periods.')
    if nfp_enabled:
        nfp=int(nfp_read)
    else:
        nfp=1

    # The last element should already been removed.
    def divide_by_nfp(in_array, nfp):
        return(in_array[:len(in_array)//nfp])

    # Delta --------------------------------------
    # The last elements in all data files repeat the first elements. the
    # [:-1] removes it.
    d0_raw = np.loadtxt(path+'d0.dat')[:-1]
    if len(d0_raw)%nfp!=0:
        print('Grid number:', len(d0), ' isn\'t exact multiple of nfp:', nfp)

    d0 = divide_by_nfp(d0_raw, nfp)

    Delta_0 = ChiPhiFunc(np.array([d0]), nfp)


    d11c = divide_by_nfp(np.loadtxt(path+'d11c.dat')[:-1], nfp)
    d11s = divide_by_nfp(np.loadtxt(path+'d11s.dat')[:-1], nfp)
    Delta_1 = ChiPhiFunc(np.array([
        d11s,
        d11c
    ]), nfp, trig_mode = True)

    d20c = divide_by_nfp(np.loadtxt(path+'d20c.dat')[:-1], nfp)
    d22c = divide_by_nfp(np.loadtxt(path+'d22c.dat')[:-1], nfp)
    d22s = divide_by_nfp(np.loadtxt(path+'d22s.dat')[:-1], nfp)
    Delta_2 = ChiPhiFunc(np.array([d22s, d20c, d22c]), nfp, trig_mode = True)
    Delta_coef_cp = ChiPhiEpsFunc_remove_zero([Delta_0, Delta_1, Delta_2], nfp, True)

    # P_perp --------------------------------------
    p0 = divide_by_nfp(np.loadtxt(path+'p0.dat')[:-1], nfp)
    p_perp_0 = ChiPhiFunc(np.array([p0]), nfp)

    pc1 = divide_by_nfp(np.loadtxt(path+'pc1.dat')[:-1], nfp)
    p_perp_1 = ChiPhiFunc(np.array([
        np.zeros_like(pc1),
        pc1
    ]), nfp, trig_mode = True)

    p20c = divide_by_nfp(np.loadtxt(path+'p20c.dat')[:-1], nfp)
    p22s = divide_by_nfp(np.loadtxt(path+'p22s.dat')[:-1], nfp)
    p22c = divide_by_nfp(np.loadtxt(path+'p22c.dat')[:-1], nfp)
    p_perp_2 = ChiPhiFunc(np.array([p22s,p20c,p22c]), nfp, trig_mode = True)
    p_perp_coef_cp = ChiPhiEpsFunc_remove_zero([p_perp_0, p_perp_1, p_perp_2], nfp, True)

    # B psi ---------------------------------------
    Bp0 = divide_by_nfp(np.loadtxt(path+'Bp0.dat')[:-1], nfp)
    B_psi_0 = ChiPhiFunc(np.array([Bp0]), nfp)

    Bpc11 = divide_by_nfp(np.loadtxt(path+'Bpc11.dat')[:-1], nfp)
    Bps11 = divide_by_nfp(np.loadtxt(path+'Bps11.dat')[:-1], nfp)
    B_psi_1 = ChiPhiFunc(np.array([
        Bps11,
        Bpc11
    ]), nfp, trig_mode = True)

    B_psi_coef_cp = ChiPhiEpsFunc_remove_zero([B_psi_0, B_psi_1], nfp, True)

    # B theta ---------------------------------------
    Btc20 = divide_by_nfp(np.loadtxt(path+'Btc20.dat')[:-1], nfp)
    B_theta_2 = ChiPhiFunc(np.array([
        Btc20
    ]), nfp, trig_mode = True)
    B_theta_coef_cp = ChiPhiEpsFunc_remove_zero([0, 0, B_theta_2], nfp, True)

    # X ---------------------------------------
    Xc1 = divide_by_nfp(np.loadtxt(path+'Xc1.dat')[:-1], nfp)
    X1 = ChiPhiFunc(np.array([
        np.zeros_like(Xc1), # sin coeff is zero
        Xc1,
    ]), nfp, trig_mode = True)

    X20 = divide_by_nfp(np.loadtxt(path+'X20c.dat')[:-1], nfp)
    X22c = divide_by_nfp(np.loadtxt(path+'X22c.dat')[:-1], nfp)
    X22s = divide_by_nfp(np.loadtxt(path+'X22s.dat')[:-1], nfp)
    X2 = ChiPhiFunc(np.array([
        X22s,
        X20,
        X22c
    ]), nfp, trig_mode = True)

    X31c = divide_by_nfp(np.loadtxt(path+'X31c.dat')[:-1], nfp)
    X31s = divide_by_nfp(np.loadtxt(path+'X31s.dat')[:-1], nfp)
    X33c = divide_by_nfp(np.loadtxt(path+'X33c.dat')[:-1], nfp)
    X33s = divide_by_nfp(np.loadtxt(path+'X33s.dat')[:-1], nfp)
    X3 = ChiPhiFunc(np.array([
        X33s,
        X31s,
        X31c,
        X33c
    ]), nfp, trig_mode = True)

    X_coef_cp = ChiPhiEpsFunc_remove_zero([0, X1, X2, X3], nfp, True)

    # Y ---------------------------------------
    Ys1 = divide_by_nfp(np.loadtxt(path+'Ys1.dat')[:-1], nfp)
    Yc1 = divide_by_nfp(np.loadtxt(path+'Yc1.dat')[:-1], nfp)
    Y1 = ChiPhiFunc(np.array([
        Ys1, # sin coeff is zero
        Yc1,
    ]), nfp, trig_mode = True)

    Y20 = divide_by_nfp(np.loadtxt(path+'Y20.dat')[:-1], nfp)
    Y22c = divide_by_nfp(np.loadtxt(path+'Y22c.dat')[:-1], nfp)
    Y22s = divide_by_nfp(np.loadtxt(path+'Y22s.dat')[:-1], nfp)
    Y2 = ChiPhiFunc(np.array([
        Y22s,
        Y20,
        Y22c
    ]), nfp, trig_mode = True)

    Y31c = divide_by_nfp(np.loadtxt(path+'Y31c.dat')[:-1], nfp)
    Y31s = divide_by_nfp(np.loadtxt(path+'Y31s.dat')[:-1], nfp)
    Y33c = divide_by_nfp(np.loadtxt(path+'Y33c.dat')[:-1], nfp)
    Y33s = divide_by_nfp(np.loadtxt(path+'Y33s.dat')[:-1], nfp)
    Y3 = ChiPhiFunc(np.array([
        Y33s,
        Y31s,
        Y31c,
        Y33c
    ]), nfp, trig_mode = True)

    Y_coef_cp = ChiPhiEpsFunc_remove_zero([0, Y1, Y2, Y3], nfp, True)

    # Z ---------------------------------------
    Z20 = divide_by_nfp(np.loadtxt(path+'Z20.dat')[:-1], nfp)
    Z22c = divide_by_nfp(np.loadtxt(path+'Z22c.dat')[:-1], nfp)
    Z22s = divide_by_nfp(np.loadtxt(path+'Z22s.dat')[:-1], nfp)
    Z2 = ChiPhiFunc(np.array([
        Z22s,
        Z20,
        Z22c
    ]), nfp, trig_mode = True)

    Z31c = divide_by_nfp(np.loadtxt(path+'Z31c.dat')[:-1], nfp)
    Z31s = divide_by_nfp(np.loadtxt(path+'Z31s.dat')[:-1], nfp)
    Z33c = divide_by_nfp(np.loadtxt(path+'Z33c.dat')[:-1], nfp)
    Z33s = divide_by_nfp(np.loadtxt(path+'Z33s.dat')[:-1], nfp)
    Z3 = ChiPhiFunc(np.array([
        Z33s,
        Z31s,
        Z31c,
        Z33c
    ]), nfp, trig_mode = True)

    Z_coef_cp = ChiPhiEpsFunc_remove_zero([0, 0, Z2, Z3], nfp, True)

    # Constants
    B_alpha_e = ChiPhiEpsFunc_remove_zero([Ba0, Ba1], nfp, True)
    kap_p = ChiPhiFunc(np.array([divide_by_nfp(np.loadtxt(path+'kappa.dat')[:-1], nfp)]), nfp)
    tau_p = ChiPhiFunc(np.array([divide_by_nfp(np.loadtxt(path+'tau.dat')[:-1], nfp)]), nfp)

    B2 = ChiPhiFunc(np.array([
        [np.average(B22s)],
        [np.average(B20)],
        [np.average(B22c)]
    ]), nfp, trig_mode = True)
    B3 = ChiPhiFunc(np.array([
        [np.average(B33s)],
        [np.average(B31s)],
        [np.average(B31c)],
        [np.average(B33c)]
    ]), nfp, trig_mode = True)
    # B1 is given by first order jacobian equation, above 14 in the first
    # half of the 2-part paper
    B_denom_coef_c = ChiPhiEpsFunc_remove_zero([1, phi_avg(-X1*2*kap_p), B2, B3], nfp, True)

    iota_e = ChiPhiEpsFunc_remove_zero(list(np.loadtxt(path+'outputs.dat')), nfp, True)

    # Not an actual representation in pyQSC.
    # only for calculating axis length.
    rc, rs = rodriguez_to_landreman(R_array, 1)# should be 1?
    zc, zs = rodriguez_to_landreman(Z_array, 1)
    stel = Qsc(rc=rc, rs=rs, zc=zc, zs=zs, nfp=1)
    dl_p = stel.axis_length/(2*np.pi)
    if plot_axis:
        print('Axis shape:')
        stel.plot_axis(frenet=False)

    if numerical_mode:
        return(
        evaluate_ChiPhiEpsFunc(B_psi_coef_cp),
        evaluate_ChiPhiEpsFunc(B_theta_coef_cp),
        evaluate_ChiPhiEpsFunc(X_coef_cp),
        evaluate_ChiPhiEpsFunc(Y_coef_cp),
        evaluate_ChiPhiEpsFunc(Z_coef_cp),
        evaluate_ChiPhiEpsFunc(iota_e),
        dl_p,
        int(nfp_read),
        Xi_0,
        eta,
        evaluate_ChiPhiEpsFunc(B_denom_coef_c),
        evaluate_ChiPhiEpsFunc(B_alpha_e),
        kap_p.content,
        tau_p.content
    )

    return(
        B_psi_coef_cp, 
        B_theta_coef_cp,
        Delta_coef_cp, 
        p_perp_coef_cp,
        X_coef_cp, 
        Y_coef_cp, 
        Z_coef_cp,
        iota_e, 
        dl_p,
        int(nfp_read), 
        Xi_0, 
        eta,
        B_denom_coef_c,
        B_alpha_e,
        kap_p, 
        tau_p
    )

# not nfp-dependent
def chiphifunc_debug_plot():
    plt.pcolormesh(np.real(test_B_theta.content))
    plt.colorbar()
    plt.show()

    plt.plot(chiphifunc.debug_max_value, label = 'max')
    plt.plot(chiphifunc.debug_avg_value, label = 'avg')
    plt.show()
    plt.pcolormesh(chiphifunc.debug_pow_diff_add)
    plt.colorbar()
    plt.show()

    chiphifunc.debug_pow_diff_add = []
    chiphifunc.debug_max_value = []
    chiphifunc.debug_avg_value = []

# nfp-dependent!!
def import_from_stel(stel, len_phi=1000, nfp_enabled=False):
    if not use_pyQSC:
        raise AttributeError(
            'use_pyQSC must be enabled to use test datasets from qyPSC.'
        )
    # Spline fit and interpolate to certain elements
    # Or if the item is scalar, creates a filled 1d array of
    # len_phi
    def to_phi(varphi, x, len_phi=len_phi):
        if np.isscalar(x):
            return(np.full((len_phi), x))
        else:
            interp_array = x
            phis = varphi
            if not nfp_enabled:
                interp_array = np.tile(interp_array,stel.nfp+1)
                for i in range(1,stel.nfp+1):
                    phis = np.append(phis,varphi+np.pi*2/stel.nfp*i)
                f = interp1d(phis, interp_array, kind='cubic')
                return(f(np.linspace(0,np.pi*2*(len_phi-1)/len_phi, len_phi)))
            else:
                f = interp1d(phis, interp_array, kind='cubic')
                return(f(np.linspace(0,np.pi*2*(len_phi-1)/len_phi/stel.nfp, len_phi)))
    if not nfp_enabled:
        nfp = 1
    else:
        nfp = stel.nfp

    dl_p = stel.abs_G0_over_B0
    iota = stel.iotaN
    iota_coef = ChiPhiEpsFunc_remove_zero([iota], nfp, True)
    tau_p = ChiPhiFunc(np.array([to_phi(stel.varphi, -stel.torsion)]), nfp)
    kap_p = ChiPhiFunc(np.array([to_phi(stel.varphi, stel.curvature)]), nfp)
    B0 = 1/stel.B0**2
    eta = stel.etabar*np.sqrt(2)*B0**0.25
    #
    r_factor = np.sqrt(2/stel.Bbar)
    # X, Y, Z -----------------------------------
    X1 = ChiPhiFunc(np.array([
        to_phi(stel.varphi, stel.X1s), # sin coeff is zero
        to_phi(stel.varphi, stel.X1c),
    ]), nfp, trig_mode = True)*r_factor
    X2 = ChiPhiFunc(np.array([
        to_phi(stel.varphi, stel.X2s),
        to_phi(stel.varphi, stel.X20),
        to_phi(stel.varphi, stel.X2c)
    ]), nfp, trig_mode = True)*r_factor**2
    X3 = ChiPhiFunc(np.array([
        to_phi(stel.varphi, stel.X3s3),
        to_phi(stel.varphi, stel.X3s1),
        to_phi(stel.varphi, stel.X3c1),
        to_phi(stel.varphi, stel.X3c3)*r_factor**3
    ]), nfp, trig_mode = True)
    X_coef_cp = ChiPhiEpsFunc_remove_zero([0, X1, X2], nfp, True)
    Y1 = ChiPhiFunc(np.array([
        to_phi(stel.varphi, stel.Y1s), # sin coeff is zero
        to_phi(stel.varphi, stel.Y1c),
    ]), nfp, trig_mode = True)*r_factor
    Y2 = ChiPhiFunc(np.array([
        to_phi(stel.varphi, stel.Y2s),
        to_phi(stel.varphi, stel.Y20),
        to_phi(stel.varphi, stel.Y2c)
    ]), nfp, trig_mode = True)*r_factor**2
    Y3 = ChiPhiFunc(np.array([
        to_phi(stel.varphi, stel.Y3s3),
        to_phi(stel.varphi, stel.Y3s1),
        to_phi(stel.varphi, stel.Y3c1),
        to_phi(stel.varphi, stel.Y3c3)
    ]), nfp, trig_mode = True)*r_factor**3
    Y_coef_cp = ChiPhiEpsFunc_remove_zero([0, Y1, Y2], nfp, True)
    Z2 = ChiPhiFunc(np.array([
        to_phi(stel.varphi, stel.Z2s),
        to_phi(stel.varphi, stel.Z20),
        to_phi(stel.varphi, stel.Z2c)
    ]), nfp, trig_mode = True)*r_factor**2
    Z3 = ChiPhiFunc(np.array([
        to_phi(stel.varphi, stel.Z3s3),
        to_phi(stel.varphi, stel.Z3s1),
        to_phi(stel.varphi, stel.Z3c1),
        to_phi(stel.varphi, stel.Z3c3)
    ]), nfp, trig_mode = True)*r_factor**3
    Z_coef_cp = ChiPhiEpsFunc_remove_zero([0, 0, Z2], nfp, trig_mode = True)
    # B components

    Btc20 = 2*stel.I2/stel.B0
    B_theta_coef_cp = ChiPhiEpsFunc_remove_zero([0, 0, ChiPhiFunc(np.array([
        to_phi(stel.varphi, Btc20)
    ]), nfp)], nfp, True)

    B_psi_coef_cp = ChiPhiEpsFunc_remove_zero([0], nfp, trig_mode = True)

    B1c = -2*B0*eta
    B1 = ChiPhiFunc(np.array([
        0,
        np.average(B1c)
    ]), nfp, trig_mode = True)
    B20 = (0.75*stel.etabar**2/np.sqrt(B0) - stel.B20)*4*B0**2
    B2c = (0.75*stel.etabar**2/np.sqrt(B0) - stel.B2c)*4*B0**2
    B2s = -4*stel.B2s*B0**2
    B2 = ChiPhiFunc(np.array([
        np.average(B2s),
        np.average(B20),
        np.average(B2c)
    ]), nfp, trig_mode = True)
    B_denom_coef_c = ChiPhiEpsFunc_remove_zero([
        B0,
        B1,
        ChiPhiFunc(
            np.array([np.average(B20)]),nfp
        )
    ],nfp, True)

    Ba0 = np.average(stel.G0)
    Ba1 = np.average(2/stel.B0*(stel.G2 + stel.iotaN*stel.I2))
    B_alpha_coef = ChiPhiEpsFunc_remove_zero([Ba0, Ba1], nfp, True)

    # X_coef_cp.mask(2), Done
    # Y_coef_cp.mask(2), Done
    # Z_coef_cp.mask(2), Done
    # B_psi_coef_cp.mask(0), Done
    # B_theta_coef_cp.mask(2), Done
    # B_denom_coef_c.mask(2), Done
    # B_alpha_coef.mask(1), Done
    # kap_p, dl_p, tau_p, Done
    # iota_coef.mask(0), eta, Done
    equilibrium_out = Equilibrium.from_known(X_coef_cp,
        Y_coef_cp,
        Z_coef_cp,
        B_psi_coef_cp,
        B_theta_coef_cp,
        B_denom_coef_c,
        B_alpha_coef,
        kap_p, dl_p, tau_p,
        iota_coef, eta,
        ChiPhiEpsFunc_remove_zero([0,0,0], nfp, True), # no pressure or delta
        ChiPhiEpsFunc_remove_zero([0,0,0], nfp, True))

    return(equilibrium_out, Y3)

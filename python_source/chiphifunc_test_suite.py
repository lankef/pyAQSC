# This file implements and tests recursion relations
from qsc import Qsc
import numpy as np
import timeit
import scipy.signal
from matplotlib import pyplot as plt

# for importing parsed codes
from chiphifunc import *
from chiphiepsfunc import *
from math_utilities import is_seq,py_sum,is_integer,diff

# Size of the chi and phi grid used for evaluation
n_grid_phi = 1000
n_grid_chi = 500
points = np.linspace(0, 2*np.pi*(1-1/n_grid_phi), n_grid_phi)
chi = np.linspace(0, 2*np.pi*(1-1/n_grid_chi), n_grid_chi)
phi = points

# Evaluate a callable on 'points' (defined above)
def evaluate(func):
    return(func(chi.reshape(-1,1), phi))

# Evaluate a ChiPhiFunc on 'points' (defined above)
def evaluate_ChiPhiFunc(chiphifunc_in):
    return(evaluate(chiphifunc_in.get_lambda()))

# Evaluate every elements of a ChiPhiEpsFunc on 'points', and returns
# a ChiPhiEpsFunc where all elements are np arrays storing evaluation results
# on 'points'.
def evaluate_ChiPhiEpsFunc(chiphepsfunc_in):
    if not isinstance(chiphepsfunc_in, ChiPhiEpsFunc):
        raise TypeError('Input must be a ChiPhiEpsFunc')
    new_list = []
    for item in chiphepsfunc_in.chiphifunc_list:
        if isinstance(item, ChiPhiFunc):
            new_list.append(evaluate_ChiPhiFunc(item))
        else:
            new_list.append(item)
    return(ChiPhiEpsFunc(new_list))

# Display an array result from evaluate() or evaluate_ChiPhiFunc()
def display(array, complex=True):
    plt.pcolormesh(chi, phi, np.real(array).T)
    plt.colorbar()
    plt.show()
    if complex:
        plt.pcolormesh(chi, phi, np.imag(array).T)
        plt.colorbar()
        plt.show()

# Plots the content of two ChiPhiFunc's and compare.
def compare_chiphifunc(A, B, fourier_mode=True, simple_mode = True, colormap_mode=False):
    if not simple_mode:
        print('A')
        A.display_content(fourier_mode=fourier_mode, colormap_mode=colormap_mode)
        print('B')
        B.display_content(fourier_mode=fourier_mode, colormap_mode=colormap_mode)


    diff_AB = A-B
    # A or B has extra components, plot those components separately
    if A.get_shape()[0]!=B.get_shape()[0]:
        amount_to_trim = abs(A.get_shape()[0]-B.get_shape()[0])//2
        center_content = diff_AB.content[amount_to_trim: -amount_to_trim].copy()
        trimmed_content = diff_AB.content.copy()
        trimmed_content[amount_to_trim: -amount_to_trim] = np.zeros_like(center_content)
        diff_AB_center = ChiPhiFunc(center_content)
        diff_AB_trimmed = ChiPhiFunc(trimmed_content)
        print('A and B has different number of components.')
        print('Difference')
        diff_AB_center.display_content(fourier_mode=fourier_mode)
        print('Extra components')
        diff_AB_trimmed.display_content(fourier_mode=fourier_mode)
    else:
        print('Difference')
        diff_AB.display_content(fourier_mode=fourier_mode)

    print('fractional errors b/w data and general formula')

    # Sometimes 2 ChiPhiFuncs being compared will have different row/col numbers.
    if A.get_shape()[0]%2!=B.get_shape()[0]%2:
        raise AttributeError('2 ChiPhiFunc\'s being compared have different'\
        'even/oddness.')
    A_content, B_content = A.stretch_phi_to_match(B)
    A = ChiPhiFunc(A_content)
    B = ChiPhiFunc(B_content)
    shape = (max(A.get_shape()[0], B.get_shape()[0]),max(A.get_shape()[1],B.get_shape()[1]))
    A_content_padded = np.zeros(shape)
    B_content_padded = np.zeros(shape)

    a_pad_row = (shape[0] - A.get_shape()[0])//2
    a_pad_col = (shape[1] - A.get_shape()[1])//2
    b_pad_row = (shape[0] - B.get_shape()[0])//2
    b_pad_col = (shape[1] - B.get_shape()[1])//2
    A_content_padded[a_pad_row:shape[0]-a_pad_row,a_pad_col:shape[1]-a_pad_col] = A.content
    B_content_padded[b_pad_row:shape[0]-b_pad_row,b_pad_col:shape[1]-b_pad_col] = B.content
    print_fractional_error(A_content_padded, B_content_padded)


# Compare 2 arrays and print out absolute and fractional error.
# Used for comparing evaluation results or contents
def print_fractional_error(guess, ans):
    if np.any(ans):
        frac = np.abs((guess-ans)/ans)
    else:
        frac = np.nan
    actual = np.abs((guess-ans))
    print('{:<15} {:<15} {:<15}'.format('Error type:','Fractional', 'Total'))
    print('{:<15} {:<15} {:<15}'.format('Avg:',np.format_float_scientific(np.average(frac),3), np.format_float_scientific(np.average(actual),3)))
    print('{:<15} {:<15} {:<15}'.format('Worst:',np.format_float_scientific(np.nanmax(frac),3), np.format_float_scientific(np.nanmax(actual),3)))
    print('{:<15} {:<15} {:<15}'.format('Std',np.format_float_scientific(np.std(frac),3), np.format_float_scientific(np.std(actual),3)))
    print('Total imaginary component')
    print(np.sum(np.imag(frac)))
    print('')

# Compare the cumulative error from repeated calls of a single-argument callable
# on a ChiPhiFunc to a single-argument callable on an EVALUATED array.
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
def eduardo_to_matt(in_array, nfp):
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
    max_i = max(sin_modes.keys())//nfp
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
def read_first_three_orders(path, R_array, Z_array, numerical_mode = False):

    # The last elements in all data files repeat the first elements. the
    # [:-1] removes it.

    # Delta ---------------------------------------
    d0 = np.loadtxt(path+'d0.dat')[:-1]
    Delta_0 = ChiPhiFunc(np.array([d0]))

    d11c = np.loadtxt(path+'d11c.dat')[:-1]
    d11s = np.loadtxt(path+'d11s.dat')[:-1]
    Delta_1 = ChiPhiFunc(np.array([
        d11s,
        d11c
    ]), fourier_mode = True)

    d20c = np.loadtxt(path+'d20c.dat')[:-1]
    d22c = np.loadtxt(path+'d22c.dat')[:-1]
    d22s = np.loadtxt(path+'d22s.dat')[:-1]
    Delta_2 = ChiPhiFunc(np.array([d22s, d20c, d22c]), fourier_mode = True)
    Delta_coef_cp = ChiPhiEpsFunc([Delta_0, Delta_1, Delta_2])

    # P_perp --------------------------------------
    p0 = np.loadtxt(path+'p0.dat')[:-1]
    p_perp_0 = ChiPhiFunc(np.array([p0]))

    pc1 = np.loadtxt(path+'pc1.dat')[:-1]
    p_perp_1 = ChiPhiFunc(np.array([
        np.zeros_like(pc1),
        pc1
    ]), fourier_mode = True)

    p20c = np.loadtxt(path+'p20c.dat')[:-1]
    p22s = np.loadtxt(path+'p22s.dat')[:-1]
    p22c = np.loadtxt(path+'p22c.dat')[:-1]
    p_perp_2 = ChiPhiFunc(np.array([p22s,p20c,p22c]), fourier_mode = True)
    p_perp_coef_cp = ChiPhiEpsFunc([p_perp_0, p_perp_1, p_perp_2])

    # B psi ---------------------------------------
    Bp0 = np.loadtxt(path+'Bp0.dat')[:-1]
    B_psi_0 = ChiPhiFunc(np.array([Bp0]))


    Bpc11 = np.loadtxt(path+'Bpc11.dat')[:-1]
    Bps11 = np.loadtxt(path+'Bps11.dat')[:-1]
    B_psi_1 = ChiPhiFunc(np.array([
        Bps11,
        Bpc11
    ]), fourier_mode = True)

    B_psi_coef_cp = ChiPhiEpsFunc([B_psi_0, B_psi_1])

    # B theta ---------------------------------------
    Btc20 = np.loadtxt(path+'Btc20.dat')[:-1]
    B_theta_2 = ChiPhiFunc(np.array([
        Btc20
    ]), fourier_mode = True)
    B_theta_coef_cp = ChiPhiEpsFunc([0, 0, B_theta_2])

    # X ---------------------------------------
    X20 = np.loadtxt(path+'X20c.dat')[:-1]
    X22c = np.loadtxt(path+'X22c.dat')[:-1]
    X22s = np.loadtxt(path+'X22s.dat')[:-1]
    X2 = ChiPhiFunc(np.array([
        X22s,
        X20,
        X22c
    ]), fourier_mode = True)

    X31c = np.loadtxt(path+'X31c.dat')[:-1]
    X31s = np.loadtxt(path+'X31s.dat')[:-1]
    X33c = np.loadtxt(path+'X33c.dat')[:-1]
    X33s = np.loadtxt(path+'X33s.dat')[:-1]
    X3 = ChiPhiFunc(np.array([
        X33s,
        X31s,
        X31c,
        X33c
    ]), fourier_mode = True)

    Xc1 = np.loadtxt(path+'Xc1.dat')[:-1]
    X1 = ChiPhiFunc(np.array([
        np.zeros_like(Xc1), # sin coeff is zero
        Xc1,
    ]), fourier_mode = True)

    X_coef_cp = ChiPhiEpsFunc([0, X1, X2, X3])

    # Y ---------------------------------------
    Y20 = np.loadtxt(path+'Y20.dat')[:-1]
    Y22c = np.loadtxt(path+'Y22c.dat')[:-1]
    Y22s = np.loadtxt(path+'Y22s.dat')[:-1]
    Y2 = ChiPhiFunc(np.array([
        Y22s,
        Y20,
        Y22c
    ]), fourier_mode = True)

    Y31c = np.loadtxt(path+'Y31c.dat')[:-1]
    Y31s = np.loadtxt(path+'Y31s.dat')[:-1]
    Y33c = np.loadtxt(path+'Y33c.dat')[:-1]
    Y33s = np.loadtxt(path+'Y33s.dat')[:-1]
    Y3 = ChiPhiFunc(np.array([
        Y33s,
        Y31s,
        Y31c,
        Y33c
    ]), fourier_mode = True)

    Ys1 = np.loadtxt(path+'Ys1.dat')[:-1]
    Yc1 = np.loadtxt(path+'Yc1.dat')[:-1]
    Y1 = ChiPhiFunc(np.array([
        Ys1, # sin coeff is zero
        Yc1,
    ]), fourier_mode = True)

    Y_coef_cp = ChiPhiEpsFunc([0, Y1, Y2, Y3])

    # Z ---------------------------------------
    Z20 = np.loadtxt(path+'Z20.dat')[:-1]
    Z22c = np.loadtxt(path+'Z22c.dat')[:-1]
    Z22s = np.loadtxt(path+'Z22s.dat')[:-1]
    Z2 = ChiPhiFunc(np.array([
        Z22s,
        Z20,
        Z22c
    ]), fourier_mode = True)

    Z31c = np.loadtxt(path+'Z31c.dat')[:-1]
    Z31s = np.loadtxt(path+'Z31s.dat')[:-1]
    Z33c = np.loadtxt(path+'Z33c.dat')[:-1]
    Z33s = np.loadtxt(path+'Z33s.dat')[:-1]
    Z3 = ChiPhiFunc(np.array([
        Z33s,
        Z31s,
        Z31c,
        Z33c
    ]), fourier_mode = True)

    Z_coef_cp = ChiPhiEpsFunc([0, 0, Z2, Z3])

    nfp, Xi_0, eta, B20, B22c, B22s, B31c, B31s, B33c, B33s, Ba0, Ba1 = np.loadtxt(path+'inputs.dat')
    nfp=int(nfp)

    B_alpha_e = ChiPhiEpsFunc([Ba0, Ba1])

    kap_p = ChiPhiFunc(np.array([np.loadtxt(path+'kappa.dat')[:-1]]))
    tau_p = ChiPhiFunc(np.array([np.loadtxt(path+'tau.dat')[:-1]]))

    B2 = ChiPhiFunc(np.array([
        [B22s],
        [B20],
        [B22c]
    ]), fourier_mode = True)
    B3 = ChiPhiFunc(np.array([
        [B33s],
        [B31s],
        [B31c],
        [B33c]
    ]), fourier_mode = True)
    # B1 is given by first order jacobian equation, above 14 in the first
    # half of the 2-part paper
    B_denom_coef_c = ChiPhiEpsFunc([1, -X1*2*1*kap_p, B2, B3])

    iota_e = ChiPhiEpsFunc(list(np.loadtxt(path+'outputs.dat')))

    # Not an actual representation in pyQSC.
    # only for calculating axis length.
    rc, rs = eduardo_to_matt(R_array, nfp)
    zc, zs = eduardo_to_matt(Z_array, nfp)
    stel = Qsc(rc=rc, rs=rs, zc=zc, zs=zs, nfp=nfp)
    dl_p = stel.axis_length/(2*np.pi)
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
        nfp,
        Xi_0,
        eta,
        evaluate_ChiPhiEpsFunc(B_denom_coef_c),
        evaluate_ChiPhiEpsFunc(B_alpha_e),
        kap_p.content,
        tau_p.content
    )

    return(
        B_psi_coef_cp, B_theta_coef_cp,
        Delta_coef_cp, p_perp_coef_cp,
        X_coef_cp, Y_coef_cp, Z_coef_cp,
        iota_e, dl_p,
        nfp, Xi_0, eta,
        B_denom_coef_c,
        B_alpha_e,
        kap_p, tau_p
    )

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

# This file implements and tests recursion relations

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
chi = np.linspace(0, 2*np.pi, n_grid_chi)
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
    new_list = []
    for item in chiphepsfunc_in.chiphifunc_list:
        if isinstance(item, ChiPhiFunc):
            new_list.append(evaluate_ChiPhiFunc(items))
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

# Plots the content of two chiphifuncs and compare.
def compare_chiphifunc(guess, ans):
    print('Guess')
    ax1 = plt.subplot(121)
    ax1.plot(np.real(guess.content).T)
    ax1.set_title('Real')
    ax2 = plt.subplot(122)
    ax2.plot(np.imag(guess.content).T)
    ax2.set_title('Imaginary')
    plt.show()
    print('Answer')
    ax1 = plt.subplot(121)
    ax1.plot(np.real(ans.content).T)
    ax1.set_title('Real')
    ax2 = plt.subplot(122)
    ax2.plot(np.imag(ans.content).T)
    ax2.set_title('Imaginary')
    plt.show()

    print('Error')
    ax1 = plt.subplot(121)
    ax1.plot(np.real((ans-guess).content).T)
    ax1.set_title('Real')
    ax2 = plt.subplot(122)
    ax2.plot(np.imag((ans-guess).content).T)
    ax2.set_title('Imaginary')
    plt.show()
    (ans-guess).display()

    print('fractional errors b/w data and general formula')
    print_fractional_error(guess.content, ans.content)


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

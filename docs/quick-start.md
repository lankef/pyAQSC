# Quick start

## Circular axis
The simplest equilibrium to construct with pyAQSC is the circular-axis equilibrium in [Ref.3](https://aip.scitation.org/doi/10.1063/5.0027575). Run:

    import aqsc
    new_equilibrium = aqsc.circular_axis()

This constructs the circular axis, anisotropic pressure equilibrium to order $n=2$ as an `aqsc.Equilibrium`.

To list all available unknowns, constants and axis information, run

    new_equilibrium.unknown.keys()
    new_equilibrium.constant.keys()
    new_equilibrium.axis_info.keys()

To plot the power series coefficients of $B_\theta(\psi, \chi, \phi)$ in $(\chi, \phi)$, run

    new_equilibrium.unknown['B_theta_coef_cp'][2].display()

To save and load results to and from a `.npy` file, run

    new_equilibrium.save(file_name='circular_axis_saved')
    new_equilibrium2 = aqsc.Equilibrium.load(file_name='circular_axis_saved')

Please continue to the [next part](init-and-iterate-eq.md) for constructing new cases and to higher orders.

## Very important notes on JAX's quirk

The flag for GPU-accelerated JIT compilation is disabled by default, and need to be enabled when needed by the user. Note an important characteristic of JIT compilation via JAX before starting a run with your own parameters:

-  When running the same funtion, JAX **_recompiles_** for every combination of **_traced variable shape_** and **_static variable value_**. 

Whether a parameter is traced or static will be listed in its [API](api-chiphifunc.md).

The number of terms in the governing equation scales by $O(n^3)$. On an Nvidia V100 GPU, compiling the $n=4$ iteration takes ~13 mins, while compiling the $n=6$ iteration takes 1.5hr. (The series reaches optimal truncation at $n\leq4$, so such high $n$ is not always necessary) Compiling reduces the evaluation time for an $n=5$ case from minutes to <100ms.

Be mindful of the paramters' type and shape when performing scans. 

### What are traced and static variables?

JAX is a language for expressing and composing transformations. It traces operations between traced variables with numpy syntax to perform JIT compile and auto differentiation. For more info, see [JAX documentation](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html).

Traced variables:
- Are behind all physical quantities and some numerical settings in pyAQSC.
- Can become independent variables of auto-differentiation.
- Supports most numpy operations (`jax.numpy`) but are shape-immutable.
- Not known are compile-time, and cannot be used with conditions like `==`, loops, or as array index.
  
Static variables
- Are behind some numerical settings in pyAQSC.
- Mostly `int`'s and `bool`'s in pyAQSC
- Cannot become independent variables in auto-differentiation.
- Known at compile-time, and can be used for conditions, loops and indexing.

A numba backend without this quirk that compiles instantly but takes ~30s per $n=5$ case may come later.
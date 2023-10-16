# GPU and auto-differentiation

## Introduction

pyAQSC uses JAX for auto-differentiation and just-in-time (JIT) compilation. Compiled functions run on GPU/TPU with substantial speed up (~30s/case to ~10ms/case). Most functions in pyAQSC supports JIT.

Time required to JIT an iteration step scales $O(n^3)$. Iteration steps beyond $n>4$ can take **hours** to compile. By default, JIT is **disabled** for all pyAQSC functions, and we do not recommend enabling on CPU runs.

## JIT for GPU acceleration

Use `jax.jit()` to enable JIT compile for any given function:

```
import jax
import aqsc
leading_orders_compiled = jax.jit(
    aqsc.leading_orders,
    static_argnums = (0,15,16)
)
```

JAX makes the distinctions between "traced" and "static" variables when compiling. 

- Think of "traced" variables (usually `float` or `complex`) as the function's independent variables. Derivatives w.r.t. these variables can be taken with `jax.grad()`. Changing their values does not require recompile.

- Think of "static" variables (usually `int` or `bool`) as immutable runtime parameters (grid sizes, filter cutoff, ...). Derivatives **cannot** be taken w.r.t. static variables. Changing their values requires recompile.

The user must supply the position of static arguments (`static_argnums`) when calling `jax.jit`. See [iteration tutorials](init-and-iterate-eq.md##creating-new-configrations) and API guides for the list of traced/static arguments for each function. If the distinction is not listed, the function does not support JIT.

## Auto-differentiation and vectorization
`jax.grad()` auto-differentiates a scalar function w.r.t. multiple traced variables. The below example differentates $\psi_{crit}$, the critical $\frac{\text{toroidal flux}}{2\pi}$ where near axis expansion fails and flux surface become self-intersecting.
```
# Define a function that calculates critical psi for a circular axis
# case with given axis major radius
def psi_crit_R0_circ_axis(R0):
    # Axis harmonics
    Rc, Rs = ([R0, 0, 0.0001], [0, 0, 0])
    Zc, Zs = ([0, 0, 0], [0, 0, 0.001])
    phis = jnp.linspace(0,2*jnp.pi*0.999,1000)
    # The rest of info on the circular-axis case
    equil = aqsc.leading_orders(
        nfp=1,
        Rc=Rc,
        Rs=Rs,
        Zc=Zc,
        Zs=Zs,
        p0=1+0.1*jnp.cos(phis)+0.1*jnp.cos(2*phis),
        Delta_0_avg=0,
        iota_0=0.52564852,
        B_theta_20_avg=1.5125089,
        B_alpha_1=0.1,
        B0=1,
        B11c=-1.8,
        B22c=0.01, B20=0.01, B22s=0.01,
        len_phi=1000,
        static_max_freq=(15, 20),
        traced_max_freq=(15.0, 20.0)
    )
    return(equil.get_psi_crit()[0])

# Vectorize and auto-differentiate the function.
# JIT compile takes too long on CPU.
d_psi_d_R0_circ_axis_arr = jax.grad(jax.vmap(psi_crit_R0_circ_axis))

# Running param sweep
r_list = jnp.linspace(0.8, 1.2, 10)
dpsi_crit = d_psi_d_R0_circ_axis_arr(r_list)

# Plotting critical psi gradient
plt.plot(r_list, dpsi_crit)
plt.xlabel('Axis major radius $R_0$')
plt.ylabel('r$\frac{d\psi_{crit}}{dR_0}$')
plt.savefig('psi_crit_diff')
```
Time taken to compile and run this sweep on an i9-9880H cpu Macbook is 


## Cautions
- A function will be recompiled when:
    - The values of static arguments change
    - The array shape or dtype of traced arguments change

    Make sure both conditions stay constant during parameter sweeps.

- Compile happens when the function is called for the first time. Re-run the function to measure actual run time. 

- JAX uses a linear-algebra-specific compiler that analyzes and simplifies code structure. As a result, compiling a daughter function will **not** speed up compile for a parent functions that calls it.

See [JAX homepage](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) for more information.


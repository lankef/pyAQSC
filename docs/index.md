# pyAQSC
Welcome to pyAQSC documentations!

pyAQSC directly constructs globally quasi-symmetric, anisotropic stellarator equilibria without costly optimization. 

The introduction of anisotropy $\Delta$ circumvents the Garren-Boozer Conundrum and allows the system to be solved to any order. This allows studies on higher order quantities and optimal truncation.

pyAQSC is written with optimization and data-driven studies in mind. The construction procedure is GPU accelerated and is fully auto-differentiable.


## Dependencies
pyAQSC requires Numpy, Matplotlib, and JAX.

The Maxima notebooks requires wxMaxima to view. The notebooks are not required to
run pyAQSC, but contains source expressions a large portion of the code is parsed from.

## Governing equations
The governing equations solved by pyAQSC are:

- The Jacobian equation 
$$
    J^{-1} = (\nabla\psi\times\nabla\theta\cdot\nabla\phi)
     = \frac{B^2}{B_\alpha^2}
$$
- The co/contravariant equation
$$
    \textbf{B}=B_\theta\nabla\chi
    +(B_\alpha-\bar{\iota}B_\theta)\nabla\phi
    +B_\psi\nabla\psi\\
    =\nabla\psi\times\nabla\chi
    +\bar{\iota}\nabla\phi\times\nabla\psi. 
$$
- The force balance equation
$$
    \textbf{j}\times\textbf{B} = \nabla\cdot\Pi = \nabla\cdot(\Delta\textbf{bb} + p_\perp \mathbb{I}).
$$
The first two equations, the "magnetic equations", encodes QS and nested flux surfaceconditions. They can be solved independenctly with pyAQSC to yield a globally QS magnetic field with no force balance.

For more detail, see Ref.1-3.

## Installation
This code can currently only be used by downloading from Github and importing locally.

## Importing
To use pyaqsc, import using 
```
import aqsc
```

## References
1. [Weakly Quasisymmetric Near-Axis Solutions to all Orders](https://doi.org/10.1063/5.0076583)
2. [Solving the problem of overdetermination of quasisymmetric equilibrium solutions by near-axis expansions. I. Generalized force balance](https://doi.org/10.1063/5.0027574)
3. [Solving the problem of overdetermination of quasisymmetric equilibrium solutions by near-axis expansions. II. Circular axis stellarator solutions](https://aip.scitation.org/doi/10.1063/5.0027575)
4. [pyQSC](https://github.com/landreman/pyQSC)

5. [Direct construction of optimized stellarator shapes. Part 1. Theory in cylindrical coordinates](https://doi.org/10.1017/S0022377818001289)
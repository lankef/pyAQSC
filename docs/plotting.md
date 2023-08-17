# pyAQSC
This code constructs globally quasi-symmetric stellarator equilibria with
anisotropic pressure near axis expansion to any order.

## Dependencies
The python codes requires Numpy, Matplotlib, and JAX.

The Maxima notebooks requires wxMaxima to view. The notebooks are not required to
run the main code, but contains source expressions much of code is parsed from.
## References
1. [Weakly Quasisymmetric Near-Axis Solutions to all Orders](https://doi.org/10.1063/5.0076583)
2. [Solving the problem of overdetermination of quasisymmetric equilibrium solutions by near-axis expansions. I. Generalized force balance](https://doi.org/10.1063/5.0027574)
3. [Solving the problem of overdetermination of quasisymmetric equilibrium solutions by near-axis expansions. II. Circular axis stellarator solutions](https://aip.scitation.org/doi/10.1063/5.0027575)
4. [pyQSC](https://github.com/landreman/pyQSC)

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.

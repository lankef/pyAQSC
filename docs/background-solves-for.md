# Solves for

## Definitions
PyAQSC solves the following unknowns self-consistently. These quantities are managed by instances of `aqsc.Equilibrium`. On how to access these quantities, see [aqsc.Equilibrium API](api-equilibrium.md).

- $X_n,Y_n,Z_n(\chi, \phi)$ - The coordinate transformation between the flux coordinate and a known Frenet frame:
  $$
        \textbf{x}(\psi, \chi, \phi) = \textbf{r}_0[l(\phi)]\\
        +X(\psi, \chi, \phi) \hat{\boldsymbol{\kappa}}_0[l(\phi)]\\
        +Y(\psi, \chi, \phi) \hat{\boldsymbol{\tau}}_0[l(\phi)]\\
        +Z(\psi, \chi, \phi) \hat{\boldsymbol{b}}_0[l(\phi)].
  $$
  By solving for $X_n, Y_n, Z_n$, pyAQSC constructs the flux coordinate self-consistently with other physical quantities.
- $B_{\psi n-2}, B_{\theta n}(\psi, \chi, \phi)$ - The magnetic field component. $B_\psi$ is always at $2$ orders lower than all other outputs. 
- $p_{\perp n}, \Delta_n(\psi, \chi, \phi)$ - The perpendicular pressure and anisotropy: 
  $$
  \nabla\cdot\Pi = \nabla\cdot(\Delta\textbf{bb} + p_\perp \mathbb{I})
  $$

When solving only the magnetic equations, $B_\theta, B_\psi$ and $Y$ will be treated as free parameters.

## Expansion form
All of other quantities are expanded as:
$$
F(\psi, \chi, \phi)\\
=\sum_{n=0}^{n_{max}}\epsilon^nF_n(\chi,\phi) (\epsilon\equiv\sqrt{\psi})\\
=\sum_{n=0}^{n_{max}}\epsilon^n\sum_{m=0|1}^n e^{im} F_{n,m}(\phi) + e^{-im} F_{n,-m}(\phi)
$$


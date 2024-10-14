# Governing equations

PyAQSC expands its governing equations in the flux coordinate
$$
(\epsilon\equiv\sqrt{\psi}, \chi\equiv\theta-\frac{N}{M}\phi, \phi),
$$
where $\psi$ is the toroidal flux, $\theta$ is the poloidal angle, and $\phi$ is the toroidal angle. The coordinate transformation $\textbf{r}(\psi, \chi, \phi)$ will be self-consistently solved with other physical quantities. 

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

The last equation enforces force balance. Solving all three equations together yields a globally QS equilibrium with anisotropic pressure.

For more details, see Ref.1-3.
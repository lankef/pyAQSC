# Free parameters

## QS equilibrium 
When constructing a QS anisotropic equilibrium, PyAQSC requires the following quantities. These quantities are managed by instances of `aqsc.Equilibrium`. On how to initialize an MHD equilibrium using these parameters, see the [equilibrium tutorial](init-and-iterate-eq.md). On how to access these quantities, see [aqsc.Equilibrium API](api-equilibrium.md).

- $R_0(\Phi)$, $Z_0(\Phi)$ - The axis shape in cylindrical coordinate $(R, \Phi, Z)$.
- $p_{\perp 0,0}(\phi)$ - The on-axis pressure.
- $\bar{\Delta}_0\equiv\oint\Delta_0d\phi$ - The average on-axis anisotropy.
- $\bar{B}_{\theta 2,0}(\phi)$ - The average $\chi$-independent component of $\bar{B}_{\theta 2}$.
- $B^-(\chi, \psi)\equiv B^{-2}(\chi, \psi)$ - The magnetic field strength.
- $B_{\alpha}(\psi)$ - The flux function component of $B_\phi$.
- $\bar{\iota}(\psi)\equiv\iota(\psi)-N/M, M=1$ - The rotational transform. $N/M$ is the helicity.


| Required at    | Variables                            
| -------------- | ------------------------------------ 
| Leading order  | $R_0(\Phi), Z_0(\Phi), p_{\perp 0,0}(\phi), \bar{\Delta}_0$ 
| Every order    | $B^-_{n,m}$ 
| Every 2 orders | $B_{\alpha(n-1)/2}$ 

## QS field without force balance 

**Supports iteration in this mode. No leading order support yet.**

Because the force blance constraints are dropped, pyAQSC requires additional free parameters when constructing a QS magnetic field. On how to initialize a QS magnetic field using these parameters, see the [QS field tutorial](init-and-iterate-mag.md). On how to access these quantities, see [aqsc.Equilibrium API](api-equilibrium.md).

- $B_{\theta n,m}(\phi)$ - The $\theta$ component of the magnetic field.
- $\bar{B}_{\psi n-2,0}(\phi)$ - The $\chi$-independent part of $B_\psi$. 
- $Y_{n,0}(\phi)$ - The $\chi$-independent part of $Y_n$, part of the transformation functions that defines $(\psi, \chi, \phi)$.
- $B^-(\chi, \psi)$ - The magnetic field magnitude.
- $B_{\alpha}(\psi)$ - The flux function component of $B_\phi$.  
- $\bar\iota(\psi)$ - The rotational transform

| Required at    | Variables                            
| -------------- | ------------------------------------ 
| Leading order  | $R_0(\Phi), Z_0(\Phi)$ (pressure not modeled) 
| Every order    | $B_{\theta n,m}(\phi)$, $B_{\psi n-2,0}(\phi)$, $Y_{n,0}$, $B^-_{n,m}$ 
| Every 2 orders | $B_{\alpha(n-1)/2}$, $\bar\iota_{(n-1)/2}$

## Expansion form

$B^-$ and $\bar{\iota}$:
$$
B^-(\psi, \chi) = \sum_{n=0}^\infty\epsilon^n\sum_{m=0|1}^{m=n}B^-_{n,m}
$$

$B_\alpha$ is expanded as:
$$
B_\alpha(\psi) = \sum_{n=0}^\infty\epsilon^{2n}B_{\alpha n}
$$






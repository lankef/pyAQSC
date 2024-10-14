# Introduction

## Near-axis expansion
Near-axis expansion (NAE) can find quasi-symmetric (QS) stellarator equilibria without costly optimization. It directly constructs an equilibrium by solving a system of governing equations encoding the QS conditions, existence of nested flux surface and force balance by expanding them as a power series in effective minor radius from a known magnetic axis.

Compared to optimization, NAE:
- Is fast and non-iterative. 
- Is reasonably accurate at order $n=2$. ([Ref.6](https://iopscience.iop.org/article/10.1088/1361-6587/ab19f6))
- Has exact control over the parameters of the resulting equilibria.
- Enforces QS as a constraint, rather than objective.
- Has lower dimensionality, because QS is enforced.

This makes it uniquely suited for rapid exploration over the space of QS configurations.

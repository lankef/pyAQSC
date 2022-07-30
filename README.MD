## List of data files
1. `/general_formulae/co_eqs_n_cp_patched.lisp`: see patch_co_contra.wxmx. Further removes conditions and collects sums.
2. `/general_formulae/fourier_coeffs/co_eqs_n_cp_separated_nice.lisp`: nth order of the co-contravariant equation with "nice indices" (i,j,k instead of i#)

## List of code files
1. `near_axis_expansion.mac`: Fourier-asympotic series for the near axis expansion project
1. `fourier_asympt_series.mac`: Fourier and asymptotic series for the near axis expansion project
2. `power_matching.mac`: Methods for extracting arbitrary order power series coefficient
3. `trig_fourier_order_matching.mac`: Methods for extracting arbitrary mode sin/cos Fourier series coefficient
3. `python_parser.mac`: Methods for parsing expressions into python. Specialized for parsing summation expressions.

## List of notebooks

## List of folders
1. `general_formulae/`: Exact general formulae of the equations. Not used in deriving recursion relations.
1. `maxima_sbatch_tool/`: Helper scripts for evaluating a Maxima function on a list of arguments in parallel on slurm.
2. `parsed_expressions/`: Parsed python expressions for evaluation
3. `python_solver/`: Solver for recursion relations.
2. `general_formulae/fourier_coeffs/`: nth order mth mode coefficients for equations.

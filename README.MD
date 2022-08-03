# APES (Anisptropic Pressure Expansion for Stellarators)
This code constructs globally quasi-symmetric stellarator equilibria with
anisotropic pressure to any order. [1 - 3]


## Dependencies
The python codes requires Numpy, Scipy, Matplotlib, Numba and Joblib. In addition,
[pqQSC](https://github.com/landreman/pyQSC) [4] us used for axis shape evaluation
and some plotting features.  

The Maxima notebooks requires wxMaxima to view. The notebooks are not required for
running the main code.


## Directory structure and files
```
├── README.MD
├── maxima_recursion_relations/             -> Maxima notebooks.
│   ├── <Quantity name>.wxmx                -> Maxima notebooks for recursion relations.
│   └── general_formulae/                   -> Maxima notebooks directly expanding
│       ├── co_contravariant_eq.wxmx           the governing equations. for reference only.
│       ├── jacobian_eq.wxmx
│       └── fourier_coeffs/  
├── maxima_scripts/                         -> Maxima libraries
│   ├── near_axis_expansion.mac             -> Stores the governing equations and
│   │                                          variable dependencies.
│   ├── power_mode_matching/                -> A standalone, symbolic infinite/finite
│   │   │                                      power/Fourier series order/mode matcher.
│   │   ├── power_matching.mac              -> The power series order matcher
│   │   └── trig_fourier_order_matching.mac -> The Fourier series mode matcher
│   ├── python_parser.mac                   -> A code translating maxima summations
│   │                                          to python codes. Works with
│   │                                          python_source/math_utilities.py
│   ├── sbatch_maxima.mac                   -> A code generating slurm jobscripts for
│   │                                          evaluating maxima expressions in parallel
│   └── test_cases/                         -> Test cases for Maxima scripts.
└── python_source/                          -> Subject to changes.
    ├── chiphiepsfunc.py
    ├── chiphifunc.py
    ├── chiphifunc_test_suite.py
    ├── deconvolution.py
    ├── eval_<Quantity name>.py
    └── math_utilities.py
```

## References
1. [Weakly Quasisymmetric Near-Axis Solutions to all Orders](https://doi.org/10.1063/5.0076583)
2. [Solving the problem of overdetermination of quasisymmetric equilibrium solutions by near-axis expansions. I. Generalized force balance](https://doi.org/10.1063/5.0027574)
3. [Solving the problem of overdetermination of quasisymmetric equilibrium solutions by near-axis expansions. II. Circular axis stellarator solutions](https://aip.scitation.org/doi/10.1063/5.0027575)
4. [pyQSC](https://github.com/landreman/pyQSC)

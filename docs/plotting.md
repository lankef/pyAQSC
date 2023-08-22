# Results and plotting
All data within an `Equilibrium` are within 3 dictionaries: `aqsc.Equilibrium.unknown`, `aqsc.Equilibrium.constant` and `aqsc.Equilibrium.axis_info`. See [Equilibrium API](api-equilibrium.md) for their contents. 

An order of an `Equilibrium` can be plotted with `aqsc.Equilibrium.display_order(n)`. See [Equilibrium API](api-equilibrium.md) for its usage.

A `ChiPhiFunc` can be plotted with `aqsc.ChiPhiFunc.display()` and `aqsc.ChiPhiFunc.display_content()`, or converted to callables with `aqsc.ChiPhiFunc.get_lambda()`. See [ChiPhiFunc API](api-chiphifunc.md) for their usage.

Two `ChiPhiFunc`'s can be compared with `aqsc.compare_chiphifunc()`. See [utility API](api-utils.md) for its usage.
# Geometric Multigrid

Working on high speed solutions for variable coefficient Poisson equations on Cartesian grids.


| ![mg solver benchmark](benchmark/MGscaling.png) | 
|:--:| 
| Median time [benchmark example](benchmark/benchmark.jl) for `mg!` solver demonstrating `NlogN` scaling with the length of the solution vector. The dots labeled `psueo` utilize the pseudo-inverse
smoother designed using AD to acheive approximately 50% speed up with Float32.|
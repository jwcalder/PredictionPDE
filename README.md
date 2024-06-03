# PredictionPDE

This python package reproduces numerical experiments from the paper 

J. Calder, N. Drenska, D. Mosaphir [Numerical solution of a PDE arising from prediction with expert advice](https://arxiv.org/abs/), arXiv preprint, 2024.

The code numerically solves a degenerate elliptic PDE arising in the problem of prediction from expert advise in relatively high dimensions. There is code for solving the PDE on a regular dense grid, which works for low dimensions, and on a sparse grid that works in relatively high dimensions.

## Full computational grid

The code for the regular dense grid is mainly contained in `solvers.py`. The scripts `test_2d.py`, `test_3d.py`, and `test_4d.py` show how to run the solvers and generate the figures from the paper for the 2,3 and 4 expert problems (the n expert problem involves solving a PDE in n-1 dimensions). The `convergence_rates.py` script generates plots showing the convergence rates in these cases, where the solution of the PDE is known. The solutions are stored in the folder `solutions/` for several different dimensions and grid resolutions. These will be loaded automatically by the code, instead of computing them, when available. 

Note that many of these solutions take up far more space than the 100MB limit in github, so they have not been included in the repository. You can reference the `.gitignore` file for a list of files that are not included. The authors are happy to provide any of these files directly to anyone upon request.

## Sparse computational grid

The code for the sparse sector grid solvers is mainly contained in `sparse_solvers.py`. The script `sparse_test.py` shows how to run the solver and generate the figures from the paper. The script `num_grid_points.py` prints out the number of grid points used by the sparse solver, compared to what the dense solver would use. The file `plots.py` contains plotting functionality used by many scripts. 

As before, there are many sparse solutions that take up to 10GB to store and cannot be included in the repository. See the `.gitignore` file for a list of additional files that are available upon request.

Any questions should be sent to Jeff Calder (`jwcalder@umn.edu`).




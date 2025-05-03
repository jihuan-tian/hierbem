clear all;
## Load reference solution.
load "/home/jihuan/Projects/hierbem/tests/functions/laplace-bem/dirichlet-full-matrix-complex/reference.output";
solution_ref = solution;
## Load computation solution
load "dirichlet-full-matrix-complex.output";

solution_l2_rel_err = norm(solution - solution_ref) / norm(solution_ref)
solution_inf_rel_err = norm(solution - solution_ref, Inf) / norm(solution_ref, Inf)

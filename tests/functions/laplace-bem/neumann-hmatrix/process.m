clear all;
## Load reference solution.
load "/home/jihuan/Projects/hierbem/tests/functions/laplace-bem/neumann-hmatrix/reference.output";
solution_ref = solution;
## Load computation solution
load "neumann-hmatrix.output";

solution_l2_rel_err = norm(solution - solution_ref) / norm(solution_ref)
solution_inf_rel_err = norm(solution - solution_ref, Inf) / norm(solution_ref, Inf)

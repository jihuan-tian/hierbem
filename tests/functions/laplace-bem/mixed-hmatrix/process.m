clear all;
## Load reference solution.
load "/home/jihuan/Projects/hierbem/tests/functions/laplace-bem/mixed-hmatrix/reference.output";
solution_ref = solution_on_combined_domain_internal_dof_numbering;
## Load computation solution
load "mixed-hmatrix.output";
solution = solution_on_combined_domain_internal_dof_numbering;

solution_l2_rel_err = norm(solution - solution_ref) / norm(solution_ref)
solution_inf_rel_err = norm(solution - solution_ref, Inf) / norm(solution_ref, Inf)

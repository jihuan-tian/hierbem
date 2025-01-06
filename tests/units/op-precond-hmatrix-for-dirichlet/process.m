clear all;
## Load reference solution.
load "/home/jihuan/Projects/hierbem/tests/units/op-precond-hmatrix-for-dirichlet/reference.output";
Br_ref = Br;
## Load computation results.
load "op-precond-hmatrix-for-dirichlet.output";

rel_err = norm(Br - Br_ref, 'fro') / norm(Br_ref, 'fro')

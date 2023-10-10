clear all;
load lapack-matrix-local-to-rkmatrix.output;

tau_subset = [7,8,9,10]+1;
sigma_subset = [10,11,12]+1;
M_b_submatrix = M(tau_subset, sigma_subset);

norm(M_b_submatrix - rkmat_no_trunc.A * rkmat_no_trunc.B', 'fro') / norm(M_b_submatrix, 'fro')

norm(M_b_submatrix - rk1mat.A * rk1mat.B', 'fro') / norm(M_b_submatrix, 'fro')

clear all;
load "rkmatrix-truncate-to-rank-with-error-matrices.output";

## rank(M) is 2.
rank(M)
norm(M - A.A * A.B', "fro") / norm(M, "fro")
norm(M - A_trunc_to_1.A * A_trunc_to_1.B', "fro") / norm(M, "fro")
norm(M - A_trunc_to_2.A * A_trunc_to_2.B', "fro") / norm(M, "fro")
norm(M - A_trunc_to_3.A * A_trunc_to_3.B', "fro") / norm(M, "fro")

## Check the sum of A*B^T and C*D^T when the truncation rank is 1.
norm(M - (A_trunc_to_1.A * A_trunc_to_1.B' + C_trunc_to_1 * D_trunc_to_1'), 'fro') / norm(M, 'fro')

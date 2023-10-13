clear all;
load hmatrix-rkmatrix-mmult.output;

## Even though M2 has a rank 20, there are only two significant
## singular values.
rank(M2)
bar(svd(M2))
## The rank of the rank-k matrix is 2, which approximates M2 very
## well.
rank(M2_rk.A * M2_rk.B')
norm(M2 - M2_rk.A * M2_rk.B', 'fro') / norm(M2, 'fro')

M = M1 * M2;
norm(M - M_rk.A * M_rk.B', 'fro') / norm(M, 'fro')

clear all;

load_packages;
load hmatrix-solve-cholesky.output;
load M.dat;
load b.dat;

## Restore the matrix M to be symmetric.
MM = tril2fullsym(M);

## Calculate relative error between H-Matrix and full matrix based on
## Frobenius-norm. Only the lower triangular part is compared.
hmat_rel_err = norm(H_full - M, 'fro') / norm(M, 'fro')

## Calculate the relative error between L*L^T and the original symmetric matrix.
product_hmat_rel_err = norm(L_full*L_full' - MM, 'fro') / norm(MM, 'fro')

## Calculate the relative error between H-matrix and full matrix solution
## vectors.
x_octave = MM \ b;
x_rel_err = norm(x_octave - x, 2) / norm(x_octave, 2)

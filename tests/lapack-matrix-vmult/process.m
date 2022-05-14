clear all;
load lapack-matrix-vmult.output;

## Get the lower triangular part of the matrix and generate a
## symmetric matrix.
L = tril(M);
MM = tril2fullsym(L);

printout_var("norm(M*x-y,2)/norm(M*x,2)");
printout_var("norm(MM*x-z,2)/norm(MM*x,2)");

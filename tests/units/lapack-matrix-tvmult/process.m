clear all;
load lapack-matrix-tvmult.output;

## Get the lower triangular part of the matrix and generate a
## symmetric matrix.
L = tril(M);
U = triu(M);
MM = tril2fullsym(L);

printout_var("norm(M'*x-y,2)/norm(M'*x,2)");
printout_var("norm(MM*x-z,2)/norm(MM*x,2)");
printout_var("norm(L'*x-z1,2)/norm(z1,2)");
printout_var("norm(U'*x-z2,2)/norm(z2,2)");

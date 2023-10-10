## Generate the lower triangular part of the positive definite and
## symmetric matrix.
p = 5;
n = 2^p;
M = randn(n);
M = M * M';
M = tril(M);
b = randn(n, 1);
save("-text", "M.dat", "M");

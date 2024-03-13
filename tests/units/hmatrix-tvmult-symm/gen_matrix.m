## Generate the lower triangular part of the symmetric positive definite matrix.
p = 6;
n = 2^p;
M = randn(n);
M = M * M';
M = tril(M);
x = randn(n, 1);
save("-text", "M.dat", "M");
save("-text", "x.dat", "x");

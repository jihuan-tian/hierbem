## Generate the lower triangular part of the symmetric positive definite matrix.
p = 6;
n = 2^p;
M = randn(n);
M = tril(M * transpose(M));
x = randn(n, 1);
M_complex = complex(randn(n), randn(n));
M_complex = tril(M_complex * transpose(M_complex));
x_complex = complex(randn(n, 1), randn(n, 1));
save("-text", "M.dat", "M", "M_complex");
save("-text", "x.dat", "x", "x_complex");

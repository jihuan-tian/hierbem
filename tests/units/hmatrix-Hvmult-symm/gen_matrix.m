## Generate the lower triangular part of the Hermite symmetric matrix.
p = 6;
n = 2^p;
M_complex = complex(randn(n), randn(n));
M_complex = tril(M_complex * transpose(M_complex));
x_complex = complex(randn(n, 1), randn(n, 1));
save("-text", "M.dat", "M_complex");
save("-text", "x.dat", "x_complex");

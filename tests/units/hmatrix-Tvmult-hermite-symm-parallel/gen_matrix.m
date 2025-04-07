## Generate the lower triangular part of the Hermite symmetric matrix.
p = 6;
n = 2^p;
M_complex = complex(randn(n), randn(n));
M_complex = tril(M_complex * ctranspose(M_complex));
x_complex = complex(randn(n, 1), randn(n, 1));
y0_complex = complex(randn(n, 1), randn(n, 1));
save("-text", "M.dat", "M_complex");
save("-text", "xy.dat", "x_complex", "y0_complex");

## Generate an upper triangular matrix.
p = 6;
n = 2^p;
M = triu(randn(n));
x = randn(n, 1);
y0 = randn(n, 1);
M_complex = triu(complex(randn(n), randn(n)));
x_complex = complex(randn(n, 1), randn(n, 1));
y0_complex = complex(randn(n, 1), randn(n, 1));
save("-text", "M.dat", "M", "M_complex");
save("-text", "xy.dat", "x", "y0", "x_complex", "y0_complex");

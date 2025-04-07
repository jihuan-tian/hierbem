## Generate an upper triangular matrix.
p = 6;
n = 2^p;
M_complex = triu(complex(randn(n), randn(n)));
x_complex = complex(randn(n, 1), randn(n, 1));
y0_complex = complex(randn(n, 1), randn(n, 1));
save("-text", "M.dat", "M_complex");
save("-text", "xy.dat", "x_complex", "y0_complex");

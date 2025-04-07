## Generate an upper triangular matrix.
p = 6;
n = 2^p;
M = triu(randn(n));
x = randn(n, 1);
M_complex = triu(complex(randn(n), randn(n)));
x_complex = complex(randn(n, 1), randn(n, 1));
save("-text", "M.dat", "M", "M_complex");
save("-text", "x.dat", "x", "x_complex");

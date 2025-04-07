## Generate a general matrix.
p = 6;
n = 2^p;
M_complex = complex(randn(n), randn(n));
x_complex = complex(randn(n, 1), randn(n, 1));
save("-text", "M.dat", "M_complex");
save("-text", "x.dat", "x_complex");

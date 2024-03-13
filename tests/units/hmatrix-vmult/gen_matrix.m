## Generate a general matrix.
p = 6;
n = 2^p;
M = randn(n);
x = randn(n, 1);
save("-text", "M.dat", "M");
save("-text", "x.dat", "x");

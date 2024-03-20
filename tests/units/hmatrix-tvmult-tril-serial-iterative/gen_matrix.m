## Generate a lower triangular matrix.
p = 6;
n = 2^p;
M = tril(randn(n));
x = randn(n, 1);
y0 = randn(n, 1);
save("-text", "M.dat", "M");
save("-text", "xy.dat", "x", "y0");


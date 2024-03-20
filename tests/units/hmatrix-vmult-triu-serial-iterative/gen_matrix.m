## Generate an upper triangular matrix.
p = 6;
n = 2^p;
M = triu(randn(n));
x = randn(n, 1);
y0 = randn(n, 1);
save("-text", "M.dat", "M");
save("-text", "xy.dat", "x", "y0");

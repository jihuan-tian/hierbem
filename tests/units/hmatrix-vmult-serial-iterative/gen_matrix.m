## Generate a general matrix.
p = 6;
n = 2^p;
M = randn(n);
x = randn(n, 1);
y0 = randn(n, 1);
save("-text", "M.dat", "M");
save("-text", "xy.dat", "x", "y0");

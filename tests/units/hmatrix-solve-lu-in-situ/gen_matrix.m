p = 5;
n = 2^p;
M = randn(n);
b = randn(n, 1);
save("-text", "M.dat", "M");
save("-text", "b.dat", "b");

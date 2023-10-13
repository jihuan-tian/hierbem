p = 5;
n = 2^p;
M = randn(n);
L = tril(M);
save("-text", "L.dat", "L");

b = rand(n, 1);
save("-text", "b.dat", "b");

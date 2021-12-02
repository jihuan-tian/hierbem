p = 5;
n = 2^p;
M = randn(n);
L = tril(M);
save("-text", "L.dat", "L");

Z = rand(n);
save("-text", "Z.dat", "Z");

p = 5;
M = randn(2^p);
L = tril(M);
save("-text", "L.dat", "L");

b = rand(2^p, 1);
save("-text", "b.dat", "b");

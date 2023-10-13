p = 5;
n = 2^p;
M = randn(n);
U = triu(M);
save("-text", "U.dat", "U");

b = rand(n, 1);
save("-text", "b.dat", "b");

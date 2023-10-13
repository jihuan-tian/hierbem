p = 5;
n = 2^p;
M = randn(n);
U = triu(M);
save("-text", "U.dat", "U");

Z = rand(n);
save("-text", "Z.dat", "Z");

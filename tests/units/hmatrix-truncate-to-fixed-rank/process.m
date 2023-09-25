clear all;
load "hmatrix-truncate-to-fixed-rank.output";

norm(M - M_prime, "fro") / norm(M, "fro")
figure;
imagesc(M)
figure;
imagesc(M_prime)

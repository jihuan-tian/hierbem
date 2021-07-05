clear all;
load svd.output;

m = 3;
n = 5;

## Construct the Sigma matrix from Sigma_r.
Sigma = [diag(Sigma_r1), zeros(m, n - m)];
norm(A - U1 * Sigma * VT1, "fro") / norm(A, "fro")

Sigma = [diag(Sigma_r2), zeros(m, n - m)];
norm(A - U2 * Sigma * VT2, "fro") / norm(A, "fro")

Sigma = [diag(Sigma_r3), zeros(m, n - m)];
norm(A - U3 * Sigma * VT3, "fro") / norm(A, "fro")

Sigma = [diag(Sigma_r4), zeros(m, n - m)];
norm(A - U4 * Sigma * VT4, "fro") / norm(A, "fro")

Sigma = [diag(Sigma_r5), zeros(m, n - m)];
norm(A - U5 * Sigma * VT5, "fro") / norm(A, "fro")

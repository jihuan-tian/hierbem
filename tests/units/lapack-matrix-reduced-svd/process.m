clear all;
load lapack-matrix-reduced-svd.output;

norm(A - U1 * diag(Sigma_r1) * VT1, "fro") / norm(A, "fro")
norm(A - U2 * diag(Sigma_r2) * VT2, "fro") / norm(A, "fro")
norm(A - U3 * diag(Sigma_r3) * VT3, "fro") / norm(A, "fro")
norm(A - U4 * diag(Sigma_r4) * VT4, "fro") / norm(A, "fro")

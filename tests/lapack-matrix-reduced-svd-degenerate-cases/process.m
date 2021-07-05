clear all;
load lapack-matrix-rsvd-degenerate-cases.output;

norm(A1 - U1 * diag(Sigma_r1) * VT1, "fro") / norm(A1, "fro")
norm(A2 - U2 * diag(Sigma_r2) * VT2, "fro") / norm(A2, "fro")
norm(A3 - U3 * diag(Sigma_r3) * VT3, "fro") / norm(A3, "fro")

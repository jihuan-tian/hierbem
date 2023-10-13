clear all;
load lapack-matrix-rank-k-decompose.output1;

norm(M - A1 * B1', "fro") / norm(M, "fro")
norm(M - A2 * B2', "fro") / norm(M, "fro")
norm(M - A3 * B3', "fro") / norm(M, "fro")
norm(M - A4 * B4', "fro") / norm(M, "fro")

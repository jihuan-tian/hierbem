clear all;
load "rkmatrix-truncate-to-rank.output";

norm(M - A.A * A.B', "fro") / norm(M, "fro")
norm(M - A_trunc_to_1.A * A_trunc_to_1.B', "fro") / norm(M, "fro")
norm(M - A_trunc_to_2.A * A_trunc_to_2.B', "fro") / norm(M, "fro")
norm(M - A_trunc_to_3.A * A_trunc_to_3.B', "fro") / norm(M, "fro")

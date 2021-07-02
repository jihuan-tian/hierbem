clear all;
load "rkmatrix-add-formatted.output";

norm(M1 - A.A * A.B', "fro") / norm(M1, "fro")
norm(M2 - B.A * B.B', "fro") / norm(M2, "fro")

M = M1 + M2;
formatted_addition_error(1) = norm(M - C_trunc_1.A * C_trunc_1.B', "fro") / norm(M, "fro")
formatted_addition_error(2) = norm(M - C_trunc_2.A * C_trunc_2.B', "fro") / norm(M, "fro")
formatted_addition_error(3) = norm(M - C_trunc_3.A * C_trunc_3.B', "fro") / norm(M, "fro")

bar(formatted_addition_error);
xlabel("Truncation rank")
ylabel("Relative error")
title("Formatted addition error in Frobenius norm");


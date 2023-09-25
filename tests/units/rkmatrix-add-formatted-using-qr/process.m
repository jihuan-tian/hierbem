clear all;
load "rkmatrix-add-formatted-using-qr.output";

norm(M1 - A.A * A.B', "fro") / norm(M1, "fro")
norm(M2 - B.A * B.B', "fro") / norm(M2, "fro")

M = M1 + M2;
formatted_addition_error = zeros(10, 1);
for m = 1:10
  var_name = cstrcat("C_trunc_", num2str(m));
  formatted_addition_error(m) = eval(cstrcat("norm(M - ", var_name, ".A * ", var_name, ".B', 'fro') / norm(M, 'fro')"));
endfor
formatted_addition_error

bar(formatted_addition_error);
xlabel("Truncation rank")
ylabel("Relative error")
title("Formatted addition error in Frobenius norm");


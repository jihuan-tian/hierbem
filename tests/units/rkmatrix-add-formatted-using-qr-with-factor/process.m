## Copyright (C) 2021-2023 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your
## option) any later version. The full text of the license can be found in the
## file LICENSE at the top level directory of HierBEM.

clear all;
load "rkmatrix-add-formatted-using-qr-with-factor.output";

norm(M1 - A.A * A.B', "fro") / norm(M1, "fro")
norm(M2 - B.A * B.B', "fro") / norm(M2, "fro")

b = 3.5;
M = M1 + b * M2;
formatted_addition_error = zeros(4, 1);
for m = 1:10
  var_name = cstrcat("C_trunc_", num2str(m));
  formatted_addition_error(m) = eval(cstrcat("norm(M - ", var_name, ".A * ", var_name, ".B', 'fro') / norm(M, 'fro')"))
endfor

bar(formatted_addition_error);
xlabel("Truncation rank")
ylabel("Relative error")
title("Formatted addition error in Frobenius norm");


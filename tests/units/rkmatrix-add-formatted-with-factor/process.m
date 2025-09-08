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
load "rkmatrix-add-formatted-with-factor.output";

if (isequal([A.A, B.A], C.A) && isequal([A.B, B.B], C.B))
  printf("Juxtaposition addition of matrix A and B is correct!\n");
endif

norm(M1 - A.A * A.B', "fro") / norm(M1, "fro")
norm(M2 - B.A * B.B', "fro") / norm(M2, "fro")

b = 3.5;
M = M1 + b * M2;
formatted_addition_error(1) = norm(M - C_trunc_1.A * C_trunc_1.B', "fro") / norm(M, "fro");
formatted_addition_error(2) = norm(M - C_trunc_2.A * C_trunc_2.B', "fro") / norm(M, "fro");
formatted_addition_error(3) = norm(M - C_trunc_3.A * C_trunc_3.B', "fro") / norm(M, "fro")

bar(formatted_addition_error);
xlabel("Truncation rank")
ylabel("Relative error")
title("Formatted addition error in Frobenius norm");

if (isequal(A_plus_B.A, C.A) && isequal(A_plus_B.B, C.B))
  printf("Add matrix B to matrix A itself is correct!\n");
endif

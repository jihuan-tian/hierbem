## Copyright (C) 2022-2023 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your
## option) any later version. The full text of the license can be found in the
## file LICENSE at the top level directory of HierBEM.

clear all;
load "B.dat";

for m = 1:6
  load(cstrcat("2022-03-19-aca-plus-for-full-matrix/", sprintf("%02d", m), ".dat"));
  R.rank
  A_aca_approx = R.A * R.B';
  norm(A - A_aca_approx, 'fro') / norm(A, 'fro')
endfor

## A_mem_size = size(A, 1) * size(A, 2) * 8 / 1024;
## A_aca_mem_size = (size(R.A, 1) * size(R.A, 2) + size(R.B, 1) * size(R.B, 2)) * 8 / 1024;
## figure;
## subplot(1, 2, 1);
## show_matrix(A);
## colorbar;
## title(cstrcat("A\nMemory: ", num2str(A_mem_size), " kB"));
## subplot(1, 2, 2);
## show_matrix(A_aca_approx);
## colorbar;
## title(cstrcat("A_{aca}\nMemory: ", num2str(A_aca_mem_size), " kB"));

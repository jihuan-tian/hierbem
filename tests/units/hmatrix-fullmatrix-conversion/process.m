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

load_packages;
format long;
load "hmatrix-fullmatrix-conversion.output";

N = 10;
relative_errors = zeros(N, 1);
for m = 1:N
  var_name = cstrcat("M_tilde", num2str(m));
  relative_errors(m) = eval(cstrcat("norm(M - ", var_name, ", 'fro') / norm(M, 'fro')"))
endfor

bar(relative_errors);
xlabel("Rank")
ylabel("Relative Frobenius errors");

## Plot the bct structure with ranks.
figure;
plot_bct_struct("bct-struct-with-rank.dat");

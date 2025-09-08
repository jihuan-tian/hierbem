## Copyright (C) 2023 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your
## option) any later version. The full text of the license can be found in the
## file LICENSE at the top level directory of HierBEM.

## Plot the condition numbers of the generated random matrices.

total_trials = 5;
cond_numbers = zeros(total_trials, 1);
for m = 0:(total_trials - 1)
  load(cstrcat("M", num2str(m), ".dat"));
  cond_numbers(m+1) = cond(M);
endfor

figure;
set_fig_size(gcf, [2000, 600]);
bar(0:(total_trials - 1), cond_numbers);
set(gca, 'xtick', 0:(total_trials - 1));
xlim([-1, total_trials]);
grid on;
title("Condition numbers of generated random matrices");

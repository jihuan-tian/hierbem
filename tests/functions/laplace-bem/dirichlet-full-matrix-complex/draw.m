## Copyright (C) 2025 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your
## option) any later version. The full text of the license can be found in the
## file LICENSE at the top level directory of HierBEM.

figure;
scale_fig(gcf, [2, 1])
subplot(1, 2, 1);
hold on;
plot(real(solution_ref), 'bo');
plot(real(solution), 'r.');
legend('Reference', 'Solution');
title("Real part");
grid on;
hold off;

subplot(1, 2, 2);
hold on;
plot(imag(solution_ref), 'bo');
plot(imag(solution), 'r.');
legend('Reference', 'Solution');
title("Imaginary part");
grid on;
hold off;

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
load hmatrix-add-formatted.output;

M = M1 + M2;
## Relative error between the result from formatted addition and the
## sum of the original full matrices.
norm(M - H_full, "fro") / norm(M, "fro")
## Relative error between the result from formatted addition and the
## sum of two full matrices converted from their H-matrices.
norm(H1_add_H2_full - H_full, "fro") / norm(H1_add_H2_full, "fro")

## Visualization of the error.
h = 1;
figure(h);
ax1 = get(h, "currentaxes");
imagesc(get(h, "currentaxes"), (H_full - M) / norm(M, "fro"));
cb1 = colorbar(get(h, "currentaxes"));
## Enforce square axe.
fig_pos = get(h, "position");
fig_width = fig_pos(3);
fig_height = fig_pos(4);
ax_normalized_width = 0.5;
ax_normalized_height = fig_width * ax_normalized_width / fig_height;
set(ax1, "position", [0.13000, 0.11666, ax_normalized_width, ax_normalized_height]);
xlabel("Columns");
ylabel("Rows");
title(get(h, "currentaxes"), "Relative error between the formatted\naddition and the sum of\nthe original full matrices");
PrintGCF("relative-error-wrt-original-full-matrices");

h = 2;
figure(h);
ax2 = get(h, "currentaxes");
imagesc(get(h, "currentaxes"), (H_full - H1_add_H2_full) / norm(H1_add_H2_full, "fro"));
cb2 = colorbar(get(h, "currentaxes"));
## Enforce square axe.
fig_pos = get(h, "position");
fig_width = fig_pos(3);
fig_height = fig_pos(4);
ax_normalized_width = 0.5;
ax_normalized_height = fig_width * ax_normalized_width / fig_height;
set(ax2, "position", [0.13000, 0.11666, ax_normalized_width, ax_normalized_height]);
xlabel("Columns");
ylabel("Rows");
t = title(get(h, "currentaxes"), "Relative error between the formatted addition and\nthe sum of the two full matrices\nconverted from H-matrices.");
PrintGCF("relative-error-wrt-hmatrices");

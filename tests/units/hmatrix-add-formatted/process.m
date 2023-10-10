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

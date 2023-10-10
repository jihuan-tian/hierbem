clear all;

load 'verify-sauter-quad-parametric-transform-same-panel.output';

figure('visible', 'off'); plot_quad_points(qgauss(:, 1:2), 'markersize', 6, 'labelsize', 12);
title('Kx quad points in Sauter parametric space');
# animate_mark_point(gcf, qgauss(:, 1:2), 1:81, 0.2);
PrintGCFLatex("kx-qgauss-points-in-sauter-parametric-space");

figure('visible', 'off'); plot_quad_points(qgauss(:, 3:4), 'markersize', 6, 'labelsize', 12);
title('Ky quad points in Sauter parametric space');
PrintGCFLatex("ky-qgauss-points-in-sauter-parametric-space");

figure('visible', 'off'); plot_quad_points(kx_k0, 'markersize', 6, 'labelsize', 12);
title('Kx quad points in reference cell (k3=0)');
PrintGCFLatex("kx-k0-qgauss-points-in-reference-cell");
figure('visible', 'off'); plot_quad_points(ky_k0, 'markersize', 6, 'labelsize', 12);
title('Ky quad points in reference cell (k3=0)');
PrintGCFLatex("ky-k0-qgauss-points-in-reference-cell");

figure('visible', 'off'); plot_quad_points(kx_k1, 'markersize', 6, 'labelsize', 12);
title('Kx quad points in reference cell (k3=1)');
PrintGCFLatex("kx-k1-qgauss-points-in-reference-cell");
figure('visible', 'off'); plot_quad_points(ky_k1, 'markersize', 6, 'labelsize', 12);
title('Ky quad points in reference cell (k3=1)');
PrintGCFLatex("ky-k1-qgauss-points-in-reference-cell");

figure('visible', 'off'); plot_quad_points(kx_k2, 'markersize', 6, 'labelsize', 12);
title('Kx quad points in reference cell (k3=2)');
PrintGCFLatex("kx-k2-qgauss-points-in-reference-cell");
figure('visible', 'off'); plot_quad_points(ky_k2, 'markersize', 6, 'labelsize', 12);
title('Ky quad points in reference cell (k3=2)');
PrintGCFLatex("ky-k2-qgauss-points-in-reference-cell");

figure('visible', 'off'); plot_quad_points(kx_k3, 'markersize', 6, 'labelsize', 12);
title('Kx quad points in reference cell (k3=3)');
PrintGCFLatex("kx-k3-qgauss-points-in-reference-cell");
figure('visible', 'off'); plot_quad_points(ky_k3, 'markersize', 6, 'labelsize', 12);
title('Ky quad points in reference cell (k3=3)');
PrintGCFLatex("ky-k3-qgauss-points-in-reference-cell");

figure('visible', 'off'); plot_quad_points(kx_k4, 'markersize', 6, 'labelsize', 12);
title('Kx quad points in reference cell (k3=4)');
PrintGCFLatex("kx-k4-qgauss-points-in-reference-cell");
figure('visible', 'off'); plot_quad_points(ky_k4, 'markersize', 6, 'labelsize', 12);
title('Ky quad points in reference cell (k3=4)');
PrintGCFLatex("ky-k4-qgauss-points-in-reference-cell");

figure('visible', 'off'); plot_quad_points(kx_k5, 'markersize', 6, 'labelsize', 12);
title('Kx quad points in reference cell (k3=5)');
PrintGCFLatex("kx-k5-qgauss-points-in-reference-cell");
figure('visible', 'off'); plot_quad_points(ky_k5, 'markersize', 6, 'labelsize', 12);
title('Ky quad points in reference cell (k3=5)');
PrintGCFLatex("ky-k5-qgauss-points-in-reference-cell");

figure('visible', 'off'); plot_quad_points(kx_k6, 'markersize', 6, 'labelsize', 12);
title('Kx quad points in reference cell (k3=6)');
PrintGCFLatex("kx-k6-qgauss-points-in-reference-cell");
figure('visible', 'off'); plot_quad_points(ky_k6, 'markersize', 6, 'labelsize', 12);
title('Ky quad points in reference cell (k3=6)');
PrintGCFLatex("ky-k6-qgauss-points-in-reference-cell");

figure('visible', 'off'); plot_quad_points(kx_k7, 'markersize', 6, 'labelsize', 12);
title('Kx quad points in reference cell (k3=7)');
PrintGCFLatex("kx-k7-qgauss-points-in-reference-cell");
figure('visible', 'off'); plot_quad_points(ky_k7, 'markersize', 6, 'labelsize', 12);
title('Ky quad points in reference cell (k3=7)');
PrintGCFLatex("ky-k7-qgauss-points-in-reference-cell");

close all;

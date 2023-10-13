% Clear all vars except enable_figure
clear -x enable_figure;

% XXX Use gnuplot backend to prevent crashing after plotting due to hardware-acceleration problems
graphics_toolkit gnuplot;

load "hmatrix-solve-lu.output";
load "M.dat";
load "b.dat";

% Calculate relative error between H-Matrix and full matrix based on Frobenius-norm
hmat_rel_err = norm(H_full - M, 'fro') / norm(M, 'fro')

% Calculate relative error between H-matrix and full matrix solution based on 2-norm
x_octave = M \ b;
x_rel_err = norm(x_octave - x, 2) / norm(x_octave, 2)

if (exist("enable_figure"))
    enable_figure
else
    disp('*** enable_figure undefined');
endif

% Generate image only when 'enable_figure' variable exists and is true
if (exist("enable_figure") && enable_figure)
    figure('visible', 'off');
    plot(x, 'r-', "marker", "+");
    hold on;
    plot(x_octave, 'g-', "marker", "^");
    hold off;

    print("solution.png", "-djpg");

    figure('visible', 'off');
    set_fig_size(gcf, [900, 400]);
    subplot(1, 2, 1);
    plot_bct_struct("H_bct.dat");
    title("H-matrix");
    subplot(1, 2, 2);
    plot_bct_struct("LU_bct.dat");
    title("LU factorization of H-matrix");

    print("H-matrices.png", "-djpg", "-r800");
endif

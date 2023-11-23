figure;
hold on;
plot(x_octave, 'b.');
plot(x, 'ro');
legend("x octave", "x cpp");

figure;
plot_bct_struct("H_bct.dat");

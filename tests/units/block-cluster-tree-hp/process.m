clear all;

load_packages;

figure;
subplot(2, 2, 1);
plot_bct_struct("bct1.dat", 'show_rank', false);
title("Block cluster tree #1");

subplot(2, 2, 2);
plot_bct_struct("bct2.dat", 'show_rank', false);
title("Block cluster tree #2");

subplot(2, 2, 3);
plot_bct_struct("bct3.dat", 'show_rank', false);
title("Block cluster tree #3");

subplot(2, 2, 4);
plot_bct_struct("bct4.dat", 'show_rank', false);
title("Block cluster tree #4");

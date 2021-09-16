clear all;
load lapack-matrix-inverse.output;

# M = reshape([-1, 5, 2, -3, 6, 1, -2, 4, 2, -3, -4, 1, -3, -1, 1, 2, -2, 4, 2, -1, 3, 1, -1, 3, -3, 7, 2, -3, 7, 2, -2, 2, 1, 0, 0, -1, 1, -4, 0, 0, 0, 2, 0, -2, 3, -1, -1, 6, -2, 4, 3, -2, 4, -1, -1, 3, 3, -4, -6, 1, -3, -3, 1, -2], 8, 8);
norm(M_inv - inv(M), 'fro') / norm(inv(M), 'fro')
norm(M_prime_inv - inv(M), 'fro') / norm(inv(M), 'fro')
norm(M_prime_inv - M_inv, 'fro') / norm(M_inv, 'fro')

## figure;
## set_fig_size(gcf, 1500, 800);
## center_fig;
## subplot(1, 2, 1);
## imagesc(M_inv);
## axis equal;
## axis off;
## colorbar;
## subplot(1, 2, 2);
## imagesc(inv(M));
## axis equal;
## axis off;
## colorbar;

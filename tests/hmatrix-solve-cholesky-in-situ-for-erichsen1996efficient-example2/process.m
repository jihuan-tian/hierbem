clear all;
load "input_matrices.dat";

## Process the results obtained from the batch run: @p
## batch-normal-trunc.sh and @p batch-spd-trunc.sh.
## rank_list = 1:8;
## trunc_error_norm = zeros(length(rank_list), 1);
## symmetry_checking_error_M = norm(M - M', 'fro') / norm(M, 'fro');
## symmetry_checking_error_H_full = zeros(length(rank_list), 1);
## is_positive_definite_list = zeros(length(rank_list), 1);

## for m = rank_list
##   ## fprintf(stdout(), "=== Normal truncation rank: %d ===\n", m);
##   ## load(cstrcat("slp-normal-truncation-rank=", num2str(m), ".dat"));
##   fprintf(stdout(), "=== SPD truncation rank: %d ===\n", m);
##   load(cstrcat("slp-spd-truncation-rank=", num2str(m), ".dat"));
##   is_positive_definite_list(m) = is_positive_definite(H_full);
##   trunc_error_norm(m) = norm(H_full - M, 'fro') / norm(M, 'fro');
##   symmetry_checking_error_H_full(m) = norm(H_full - H_full', 'fro') / norm(H_full, 'fro');
## endfor

## Check the standard output file.
load "hmatrix-solve-cholesky-in-situ-for-erichsen1996efficient-example2.output";
is_positive_definite(H_full)
norm(H_full - M, 'fro') / norm(M, 'fro')
norm(H_full - H_full', 'fro') / norm(H_full, 'fro')

## show_matrix(M);
## title("Single layer potential matrix");
## savefig("M.jpg");

## Compare the solution vectors.
x_accurate = M \ b;
norm(x - x_accurate) / norm(x_accurate)
figure;
plot(x, "b-");
hold on;
plot(x_accurate, "r+");
legend({"x from H-Cholesky", "x accurate"});
savefig("x.jpg");
